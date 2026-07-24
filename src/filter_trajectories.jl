# ──────────────────────────────────────────────────────────────────
# Kinematic quality filtering of thermal trajectories
# ──────────────────────────────────────────────────────────────────
#
# A second, purely-kinematic quality stage that runs AFTER smooth_trajectories. Each
# smoothed trajectory is scored on a handful of markers of "is this a real, coherent
# thermal?", and the ones that fail are dropped. The survivors are written to a separate
# file (raw_thermal_trajectories.nc → filtered_thermal_trajectories.nc), preserving the
# original thermal_ix labels and the full time axis, so the result is a schema-identical
# drop-in input for compute_thermal_shapes. Every threshold accepts `nothing` to skip that
# individual filter (all five `nothing` ⇒ a pass-through copy).
#
# The five markers (mirroring the Python thermal_quality_diagnostics notebook):
#   persistent       lifetime_seconds        ≥ threshold   (lives more than a few frames)
#   strong updraft   peak_w_max              ≥ threshold   (a real updraft forms)
#   rises rel. self  median(w_max − z_prop)  ≥ threshold   (core rises relative to the thermal)
#   ascends          ascent_fraction         ≥ threshold   (center climbs steadily)
#   smooth speed     prop_speed_jitter       ≤ threshold   (propagation speed evolves smoothly)

# median/mean/std come from the module-level `using Statistics`; std uses corrected=false
# (population) below to match numpy's nanstd, so the jitter threshold calibrates the same way.

# ──────────────────────────────────────────────────────────────────
# Per-trajectory metrics
# ──────────────────────────────────────────────────────────────────

"""
    compute_trajectory_quality_metrics(thermal_trajectories_path)

Compute per-trajectory kinematic quality metrics from a (thermal_ix, time) trajectory
dataset. Returns a NamedTuple of vectors, one entry per thermal in the file's `thermal_ix`
order:

- `thermal_ix`         — id label (preserved from the file).
- `n_steps`            — number of valid timesteps.
- `lifetime_seconds`   — span of valid times (persistence).
- `peak_w_max`         — strongest updraft over the life (m/s).
- `median_wtr_at_wmax` — median thermal-relative motion at the peak, `median(w_max − z_prop)`.
- `ascent_fraction`    — fraction of consecutive steps whose smoothed center rises.
- `prop_speed_jitter`  — `std(Δz_prop)/std(z_prop)`; <1 smooth, ≈1.41 step-to-step noise
                         (NaN when a track has <4 steps or a flat propagation speed).
"""
function compute_trajectory_quality_metrics(thermal_trajectories_path::AbstractString)
	NCDataset(String(thermal_trajectories_path), "r") do ds
		thermal_ix = Int.(Array(ds["thermal_ix"]))
		# 2-D variables are (time, thermal_ix) in Julia; collapse fill → NaN like the pipeline.
		w_max_2d = coalesce.(Array(ds["w_max"]), NaN)
		z_smoothed_2d = coalesce.(Array(ds["z_smoothed"]), NaN)
		z_prop_2d = coalesce.(Array(ds["z_propagation_speed"]), NaN)
		t_seconds_2d = coalesce.(Array(ds["t_seconds"]), NaN)

		n_thermals = length(thermal_ix)
		n_steps = zeros(Int, n_thermals)
		lifetime_seconds = fill(NaN, n_thermals)
		peak_w_max = fill(NaN, n_thermals)
		median_wtr_at_wmax = fill(NaN, n_thermals)
		ascent_fraction = fill(NaN, n_thermals)
		prop_speed_jitter = fill(NaN, n_thermals)

		for ti in 1:n_thermals
			# Active timesteps for this thermal = rows with a finite w_max. The file's time
			# axis is ascending, so row order is already chronological.
			active = findall(isfinite, @view w_max_2d[:, ti])
			n_steps[ti] = length(active)
			isempty(active) && continue

			t = t_seconds_2d[active, ti]
			w = w_max_2d[active, ti]
			z_center = z_smoothed_2d[active, ti]
			z_prop = z_prop_2d[active, ti]

			# Persistence and strength.
			lifetime_seconds[ti] = maximum(t) - minimum(t)
			peak_w_max[ti] = maximum(w)
			# Thermal-relative motion at the tracked peak.
			median_wtr_at_wmax[ti] = median(w .- z_prop)

			# Ascent coherence: fraction of steps whose smoothed center rises.
			if length(active) >= 2
				ascent_fraction[ti] = mean(diff(z_center) .> 0)
			end

			# Propagation-speed smoothness: increment scatter vs the series' own spread.
			# Needs a few points and a non-flat series (population std, corrected=false).
			if length(active) >= 4
				z_prop_spread = std(z_prop; corrected = false)
				if z_prop_spread > 0
					prop_speed_jitter[ti] =
						std(diff(z_prop); corrected = false) / z_prop_spread
				end
			end
		end

		return (
			thermal_ix = thermal_ix,
			n_steps = n_steps,
			lifetime_seconds = lifetime_seconds,
			peak_w_max = peak_w_max,
			median_wtr_at_wmax = median_wtr_at_wmax,
			ascent_fraction = ascent_fraction,
			prop_speed_jitter = prop_speed_jitter,
		)
	end
end

# ──────────────────────────────────────────────────────────────────
# Threshold → per-thermal pass masks
# ──────────────────────────────────────────────────────────────────

# Per-thermal pass mask for a "value ≥ threshold" filter; a `nothing` threshold passes
# everything, and a NaN (unevaluable) value fails. `_pass_at_most` is the "≤" analogue.
function _pass_at_least(values::Vector{Float64}, threshold::Union{Nothing, Real})
	threshold === nothing && return trues(length(values))
	return [isfinite(v) && v >= threshold for v in values]
end
function _pass_at_most(values::Vector{Float64}, threshold::Union{Nothing, Real})
	threshold === nothing && return trues(length(values))
	return [isfinite(v) && v <= threshold for v in values]
end

"""
    trajectory_quality_masks(metrics; min_lifetime_seconds=nothing, min_peak_w_max=nothing,
                             min_median_wtr_at_wmax=nothing, min_ascent_fraction=nothing,
                             max_prop_speed_jitter=nothing)

Turn quality `metrics` (from `compute_trajectory_quality_metrics`) plus thresholds into
per-thermal boolean pass masks: one per marker plus `passes_all` (their logical AND). A
`nothing` threshold disables that marker (its mask is all-true). Shared by the filter and
the notebook's diagnostic plots so the two never drift.
"""
function trajectory_quality_masks(
	metrics;
	min_lifetime_seconds::Union{Nothing, Real} = nothing,
	min_peak_w_max::Union{Nothing, Real} = nothing,
	min_median_wtr_at_wmax::Union{Nothing, Real} = nothing,
	min_ascent_fraction::Union{Nothing, Real} = nothing,
	max_prop_speed_jitter::Union{Nothing, Real} = nothing,
)
	persistent = _pass_at_least(metrics.lifetime_seconds, min_lifetime_seconds)
	strong_updraft = _pass_at_least(metrics.peak_w_max, min_peak_w_max)
	rises_rel_self = _pass_at_least(metrics.median_wtr_at_wmax, min_median_wtr_at_wmax)
	ascends = _pass_at_least(metrics.ascent_fraction, min_ascent_fraction)
	smooth_speed = _pass_at_most(metrics.prop_speed_jitter, max_prop_speed_jitter)
	passes_all = persistent .& strong_updraft .& rises_rel_self .& ascends .& smooth_speed
	return (
		persistent = persistent,
		strong_updraft = strong_updraft,
		rises_rel_self = rises_rel_self,
		ascends = ascends,
		smooth_speed = smooth_speed,
		passes_all = passes_all,
	)
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
    filter_thermal_trajectories(; raw_thermal_trajectories_path, output_path,
                                 min_lifetime_seconds=nothing, min_peak_w_max=nothing,
                                 min_median_wtr_at_wmax=nothing, min_ascent_fraction=nothing,
                                 max_prop_speed_jitter=nothing)

Score each smoothed trajectory on the kinematic quality markers and write only the
trajectories that pass every *applied* filter to `output_path`. Original `thermal_ix`
labels and the full time axis are preserved, so the output is a drop-in replacement input
for `compute_thermal_shapes`. Any threshold left as `nothing` disables that one filter;
all five `nothing` gives a straight pass-through copy.
"""
function filter_thermal_trajectories(;
	raw_thermal_trajectories_path::AbstractString,
	output_path::AbstractString,
	min_lifetime_seconds::Union{Nothing, Real} = nothing,
	min_peak_w_max::Union{Nothing, Real} = nothing,
	min_median_wtr_at_wmax::Union{Nothing, Real} = nothing,
	min_ascent_fraction::Union{Nothing, Real} = nothing,
	max_prop_speed_jitter::Union{Nothing, Real} = nothing,
)
	raw_path = String(raw_thermal_trajectories_path)
	println("Reading trajectories from $raw_path")
	metrics = compute_trajectory_quality_metrics(raw_path)
	n_thermals = length(metrics.thermal_ix)

	masks = trajectory_quality_masks(
		metrics;
		min_lifetime_seconds = min_lifetime_seconds,
		min_peak_w_max = min_peak_w_max,
		min_median_wtr_at_wmax = min_median_wtr_at_wmax,
		min_ascent_fraction = min_ascent_fraction,
		max_prop_speed_jitter = max_prop_speed_jitter,
	)

	# Report each filter's outcome (or that it was skipped) and the combined survival.
	report(name, threshold, mask) =
		threshold === nothing ? println("  $name: (skipped)") :
		println("  $name (threshold $threshold): $(count(mask))/$n_thermals pass")
	report("persistent     ", min_lifetime_seconds, masks.persistent)
	report("strong updraft ", min_peak_w_max, masks.strong_updraft)
	report("rises rel. self", min_median_wtr_at_wmax, masks.rises_rel_self)
	report("ascends        ", min_ascent_fraction, masks.ascends)
	report("smooth speed   ", max_prop_speed_jitter, masks.smooth_speed)
	println("  → $(count(masks.passes_all))/$n_thermals thermals pass all applied filters")

	keep_positions = findall(masks.passes_all)
	_subset_thermal_trajectories(raw_path, String(output_path), keep_positions)
	println("Wrote $(output_path)")
	return nothing
end

# ──────────────────────────────────────────────────────────────────
# Column subset writer (preserves labels, time axis, fill values, attrs)
# ──────────────────────────────────────────────────────────────────

"""
Copy `raw_path` to `output_path`, keeping only the `thermal_ix` positions in
`keep_positions` (1-based). Variables carrying the `thermal_ix` dimension are subset along
it; everything else (the `time` axis and its coordinate) is copied whole. Original
`thermal_ix` labels, per-variable fill values, and attributes are preserved, so the result
is schema-identical to the input trajectory dataset.
"""
function _subset_thermal_trajectories(
	raw_path::String,
	output_path::String,
	keep_positions::Vector{Int},
)
	isfile(output_path) && rm(output_path)
	NCDataset(raw_path, "r") do src
		NCDataset(output_path, "c") do dst
			# Recreate dimensions: thermal_ix shrinks to the kept count; others unchanged.
			for dname in keys(src.dim)
				defDim(
					dst,
					dname,
					dname == "thermal_ix" ? length(keep_positions) : src.dim[dname],
				)
			end
			for varname in keys(src)
				srcvar = src[varname]
				dims = dimnames(srcvar)
				# Keep all attributes except _FillValue, which is re-applied via `fillvalue`.
				attrs = Dict(k => v for (k, v) in srcvar.attrib if k != "_FillValue")
				fillval =
					haskey(srcvar.attrib, "_FillValue") ? srcvar.attrib["_FillValue"] :
					nothing
				# Read raw (undecoded) values so CF-encoded time is copied verbatim, with the
				# stored fill sentinel (NaN / -1) already in place.
				data = Array(srcvar.var)
				if "thermal_ix" in dims
					axis = findfirst(==("thermal_ix"), dims)
					data = copy(selectdim(data, axis, keep_positions))
				end
				if fillval === nothing
					defVar(dst, varname, data, dims; attrib = attrs)
				else
					defVar(dst, varname, data, dims; attrib = attrs, fillvalue = fillval)
				end
			end
		end
	end
end
