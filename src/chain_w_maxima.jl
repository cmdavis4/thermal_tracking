using ProgressMeter: Progress, next!

# ──────────────────────────────────────────────────────────────────
# Internal dataclasses
# ──────────────────────────────────────────────────────────────────

struct WMaximum
	ix::Int            # globally unique per run
	time_ix::Int
	time::DateTime
	t_minutes::Float64
	x::Float64
	y::Float64
	z::Float64
	x_ix::Int
	y_ix::Int
	z_ix::Int
	w::Float64
	u::Float64
	v::Float64
end

mutable struct WMaxCluster
	w_maxima::Vector{WMaximum}
end

struct _Match
	from::WMaximum   # "maximum1" in Python
	to::WMaximum     # "maximum2"
	distance::Float64
end

# ──────────────────────────────────────────────────────────────────
# Per-timestep peak extraction
# ──────────────────────────────────────────────────────────────────

function _find_w_maxima_at_time(
	rams_path::String,
	time_ix::Int,
	time::DateTime,
	t_minutes::Float64,
	x_coords::Vector{Float64},
	y_coords::Vector{Float64},
	z_coords::Vector{Float64};
	min_distance_pixels::Int,
	threshold_abs::Real,
	ix_counter::Ref{Int},
)::Vector{WMaximum}
	wc, uc, vc = NCDataset(rams_path, "r") do ds
		(
			Float32.(ds["WC"][:, :, :]),   # Julia shape (nx, ny, nz)
			Float32.(ds["UC"][:, :, :]),
			Float32.(ds["VC"][:, :, :]),
		)
	end

	# local_maxima_3d returns (i, j, k) == (ix_x, ix_y, ix_z) matching array layout
	peaks = local_maxima_3d(
		wc;
		min_distance=min_distance_pixels,
		threshold_abs=threshold_abs,
	)

	maxima = Vector{WMaximum}(undef, length(peaks))
	@inbounds for (p_ix, (ix_x, ix_y, ix_z)) in enumerate(peaks)
		ix_counter[] += 1
		maxima[p_ix] = WMaximum(
			ix_counter[],
			time_ix,
			time,
			t_minutes,
			x_coords[ix_x],
			y_coords[ix_y],
			z_coords[ix_z],
			ix_x,
			ix_y,
			ix_z,
			Float64(wc[ix_x, ix_y, ix_z]),
			Float64(uc[ix_x, ix_y, ix_z]),
			Float64(vc[ix_x, ix_y, ix_z]),
		)
	end
	return maxima
end

# ──────────────────────────────────────────────────────────────────
# Chaining across time
# ──────────────────────────────────────────────────────────────────

function _chain_step!(
	w_clusters::Vector{WMaxCluster},
	last_maxima::Vector{WMaximum},
	current_maxima::Vector{WMaximum},
	timestep_seconds::Float64;
	filter_by_horizontal_distance::Bool=true,
	max_opposite_wind_displacement::Real=0.0,
	peak_position_tolerance::Real=0.0,
)
	# Build candidate lists per side, keyed by maximum ix.
	current_cands = Dict{Int, Vector{_Match}}()
	last_cands = Dict{Int, Vector{_Match}}()

	pos_tol = Float64(peak_position_tolerance)

	for cm in current_maxima, lm in last_maxima
		# z candidate: current must be above last, and both within the 1.5·w·Δt
		# band. `pos_tol` slackens each bound to absorb peak-detection wobble
		# (e.g. when the local W max jumps between adjacent cells within a
		# broad updraft core, the detected position can shift by several grid
		# spacings between consecutive timesteps).
		pair_is_z_candidate =
			(cm.z >= lm.z - pos_tol) &&
			((cm.z - cm.w * timestep_seconds * 1.5) <= lm.z + pos_tol) &&
			((lm.z + lm.w * timestep_seconds * 1.5) >= cm.z - pos_tol)
		if filter_by_horizontal_distance
			# Direction-aware: displacement must either point the same way as
			# the mean wind on that axis (and be within 1.5·|wind|·Δt at both
			# endpoints, plus `pos_tol` wobble slack), OR be small enough to
			# fit within `max_opposite_wind_displacement` — the latter allows
			# for weak wind / turbulence where the snapshot-to-snapshot
			# displacement may be small but counter to the local wind.
			dx = cm.x - lm.x
			dy = cm.y - lm.y
			mowd_tol = Float64(max_opposite_wind_displacement)
			x_ok =
				(abs(dx) <= mowd_tol) || (
					(dx * (cm.u + lm.u) >= 0) &&
					(abs(dx) <= abs(cm.u) * timestep_seconds * 1.5 + pos_tol) &&
					(abs(dx) <= abs(lm.u) * timestep_seconds * 1.5 + pos_tol)
				)
			y_ok =
				(abs(dy) <= mowd_tol) || (
					(dy * (cm.v + lm.v) >= 0) &&
					(abs(dy) <= abs(cm.v) * timestep_seconds * 1.5 + pos_tol) &&
					(abs(dy) <= abs(lm.v) * timestep_seconds * 1.5 + pos_tol)
				)
			pair_is_z_candidate = pair_is_z_candidate && x_ok && y_ok
		end
		pair_is_z_candidate || continue

		# distance with a half-timestep forward extrapolation from `last`, mirror from `current`
		dz_last = lm.z + 0.5 * timestep_seconds * lm.w - cm.z
		dy_last = lm.y - cm.y
		dx_last = lm.x - cm.x
		d_last = sqrt(dx_last^2 + dy_last^2 + dz_last^2)

		dz_curr = cm.z - 0.5 * timestep_seconds * cm.w - lm.z
		dy_curr = cm.y - lm.y
		dx_curr = cm.x - lm.x
		d_curr = sqrt(dx_curr^2 + dy_curr^2 + dz_curr^2)

		push!(get!(last_cands, lm.ix, _Match[]), _Match(lm, cm, d_last))
		push!(get!(current_cands, cm.ix, _Match[]), _Match(cm, lm, d_curr))
	end

	best_match(d) = Dict(ix => argmin(m -> m.distance, cands) for (ix, cands) in d)
	current_best = best_match(current_cands)
	last_best = best_match(last_cands)

	matches = _Match[]
	for (_, lbm) in last_best
		cbm = get(current_best, lbm.to.ix, nothing)
		cbm === nothing && continue
		if cbm.to.ix == lbm.from.ix
			push!(matches, lbm)
		end
	end

	# Index clusters by the ix of every maximum in them for O(1) lookup
	cluster_by_wm_ix = Dict{Int, WMaxCluster}()
	for c in w_clusters, wm in c.w_maxima
		cluster_by_wm_ix[wm.ix] = c
	end

	matched_current_ixs = Set{Int}()
	for m in matches
		parent = cluster_by_wm_ix[m.from.ix]
		push!(parent.w_maxima, m.to)
		push!(matched_current_ixs, m.to.ix)
	end

	# Any unmatched current maximum starts a new cluster
	for cm in current_maxima
		if !(cm.ix in matched_current_ixs)
			push!(w_clusters, WMaxCluster([cm]))
		end
	end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
	chain_w_maxima(; rams_output_dir, rams_grid_path, output_path,
					 min_distance, threshold_abs,
					 peak_position_tolerance)

Find updraft (`WC`) maxima at each RAMS timestep and chain them across time into
clusters. Writes a flat per-peak table to `output_path`.

- `min_distance` is in meters (matches Python default 500).
- `threshold_abs` is in m/s (matches Python default 0.8).
- `peak_position_tolerance` (m) widens the z and (when
  `filter_by_horizontal_distance`) the xy band checks by this amount on each
  axis, and relaxes the strict `cm.z >= lm.z` monotonicity by the same. Useful
  for absorbing peak-detection wobble when the detected W max jumps between
  nearby grid cells within a broad updraft core. Default `0.0` preserves the
  pre-tolerance behavior.
"""
function chain_w_maxima(;
	rams_output_dir::AbstractString,
	rams_grid_path::AbstractString,
	output_path::AbstractString,
	min_distance::Real=500.0,
	threshold_abs::Real=0.8,
	limit::Union{Integer, Nothing}=nothing,
	filter_by_horizontal_distance::Bool=true,
	max_opposite_wind_displacement::Real=0.0,
	peak_position_tolerance::Real=0.0,
)
	println("Listing RAMS files in $rams_output_dir")
	files = list_rams_files(String(rams_output_dir))
	if !isnothing(limit)
		files = files[1:limit]
	end
	isempty(files) && error("No RAMS files found in $rams_output_dir")
	println("  Found $(length(files)) RAMS files")

	println("Reading RAMS grid from $rams_grid_path")
	gc = read_rams_grid(String(rams_grid_path))

	# Grid spacings — Python uses xy spacing for `min_distance` conversion
	dx = gc.x[2] - gc.x[1]
	min_distance_pixels = floor(Int, Float64(min_distance) / Float64(dx))
	println("  Grid dx=$(dx) m → min_distance=$(min_distance_pixels) pixels")

	# Verify dim ordering before the main loop
	verify_rams_dims(files[1], gc.x, gc.y, gc.z, "WC")

	# Time step in seconds (from the first two files)
	t0 = rams_path_to_datetime(files[1])
	t1 = rams_path_to_datetime(files[2])
	timestep_seconds = Float64(Dates.value(Millisecond(t1 - t0))) / 1000.0
	println("  Timestep: $(timestep_seconds) s")

	w_clusters = WMaxCluster[]
	ix_counter = Ref(0)
	last_maxima = WMaximum[]

	pbar = Progress(length(files); output=stdout)
	for (time_ix, path) in enumerate(files)
		time = rams_path_to_datetime(path)
		t_minutes = Float64(Dates.value(Millisecond(time - t0))) / 60_000.0

		current_maxima = _find_w_maxima_at_time(
			path,
			time_ix,
			time,
			t_minutes,
			gc.x,
			gc.y,
			gc.z;
			min_distance_pixels=min_distance_pixels,
			threshold_abs=threshold_abs,
			ix_counter=ix_counter,
		)

		if time_ix == 1
			for wm in current_maxima
				push!(w_clusters, WMaxCluster([wm]))
			end
		else
			_chain_step!(
				w_clusters,
				last_maxima,
				current_maxima,
				timestep_seconds;
				filter_by_horizontal_distance=filter_by_horizontal_distance,
				max_opposite_wind_displacement=max_opposite_wind_displacement,
				peak_position_tolerance=peak_position_tolerance,
			)
		end

		last_maxima = current_maxima
		next!(pbar)
	end

	n_clusters = length(w_clusters)
	n_peaks = sum(length(c.w_maxima) for c in w_clusters; init=0)
	println("Found $n_clusters clusters, $n_peaks total peaks")

	_write_raw_wmax_clusters(String(output_path), w_clusters)
	println("Wrote $(output_path)")
	return nothing
end

# ──────────────────────────────────────────────────────────────────
# Flat-table output
# ──────────────────────────────────────────────────────────────────

function _write_raw_wmax_clusters(path::String, clusters::Vector{WMaxCluster})
	n_peaks = sum(length(c.w_maxima) for c in clusters; init=0)
	cluster_id = Vector{Int}(undef, n_peaks)
	seq_ix = Vector{Int}(undef, n_peaks)
	time_ix = Vector{Int}(undef, n_peaks)
	t_minutes = Vector{Float64}(undef, n_peaks)
	x = Vector{Float64}(undef, n_peaks)
	y = Vector{Float64}(undef, n_peaks)
	z = Vector{Float64}(undef, n_peaks)
	x_ix = Vector{Int}(undef, n_peaks)
	y_ix = Vector{Int}(undef, n_peaks)
	z_ix = Vector{Int}(undef, n_peaks)
	w = Vector{Float64}(undef, n_peaks)
	u = Vector{Float64}(undef, n_peaks)
	v = Vector{Float64}(undef, n_peaks)
	times = Vector{DateTime}(undef, n_peaks)

	k = 0
	for (cid, cluster) in enumerate(clusters)
		for (si, wm) in enumerate(cluster.w_maxima)
			k += 1
			cluster_id[k] = cid - 1   # 0-based to match Python convention
			seq_ix[k] = si - 1
			time_ix[k] = wm.time_ix - 1
			t_minutes[k] = wm.t_minutes
			x[k] = wm.x
			y[k] = wm.y
			z[k] = wm.z
			x_ix[k] = wm.x_ix - 1
			y_ix[k] = wm.y_ix - 1
			z_ix[k] = wm.z_ix - 1
			w[k] = wm.w
			u[k] = wm.u
			v[k] = wm.v
			times[k] = wm.time
		end
	end

	ref_time = DateTime(2000, 1, 1)
	time_seconds =
		Float64[Dates.value(Millisecond(t - ref_time)) / 1000.0 for t in times]

	isfile(path) && rm(path)
	NCDataset(path, "c") do ds
		defDim(ds, "peak_ix", n_peaks)
		defVar(ds, "cluster_id", cluster_id, ("peak_ix",))
		defVar(ds, "seq_ix", seq_ix, ("peak_ix",))
		defVar(ds, "time_ix", time_ix, ("peak_ix",))
		defVar(ds, "t_minutes", t_minutes, ("peak_ix",))
		defVar(ds, "x", x, ("peak_ix",))
		defVar(ds, "y", y, ("peak_ix",))
		defVar(ds, "z", z, ("peak_ix",))
		defVar(ds, "x_ix", x_ix, ("peak_ix",))
		defVar(ds, "y_ix", y_ix, ("peak_ix",))
		defVar(ds, "z_ix", z_ix, ("peak_ix",))
		defVar(ds, "w", w, ("peak_ix",))
		defVar(ds, "u", u, ("peak_ix",))
		defVar(ds, "v", v, ("peak_ix",))
		v = defVar(
			ds,
			"time",
			Float64,
			("peak_ix",);
			attrib=Dict(
				"units" => "seconds since 2000-01-01 00:00:00",
				"calendar" => "standard",
			),
		)
		v[:] = time_seconds
	end
end
