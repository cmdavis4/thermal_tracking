using Interpolations: interpolate, Gridded, Linear, extrapolate
using ProgressMeter: Progress, next!

# ──────────────────────────────────────────────────────────────────
# Spherical grid
# ──────────────────────────────────────────────────────────────────

struct SphericalGrid
	r_vals::Vector{Float64}
	theta_vals::Vector{Float64}
	phi_vals::Vector{Float64}
	n_r::Int
	n_theta::Int
	n_phi::Int
	# Precomputed trig arrays, shaped for broadcasting to (n_r, n_theta, n_phi).
	R::Array{Float64, 3}
	sin_theta_cos_phi::Array{Float64, 3}
	sin_theta_sin_phi::Array{Float64, 3}
	cos_theta::Array{Float64, 3}
end

function build_spherical_grid(n_r, n_theta, n_phi, r_norm_max)::SphericalGrid
	r_vals = collect(range(0, r_norm_max, length=n_r))
	theta_vals = collect(range(0, π, length=n_theta))
	phi_vals = collect(range(0, step=2π / n_phi, length=n_phi))

	Θ = reshape(theta_vals, 1, :, 1)
	Φ = reshape(phi_vals, 1, 1, :)

	SphericalGrid(
		r_vals,
		theta_vals,
		phi_vals,
		n_r,
		n_theta,
		n_phi,
		reshape(r_vals, :, 1, 1),
		sin.(Θ) .* cos.(Φ),
		sin.(Θ) .* sin.(Φ),
		cos.(Θ),
	)
end

# ──────────────────────────────────────────────────────────────────
# Thermal track reader
# ──────────────────────────────────────────────────────────────────

"""
Read thermal tracking dataset.

Returns `(tracks, extras, extra_names, extra_attrs)`:
- `tracks::Vector{ThermalTrack}` — one per source thermal with at least one valid timestep.
- `extras::Dict{String, Vector{Vector{Float64}}}` — for each auto-discovered
  (thermal_ix, time)-shaped variable, a per-track vector of per-timestep values
  in lock-step with `tracks[i].timesteps`.
- `extra_names::Vector{String}` — stable iteration order for `extras`.
- `extra_attrs::Dict{String, Dict{String, Any}}` — NetCDF attributes copied from
  the source, minus fill/missing markers that the writer will set itself.

NCDatasets reverses dimension order from Python/ncdump convention.
For thermal_ds.nc with Python dims (thermal_ix, time), Julia sees (time, thermal_ix).
"""
function read_thermal_tracks(path::String; scale_var::String)
	NCDataset(path, "r") do ds
		thermal_ixs = Array(ds["thermal_ix"])
		time_vals = Array(ds["time"])

		n_thermals = length(thermal_ixs)
		n_time = length(time_vals)

		# Harmonized with the fit stage: composite around the SMOOTHED center and scale by
		# the chosen (smoothed) sphere radius; `scale_var` selects which sphere method.
		x_all = Array(ds["x_smoothed"])
		y_all = Array(ds["y_smoothed"])
		z_all = Array(ds["z_smoothed"])
		sr_all = Array(ds[scale_var])
		dt_all = Array(ds["dt_thermal_seconds"])

		data_size = size(x_all)
		if data_size == (n_time, n_thermals)
			time_first = true
		elseif data_size == (n_thermals, n_time)
			time_first = false
		else
			error(
				"Cannot determine dim order: data size $data_size vs " *
				"n_time=$n_time, n_thermals=$n_thermals",
			)
		end

		get_val(arr, di, ti) = time_first ? arr[di, ti] : arr[ti, di]

		# Variables consumed directly (struct fields or output coords) — don't
		# double-write them as propagated extras.
		consumed = Set([
			"x",
			"y",
			"z",
			"dt_thermal_seconds",
			"time",
			"thermal_ix",
		])
		# Attributes the writer will set explicitly; copying them would clash.
		skip_attrs = Set(["_FillValue", "missing_value"])

		extra_names = String[]
		extra_raw = Dict{String, Any}()
		extra_attrs = Dict{String, Dict{String, Any}}()
		for name in keys(ds)
			name in consumed && continue
			ncvar = ds[name]
			sz = size(ncvar)
			(sz == (n_time, n_thermals) || sz == (n_thermals, n_time)) ||
				continue
			push!(extra_names, name)
			extra_raw[name] = Array(ncvar)
			extra_attrs[name] = Dict{String, Any}(
				k => ncvar.attrib[k] for
				k in keys(ncvar.attrib) if !(k in skip_attrs)
			)
		end

		tracks = ThermalTrack[]
		extras = Dict(name => Vector{Vector{Float64}}() for name in extra_names)
		for ti in 1:n_thermals
			timesteps = ThermalTimeStep[]
			per_track_extras = Dict(name => Float64[] for name in extra_names)
			for di in 1:n_time
				# Every field of a ThermalTimeStep must be a finite Float64. A timestep
				# with no smoothed center (x/y/z), no timestep spacing (dt), or no sphere
				# radius can't be composited, so skip it. The smoothed center can go
				# missing on a timestep where sr survives (e.g. edge-of-domain / short
				# tracks the smoothing stage couldn't fill in), so guard every field, not
				# just sr — otherwise Float64(::Missing) throws below.
				sr = get_val(sr_all, di, ti)
				x_val = get_val(x_all, di, ti)
				y_val = get_val(y_all, di, ti)
				z_val = get_val(z_all, di, ti)
				dt_val = get_val(dt_all, di, ti)
				any_missing_or_nan =
					ismissing(sr) || isnan(Float64(sr)) ||
					ismissing(x_val) || ismissing(y_val) ||
					ismissing(z_val) || ismissing(dt_val)
				any_missing_or_nan && continue
				push!(
					timesteps,
					ThermalTimeStep(
						Float64(x_val),
						Float64(y_val),
						Float64(z_val),
						Float64(sr),
						Float64(dt_val),
						time_vals[di],
					),
				)
				for name in extra_names
					val = get_val(extra_raw[name], di, ti)
					push!(
						per_track_extras[name],
						ismissing(val) ? NaN : Float64(val),
					)
				end
			end
			isempty(timesteps) && continue
			push!(tracks, ThermalTrack(Int(thermal_ixs[ti]), timesteps))
			for name in extra_names
				push!(extras[name], per_track_extras[name])
			end
		end
		return (
			tracks=tracks,
			extras=extras,
			extra_names=extra_names,
			extra_attrs=extra_attrs,
		)
	end
end

# ──────────────────────────────────────────────────────────────────
# Interpolation
# ──────────────────────────────────────────────────────────────────

# RAMS Arakawa-C staggering: which axis (if any) a variable is offset on. U/UP/UC live on
# the x-momentum levels, V/VP/VC on y, W/WP/WC on z; everything else is a thermodynamic
# (scalar) field co-located with the grid centers. Extend these sets if you composite other
# staggered variables.
const _X_STAGGERED_VARS = Set(["U", "UP", "UC"])
const _Y_STAGGERED_VARS = Set(["V", "VP", "VC"])
const _Z_STAGGERED_VARS = Set(["W", "WP", "WC"])

"""Axis (`:x`, `:y`, `:z`, or `:none`) that RAMS variable `varname` is C-grid–staggered on."""
function _staggered_axis(varname::AbstractString)::Symbol
	varname in _X_STAGGERED_VARS && return :x
	varname in _Y_STAGGERED_VARS && return :y
	varname in _Z_STAGGERED_VARS && return :z
	return :none
end

"""Common Arakawa-C staggering axis shared by all of `spec`'s inputs (errors if they differ)."""
function _spec_staggered_axis(spec::CompositeSpec)::Symbol
	axes = unique(_staggered_axis(v) for v in spec.inputs)
	length(axes) == 1 || error(
		"Composite '$(spec.name)' mixes Arakawa-C staggering across its inputs " *
		"($(join(spec.inputs, ", ")) → $(axes)); all inputs must share one axis so they " *
		"can be combined pointwise on a single grid.",
	)
	return only(axes)
end

function interpolate_timestep!(
	result::Dict{String, Array{Float64, 3}},
	rams_path::String,
	specs::Vector{CompositeSpec},
	subset::NamedTuple,
	grid::SphericalGrid,
	ts::ThermalTimeStep,
)
	rho = grid.R .* ts.sphere_radius
	z_q = rho .* grid.cos_theta .+ ts.center_z
	x_q = rho .* grid.sin_theta_cos_phi .+ ts.center_x
	y_q = rho .* grid.sin_theta_sin_phi .+ ts.center_y

	NCDataset(rams_path, "r") do ds
		for spec in specs
			# Sample staggered variables on their momentum axis and scalars on the thermo
			# grid; only the staggered axis differs from the scalar coordinates. This places
			# each input at its true C-grid location before interpolating to (r, θ, φ).
			staggered_axis = _spec_staggered_axis(spec)
			x_coords = staggered_axis === :x ? subset.xm_sub : subset.x_sub
			y_coords = staggered_axis === :y ? subset.ym_sub : subset.y_sub
			z_coords = staggered_axis === :z ? subset.zm_sub : subset.z_sub

			# Read each input on the native subset grid, then combine into the field to
			# interpolate. For a plain pass-through variable this is just the variable itself;
			# for a derived variable the combination happens here, on the native grid.
			native = Dict{String, Array{Float64, 3}}(
				v => Float64.(ds[v][subset.x_ixs, subset.y_ixs, subset.z_ixs])
				for v in spec.inputs
			)
			data_sub = spec.combine(native)

			itp = interpolate(
				(x_coords, y_coords, z_coords),
				data_sub,
				Gridded(Linear()),
			)
			eitp = extrapolate(itp, NaN)

			result[spec.name] .= eitp.(x_q, y_q, z_q)
		end
	end
end

"""
Reduce an interpolated `(r, θ, φ)` field to its azimuthal (φ) mean and standard deviation,
returning two `(r, θ)` arrays. NaN samples (points that fell outside the RAMS domain) are
ignored, so partial-coverage shells still get a mean over their valid φ; an `(r, θ)` point
with no valid φ becomes NaN. The std is the population std over the available φ samples and
serves as a per-`(r, θ)` measure of azimuthal asymmetry.
"""
function _azimuthal_mean_std(field::Array{Float64, 3})
	n_r, n_theta, n_phi = size(field)
	mean_rt = Array{Float64}(undef, n_r, n_theta)
	std_rt = Array{Float64}(undef, n_r, n_theta)
	@inbounds for j in 1:n_theta, i in 1:n_r
		# Pass 1: mean over the valid (non-NaN) φ samples.
		sum_val = 0.0
		n_valid = 0
		for k in 1:n_phi
			v = field[i, j, k]
			isnan(v) && continue
			sum_val += v
			n_valid += 1
		end
		if n_valid == 0
			mean_rt[i, j] = NaN
			std_rt[i, j] = NaN
			continue
		end
		m = sum_val / n_valid
		mean_rt[i, j] = m
		# Pass 2: population variance about the mean. The two-pass form avoids the
		# catastrophic cancellation of E[x²] − E[x]² for large-magnitude fields (e.g. number
		# concentrations), which would otherwise corrupt the low-order std digits.
		sum_sq_dev = 0.0
		for k in 1:n_phi
			v = field[i, j, k]
			isnan(v) && continue
			d = v - m
			sum_sq_dev += d * d
		end
		std_rt[i, j] = sqrt(sum_sq_dev / n_valid)
	end
	return mean_rt, std_rt
end

# ──────────────────────────────────────────────────────────────────
# Output writer
# ──────────────────────────────────────────────────────────────────

"""
Write the normalized dataset to NetCDF.

Julia defVar dim tuples are the reverse of Python/ncdump/CDL order. With the full
`(r, θ, φ)` grid the composited fields have Python dims
`(dt_thermal_seconds, thermal_ix, r_norm, theta, phi)` → defVar dims
`("phi", "theta", "r_norm", "thermal_ix", "dt_thermal_seconds")`. When `azimuthal_average`
is set the φ axis has already been collapsed, so the fields are 4D `(…, r_norm, theta)` and
each also gets a `<var>_phi_std` companion (when `output_std` is provided) holding the
azimuthal standard deviation as an anisotropy diagnostic.
"""
function _write_normalize_output(
	path::String,
	output::AbstractDict,
	grid::SphericalGrid,
	all_dt_seconds::Vector{Float64},
	tracks::Vector{ThermalTrack},
	time_seconds_2d::Matrix{Float64},
	composited_var_names::Vector{String},
	extras_2d::Dict{String, Matrix{Float64}},
	extra_names::Vector{String},
	extra_attrs::Dict{String, Dict{String, Any}};
	azimuthal_average::Bool,
	output_std::Union{Nothing, AbstractDict}=nothing,
)
	thermal_ixs = [t.thermal_ix for t in tracks]

	r2d = grid.r_vals * ones(length(grid.theta_vals))'
	t2d = ones(length(grid.r_vals)) * grid.theta_vals'
	x_norm = r2d .* sin.(t2d)
	z_norm = r2d .* cos.(t2d)

	NCDataset(path, "c") do ds
		defDim(ds, "dt_thermal_seconds", length(all_dt_seconds))
		defDim(ds, "thermal_ix", length(tracks))
		defDim(ds, "r_norm", grid.n_r)
		defDim(ds, "theta", grid.n_theta)
		# The φ axis only exists in the output when it wasn't averaged out.
		azimuthal_average || defDim(ds, "phi", grid.n_phi)

		defVar(
			ds,
			"dt_thermal_seconds",
			all_dt_seconds,
			("dt_thermal_seconds",),
		)
		defVar(ds, "thermal_ix", thermal_ixs, ("thermal_ix",))
		defVar(ds, "r_norm", grid.r_vals, ("r_norm",))
		defVar(ds, "theta", grid.theta_vals, ("theta",))
		azimuthal_average || defVar(ds, "phi", grid.phi_vals, ("phi",))

		v = defVar(
			ds,
			"time",
			Float64,
			("dt_thermal_seconds", "thermal_ix");
			fillvalue=NaN,
			attrib=Dict(
				"units" => "seconds since 2000-01-01 00:00:00",
				"calendar" => "standard",
			),
		)
		v[:, :] = time_seconds_2d

		for name in extra_names
			v = defVar(
				ds,
				name,
				Float64,
				("dt_thermal_seconds", "thermal_ix");
				fillvalue=NaN,
				attrib=extra_attrs[name],
			)
			v[:, :] = extras_2d[name]
		end

		defVar(ds, "x_norm", permutedims(x_norm), ("theta", "r_norm"))
		defVar(ds, "z_norm", permutedims(z_norm), ("theta", "r_norm"))

		# Composited-field dimensions in defVar (reversed) order, and the matching permutedims
		# order that maps our (dt, thermal, r, θ[, φ]) storage into it. Averaged fields drop φ.
		if azimuthal_average
			field_dims = ("theta", "r_norm", "thermal_ix", "dt_thermal_seconds")
			storage_to_defvar_perm = (4, 3, 2, 1)
		else
			field_dims =
				("phi", "theta", "r_norm", "thermal_ix", "dt_thermal_seconds")
			storage_to_defvar_perm = (5, 4, 3, 2, 1)
		end
		all_colons = ntuple(_ -> Colon(), length(field_dims))

		for var in composited_var_names
			println("  Writing $var...")
			v = defVar(ds, var, Float64, field_dims; fillvalue=NaN)
			v[all_colons...] = permutedims(output[var], storage_to_defvar_perm)

			# Azimuthal-asymmetry companion field (φ-std), same shape as the averaged field.
			if azimuthal_average && output_std !== nothing
				vstd = defVar(
					ds,
					"$(var)_phi_std",
					Float64,
					field_dims;
					fillvalue=NaN,
					attrib=Dict(
						"long_name" => "azimuthal (phi) standard deviation of $var",
					),
				)
				vstd[all_colons...] =
					permutedims(output_std[var], storage_to_defvar_perm)
			end
		end
	end
	println("Output written to $path")
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
	normalize_thermals(; thermal_ds_path, rams_output_dir, rams_grid_path, output_path,
						 composited_vars, n_r=20, n_theta=36, n_phi=72, r_norm_max=2.0,
						 lite=true, azimuthal_average=true, include_azimuthal_std=true)

Build a composite dataset in thermal-normalized spherical coordinates. For each
thermal at each timestep, linearly interpolates RAMS data from the Cartesian
grid onto an `(r_norm, theta, phi)` grid centered on the thermal where
`r_norm = 1` is the thermal edge. `r_norm_max` sets how far out (in thermal radii)
the grid extends; the Cartesian subset is padded to match, so the whole grid is
sampled (see the `compute_spatial_subset` call).

`composited_vars` is a vector whose entries are each either a RAMS variable name (a
`String`, interpolated as-is) or a `CompositeSpec` built with `derived`/`derived_sum`
(a quantity computed on the native RAMS grid before interpolation — see `CompositeSpec`).

`lite` selects which RAMS files to composite: lite (`a-L-`) or full-analysis (`a-A-`).
Only thermal_ds times with a matching RAMS file are composited (see the availability check).

`azimuthal_average` (default `true`) collapses the φ axis to its mean during accumulation,
so the composited fields are `(dt_thermal_seconds, thermal_ix, r_norm, theta)` — the
per-thermal azimuthal profiles, at 1/n_phi the memory of the full `(…, phi)` field. Set it
`false` to keep the full φ axis (e.g. to inspect the azimuthal structure itself), at the
cost of an n_phi× larger array. When averaging, `include_azimuthal_std` (default `true`)
additionally writes a `<var>_phi_std` field holding the azimuthal standard deviation, a
per-`(r, θ)` measure of how much azimuthal variation the mean is hiding.
"""
function normalize_thermals(;
	thermal_ds_path::AbstractString,
	rams_output_dir::AbstractString,
	rams_grid_path::AbstractString,
	output_path::AbstractString,
	composited_vars::AbstractVector,
	n_r::Int=20,
	n_theta::Int=36,
	n_phi::Int=72,
	r_norm_max::Real=2.0,
	lite::Bool=true,
	azimuthal_average::Bool=true,
	include_azimuthal_std::Bool=true,
	scale_var::AbstractString="sphere_radius_disk_smoothed",
)
	# Normalize each requested variable into a CompositeSpec: bare strings become pass-through
	# specs; CompositeSpec entries (from derived/derived_sum) are kept as-is.
	specs = CompositeSpec[
		el isa CompositeSpec ? el : composite(String(el)) for el in composited_vars
	]
	thermal_ds_path = String(thermal_ds_path)
	rams_output_dir = String(rams_output_dir)
	rams_grid_path = String(rams_grid_path)
	output_path = String(output_path)

	println("Reading thermal tracks from $thermal_ds_path")
	tracks, extras, extra_names, extra_attrs =
		read_thermal_tracks(thermal_ds_path; scale_var=String(scale_var))
	println("  Found $(length(tracks)) thermals")
	if !isempty(extra_names)
		println(
			"  Propagating $(length(extra_names)) extra variable(s): " *
			join(extra_names, ", "),
		)
	end

	println("Reading RAMS grid from $rams_grid_path")
	gc = read_rams_grid(rams_grid_path)

	grid = build_spherical_grid(n_r, n_theta, n_phi, Float64(r_norm_max))
	println(
		"  Spherical grid: r=$(grid.n_r) × θ=$(grid.n_theta) × φ=$(grid.n_phi)",
	)

	# Determine which thermal-track timesteps have a corresponding RAMS output file.
	# Tracking is typically run on high-frequency lite output, while normalization may be
	# run against lower-frequency full-analysis output (lite=false). In that case most
	# thermal_ds times have no matching RAMS file, so we composite only the times where
	# both the thermal tracks and the RAMS output are present. The existence check is done
	# once here (and cached) so the hot loop below neither re-stats files nor warns per-file.
	unique_track_times =
		sort(collect(Set(ts.time for track in tracks for ts in track.timesteps)))
	rams_file_available = Dict{DateTime, Bool}()
	for track_time in unique_track_times
		rams_file_available[track_time] =
			isfile(datetime_to_rams_path(rams_output_dir, track_time, lite=lite))
	end
	n_times_total = length(unique_track_times)
	n_times_available = count(values(rams_file_available))

	# Warn (once) if not every thermal_ds time has a matching RAMS file — the expected case
	# when normalizing high-frequency tracks against lower-frequency full-analysis output.
	if n_times_available < n_times_total
		file_kind = lite ? "lite (a-L-)" : "full-analysis (a-A-)"
		@warn "Only $n_times_available of $n_times_total thermal_ds times have a matching " *
			  "$file_kind RAMS file in $rams_output_dir; normalizing those times only " *
			  "(remaining $(n_times_total - n_times_available) time(s) skipped)."
	end
	if n_times_available == 0
		error(
			"None of the $n_times_total thermal_ds times have a matching " *
			"$(lite ? "lite (a-L-)" : "full-analysis (a-A-)") RAMS file in $rams_output_dir.",
		)
	end

	# Use the first available time (not necessarily the first track's first timestep, which
	# generally does not align with a full-analysis file) to discover variables and check dims.
	first_available_time = first(t for t in unique_track_times if rams_file_available[t])
	first_path =
		datetime_to_rams_path(rams_output_dir, first_available_time, lite=lite)

	available_vars = NCDataset(first_path, "r") do ds
		Set(keys(ds))
	end
	# Keep only specs whose every RAMS input is present in the file; a derived spec is dropped
	# whole if any of its inputs is missing (it can't be computed without all of them).
	kept_specs = CompositeSpec[]
	for spec in specs
		missing_inputs = [v for v in spec.inputs if !(v in available_vars)]
		if isempty(missing_inputs)
			push!(kept_specs, spec)
		else
			println(
				"  Skipping composite '$(spec.name)': input(s) not present in " *
				"$first_path: " * join(missing_inputs, ", "),
			)
		end
	end
	specs = kept_specs
	if isempty(specs)
		error(
			"None of the requested composited_vars can be built from $first_path",
		)
	end
	spec_names = [spec.name for spec in specs]

	verify_rams_dims(first_path, gc.x, gc.y, gc.z, specs[1].inputs[1])

	# Build the dt_thermal_seconds output axis only from timesteps whose RAMS file exists, so
	# times with no matching RAMS output don't create all-NaN slabs in the composite.
	all_dt_set = Set{Float64}()
	for track in tracks, ts in track.timesteps
		rams_file_available[ts.time] || continue
		push!(all_dt_set, ts.dt_thermal_seconds)
	end
	all_dt_seconds = sort(collect(all_dt_set))
	dt_to_ix = Dict(dt => ix for (ix, dt) in enumerate(all_dt_seconds))

	n_dt = length(all_dt_seconds)
	n_thermals = length(tracks)
	println(
		"  Output: dt=$n_dt × thermals=$n_thermals × " *
		"r=$(grid.n_r) × θ=$(grid.n_theta)" *
		(azimuthal_average ? " (φ averaged)" : " × φ=$(grid.n_phi)") *
		" × vars=$(length(specs))" *
		(azimuthal_average && include_azimuthal_std ? " (+φ-std)" : ""),
	)

	# Composited-field storage. When averaging over φ the fields are 4D (dt, thermal, r, θ)
	# and — if requested — an equally-shaped std field accumulates the azimuthal spread;
	# otherwise the fields keep the full φ axis (dt, thermal, r, θ, φ).
	if azimuthal_average
		output = Dict(
			name => fill(NaN, n_dt, n_thermals, grid.n_r, grid.n_theta)
			for name in spec_names
		)
		output_std =
			include_azimuthal_std ?
			Dict(
				name => fill(NaN, n_dt, n_thermals, grid.n_r, grid.n_theta)
				for name in spec_names
			) : nothing
	else
		output = Dict(
			name =>
				fill(NaN, n_dt, n_thermals, grid.n_r, grid.n_theta, grid.n_phi)
			for name in spec_names
		)
		output_std = nothing
	end
	extras_2d = Dict{String, Matrix{Float64}}(
		name => fill(NaN, n_dt, n_thermals) for name in extra_names
	)
	ref_time = DateTime(2000, 1, 1)
	time_seconds_2d = fill(NaN, n_dt, n_thermals)

	n_threads = Threads.nthreads()
	println("Processing thermals ($n_threads thread(s))...")
	pbar = Progress(n_thermals; output=stdout)
	pbar_lock = ReentrantLock()

	@tasks for thermal_slot in 1:n_thermals
		# Per-task scratch buffer for the full (r, θ, φ) interpolation of each field; the φ
		# reduction (if any) happens out of this buffer before storing into the output arrays.
		@local buf = Dict(
			name =>
				Array{Float64}(undef, grid.n_r, grid.n_theta, grid.n_phi)
			for name in spec_names
		)

		track = tracks[thermal_slot]
		# Pad the Cartesian bounding box out to the full normalized radius (r_norm_max × the
		# thermal radius), so every query point on the (r_norm ≤ r_norm_max) sphere lies inside
		# the subset. With a smaller pad the outer shells fall outside and interpolate to NaN.
		# (coord_range already adds one grid cell on each side for the interpolation stencil.)
		subset = compute_spatial_subset(
			track,
			gc.x,
			gc.y,
			gc.z;
			pad=Float64(r_norm_max),
			xm_coords=gc.xm,
			ym_coords=gc.ym,
			zm_coords=gc.zm,
		)

		for (ts_ix, ts) in enumerate(track.timesteps)
			# Skip timesteps with no matching RAMS file (already reported in the summary
			# warning above); this is the common case when normalizing high-frequency tracks
			# against lower-frequency full-analysis output.
			rams_file_available[ts.time] || continue
			rams_path =
				datetime_to_rams_path(rams_output_dir, ts.time, lite=lite)

			interpolate_timestep!(buf, rams_path, specs, subset, grid, ts)

			dt_ix = dt_to_ix[ts.dt_thermal_seconds]
			for name in spec_names
				if azimuthal_average
					# Collapse φ to its mean (and spread) so only the (r, θ) profile is stored.
					phi_mean, phi_std = _azimuthal_mean_std(buf[name])
					output[name][dt_ix, thermal_slot, :, :] .= phi_mean
					if include_azimuthal_std
						output_std[name][dt_ix, thermal_slot, :, :] .= phi_std
					end
				else
					output[name][dt_ix, thermal_slot, :, :, :] .= buf[name]
				end
			end
			time_seconds_2d[dt_ix, thermal_slot] =
				Dates.value(Millisecond(ts.time - ref_time)) / 1000.0
			for name in extra_names
				extras_2d[name][dt_ix, thermal_slot] =
					extras[name][thermal_slot][ts_ix]
			end
		end

		lock(pbar_lock) do
			next!(pbar)
		end
	end

	println("Writing output to $output_path")
	_write_normalize_output(
		output_path,
		output,
		grid,
		all_dt_seconds,
		tracks,
		time_seconds_2d,
		spec_names,
		extras_2d,
		extra_names,
		extra_attrs;
		azimuthal_average=azimuthal_average,
		output_std=output_std,
	)
	println("Done.")
	return nothing
end
