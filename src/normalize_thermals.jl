using Interpolations: interpolate, Gridded, Linear, extrapolate
using ProgressMeter: Progress

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
# Direction-dependent boundary radii
# ──────────────────────────────────────────────────────────────────

"""
	read_boundary_radii(path, grid)

Read a file of direction-dependent thermal boundary radii into a lookup from
`(thermal_ix, time)` to an `(n_theta, n_phi)` radius matrix.

The file (written by `thermals.ellipsoid_boundaries` in the thermals repository) is indexed by
sample, not by thermal and time, because only a small share of all (thermal, time) pairs hold a
thermal. Python dimensions: `boundary_radius(sample, theta, phi)` in meters,
`sample_thermal_ix(sample)`, and `sample_time(sample)`, with `theta` and `phi` coordinates.
NCDatasets reverses the dimension order, so Julia reads `boundary_radius` as
`(phi, theta, sample)`.

The file's directions must be the normalization grid's own (same θ and φ values), because each
radius is applied to the grid direction it was measured along. A sample with any NaN radius (no
boundary found) is left out of the lookup, so the normalization skips it.
"""
function read_boundary_radii(path::String, grid)
	NCDataset(path, "r") do ds
		file_theta = Array(ds["theta"])
		file_phi = Array(ds["phi"])
		(
			length(file_theta) == grid.n_theta &&
			length(file_phi) == grid.n_phi &&
			maximum(abs.(file_theta .- grid.theta_vals)) < 1e-6 &&
			maximum(abs.(file_phi .- grid.phi_vals)) < 1e-6
		) || error(
			"The boundary radii in $path are on a different (θ, φ) grid than the " *
			"normalization (n_theta=$(grid.n_theta), n_phi=$(grid.n_phi)).",
		)
		radii = Array(ds["boundary_radius"])
		thermal_ixs = Array(ds["sample_thermal_ix"])
		times = Array(ds["sample_time"])
		lookup = Dict{Tuple{Int, DateTime}, Matrix{Float64}}()
		for sample in eachindex(thermal_ixs)
			# (phi, theta) as stored -> (theta, phi), the grid's order
			radius_matrix = permutedims(Float64.(coalesce.(radii[:, :, sample], NaN)))
			any(isnan, radius_matrix) && continue
			lookup[(Int(thermal_ixs[sample]), times[sample])] = radius_matrix
		end
		return lookup
	end
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

With a `boundary_lookup` (from `read_boundary_radii`), each timestep's direction-dependent radii
come from the lookup, and a timestep that is not in it is skipped, together with its extras.
Without one, every timestep is a sphere of radius `scale_var`.
"""
function read_thermal_tracks(path::String; scale_var::String, boundary_lookup=nothing)
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
		consumed =
			Set(["x", "y", "z", "dt_thermal_seconds", "time", "thermal_ix"])
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
					ismissing(sr) ||
					isnan(Float64(sr)) ||
					ismissing(x_val) ||
					ismissing(y_val) ||
					ismissing(z_val) ||
					ismissing(dt_val)
				any_missing_or_nan && continue
				# Direction-dependent radii, when given; a timestep without them is skipped
				boundary_radius = fill(Float64(sr), 1, 1)
				if boundary_lookup !== nothing
					key = (Int(thermal_ixs[ti]), time_vals[di])
					haskey(boundary_lookup, key) || continue
					boundary_radius = boundary_lookup[key]
				end
				push!(
					timesteps,
					ThermalTimeStep(
						Float64(x_val),
						Float64(y_val),
						Float64(z_val),
						Float64(sr),
						Float64(dt_val),
						time_vals[di],
						boundary_radius,
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

"""
	_timestep_subset(ts, rams_grid, r_norm_max, grid)

Native-grid index box around ONE thermal timestep: on each axis, from the lowest to the highest
offset of the outermost normalized shell (`r_norm_max` times the boundary radius in each grid
direction), plus the one-cell interpolation margin that `coord_range` adds.

This replaces the older per-TRACK box (the bounding box of the whole lifetime path, padded by
the lifetime-maximum radius), which every timestep of the track then re-read. The
per-timestep box is typically several times smaller, yet still holds every
(r_norm ≤ r_norm_max) query point. `Gridded(Linear())` uses only the 8 native points around
each query point, so a smaller box that still brackets every query point gives the same
values. Query points beyond the domain edge extrapolate to NaN with either box.

Each axis's index range is the union of the ranges that bracket the sphere on the thermo
(scalar) coordinates AND on the momentum (staggered) coordinates. A staggered variable such
as WC is sampled on the momentum levels, which sit half a cell away from the thermo levels.
A range built from the thermo levels alone can therefore stop just short of the sphere's
outermost points on the momentum levels, turning them into NaN. (The older per-track box hid
this everywhere except at each track's extreme timesteps, where it produced such NaNs at the
bottom of the r_norm ≈ 2 shell.)
"""
function _timestep_subset(ts::ThermalTimeStep, rams_grid, r_norm_max::Float64, grid::SphericalGrid)
	# Reach of the outermost normalized shell along each axis, on each side of the center (m).
	# The shell's point in direction (θ, φ) lies r_norm_max R(θ, φ) away along that direction,
	# so its offset along x is r_norm_max R(θ, φ) sin θ cos φ, and likewise for y and z. The box
	# spans the smallest to the largest of those offsets. For a sphere (one radius in every
	# direction) that is ±r_norm_max R on every axis, because the grid holds the directions along
	# ±x, ±y, and ±z. For a direction-dependent boundary the box follows the boundary's shape:
	# a long ray downward extends only the bottom of the box.
	shell_radius = r_norm_max .* reshape(ts.boundary_radius, 1, size(ts.boundary_radius)...)
	offsets_x = shell_radius .* grid.sin_theta_cos_phi
	offsets_y = shell_radius .* grid.sin_theta_sin_phi
	offsets_z = shell_radius .* grid.cos_theta

	# Index range bracketing [center + lowest offset, center + highest offset] on both the
	# thermo and the momentum coordinates of one axis
	function bracketing_range(thermo_coords, momentum_coords, center, offsets)
		low, high = center + minimum(offsets), center + maximum(offsets)
		thermo_ixs = coord_range(thermo_coords, low, high)
		momentum_ixs = coord_range(momentum_coords, low, high)
		return min(first(thermo_ixs), first(momentum_ixs)):max(
			last(thermo_ixs),
			last(momentum_ixs),
		)
	end

	x_ixs = bracketing_range(rams_grid.x, rams_grid.xm, ts.center_x, offsets_x)
	y_ixs = bracketing_range(rams_grid.y, rams_grid.ym, ts.center_y, offsets_y)
	z_ixs = bracketing_range(rams_grid.z, rams_grid.zm, ts.center_z, offsets_z)

	# Same layout as compute_spatial_subset: index ranges plus the coordinates they select
	return (
		x_ixs=x_ixs,
		y_ixs=y_ixs,
		z_ixs=z_ixs,
		x_sub=rams_grid.x[x_ixs],
		y_sub=rams_grid.y[y_ixs],
		z_sub=rams_grid.z[z_ixs],
		xm_sub=rams_grid.xm[x_ixs],
		ym_sub=rams_grid.ym[y_ixs],
		zm_sub=rams_grid.zm[z_ixs],
	)
end

# Full-domain field buffers, one per input POSITION within a spec, kept alive across files
# in this process. Each is ~300 MB, and allocating a fresh set for every file made the GC do
# a large share of the work. Every process (the main one, or each Distributed worker) has
# its own copy of this module global, and each process runs one file at a time, so no
# locking is needed. `normalize_thermals` empties it when it finishes.
const _FIELD_BUFFERS = Dict{Int, Any}()

"""
	_normalize_one_file(rams_path, timesteps, context)

Composite every thermal timestep in `timesteps`, all of which share the RAMS file
`rams_path`.

Mechanism: open the file once. For each composited quantity, decompress the union of chunks
that any listed thermal's per-timestep box touches (see `_chunk_aligned_blocks`). Then, for
each thermal, cut its box out of that in-memory field and run the unchanged per-thermal
steps, in the same order as before:

1. convert the box to Float64;
2. clip the inputs listed in `context.nonnegative_vars` at zero;
3. combine the inputs (a pass-through for a plain variable);
4. interpolate linearly onto the thermal's (r, θ, φ) grid (NaN outside the domain);
5. reduce over φ to a mean and a spread, when `context.azimuthal_average` is set.

Only the source of the native values changes relative to the older thermal-major loop, so
every output value is identical.

This function is the unit of parallel work. Files are independent, so `normalize_thermals`
hands one file at a time to each worker process.

Returns `(mean, std)`. Each is a Dict from composite name to an array whose LAST axis
indexes `timesteps`. `std` is `nothing` unless φ-averaging with the spread was requested.
"""
function _normalize_one_file(
	rams_path::String,
	timesteps::Vector{ThermalTimeStep},
	context,
)
	grid = context.grid
	n_timesteps = length(timesteps)

	# Native-grid index box of each thermal at this time
	subsets = [
		_timestep_subset(ts, context.rams_grid, context.r_norm_max, grid) for
		ts in timesteps
	]

	# Model-coordinate positions of each thermal's (r, θ, φ) query points, centered on this
	# timestep's smoothed center. Each direction is scaled by the thermal's radius in that
	# direction: a (1, n_theta, n_phi) array for a directional boundary, or a single radius
	# (1, 1, 1) for a sphere, both of which broadcast against the (n_r, 1, 1) radial axis.
	query_points = map(timesteps) do ts
		rho = grid.R .* reshape(ts.boundary_radius, 1, size(ts.boundary_radius)...)
		(
			x=rho .* grid.sin_theta_cos_phi .+ ts.center_x,
			y=rho .* grid.sin_theta_sin_phi .+ ts.center_y,
			z=rho .* grid.cos_theta .+ ts.center_z,
		)
	end

	# Stored per-thermal shape: (r, θ) after the φ reduction, or the full (r, θ, φ)
	stored_shape =
		context.azimuthal_average ? (grid.n_r, grid.n_theta) :
		(grid.n_r, grid.n_theta, grid.n_phi)
	mean_by_name = Dict(
		spec.name => Array{Float64}(undef, stored_shape..., n_timesteps) for
		spec in context.specs
	)
	std_by_name =
		context.azimuthal_average && context.include_azimuthal_std ?
		Dict(
			spec.name => Array{Float64}(undef, stored_shape..., n_timesteps) for
			spec in context.specs
		) : nothing

	# Scratch array for one thermal's full (r, θ, φ) interpolation
	interpolated = Array{Float64}(undef, grid.n_r, grid.n_theta, grid.n_phi)

	# One reusable full-domain buffer per input POSITION within a spec. A derived spec needs
	# all of its inputs in memory at once, so each position gets its own buffer.
	field_buffers = _FIELD_BUFFERS

	NCDataset(rams_path, "r") do ds
		for spec in context.specs
			# A horizontal-wind projection interpolates u and v separately, each on its own
			# staggered axis, and projects them at the query points (see
			# `horizontal_wind_component`)
			is_projection = spec.projection !== :none
			# Staggered inputs are sampled on their momentum axis, scalars on the thermo
			# grid; a projection's inputs keep their own axes, so it has no common one
			staggered_axis = is_projection ? :none : _spec_staggered_axis(spec)

			# Decompress, once, every chunk that some thermal at this time needs
			input_fields = Dict{String, Any}()
			for (input_position, input_name) in enumerate(spec.inputs)
				variable = ds[input_name]
				nx, ny, _ = size(variable)
				# Horizontal chunk tile size; a contiguous (unchunked) variable is one tile
				storage, chunk_sizes = chunking(variable.var)
				chunk_size_x, chunk_size_y =
					storage === :chunked ? (chunk_sizes[1], chunk_sizes[2]) :
					(nx, ny)
				blocks = _chunk_aligned_blocks(
					subsets,
					chunk_size_x,
					chunk_size_y,
					nx,
					ny,
				)
				field_buffers[input_position] = _load_blocks(
					variable,
					blocks,
					get(field_buffers, input_position, nothing),
				)
				input_fields[input_name] = field_buffers[input_position]
			end

			for (timestep_index, (subset, query)) in
				enumerate(zip(subsets, query_points))
				if is_projection
					# Project the separately interpolated u and v onto the thermal's
					# horizontal axes at every (r, θ, φ) query point
					_project_horizontal_wind!(
						interpolated,
						spec,
						input_fields,
						subset,
						query,
						grid,
					)
				else
					# Native coordinates of this box, on the staggered axis where applicable
					x_coords = staggered_axis === :x ? subset.xm_sub : subset.x_sub
					y_coords = staggered_axis === :y ? subset.ym_sub : subset.y_sub
					z_coords = staggered_axis === :z ? subset.zm_sub : subset.z_sub

					# Cut this thermal's box out of the in-memory fields
					native = Dict{String, Array{Float64, 3}}(
						input_name => Float64.(
							input_fields[input_name][
								subset.x_ixs,
								subset.y_ixs,
								subset.z_ixs,
							],
						) for input_name in spec.inputs
					)

					# Clip non-negative inputs at zero BEFORE combining, so a derived total is
					# summed from clipped species. Lossy compression turns true zeros into small
					# values of either sign.
					for input_name in spec.inputs
						if input_name in context.nonnegative_vars
							native[input_name] = max.(native[input_name], 0.0)
						end
					end

					# Combine inputs into the field to interpolate (pass-through for plain vars)
					data_sub = spec.combine(native)

					# Linear interpolation onto the (r, θ, φ) query points; NaN outside the box
					itp = interpolate(
						(x_coords, y_coords, z_coords),
						data_sub,
						Gridded(Linear()),
					)
					eitp = extrapolate(itp, NaN)
					interpolated .= eitp.(query.x, query.y, query.z)
				end

				if context.azimuthal_average
					# Collapse φ to its mean (and spread) so only the (r, θ) profile is kept.
					# With a directional boundary, each φ is weighted by its volume, R(θ, φ)³.
					ts = timesteps[timestep_index]
					phi_mean, phi_std =
						is_directional(ts) ?
						_azimuthal_mean_std(interpolated, ts.boundary_radius .^ 3) :
						_azimuthal_mean_std(interpolated)
					mean_by_name[spec.name][:, :, timestep_index] .= phi_mean
					if std_by_name !== nothing
						std_by_name[spec.name][:, :, timestep_index] .= phi_std
					end
				else
					mean_by_name[spec.name][:, :, :, timestep_index] .= interpolated
				end
			end
		end
	end
	return (mean_by_name, std_by_name)
end

"""
	_interpolate_input_to_query(field, subset, staggered_axis, query)

One RAMS input, cut to this thermal's native box and linearly interpolated to the thermal's
`(r, θ, φ)` query points (NaN outside the box). `staggered_axis` picks the momentum
coordinates on the axis the input is staggered along, so u and v can each be interpolated on
their own Arakawa-C grid.
"""
function _interpolate_input_to_query(field, subset, staggered_axis::Symbol, query)
	# Native coordinates of this box, on the staggered axis where applicable
	x_coords = staggered_axis === :x ? subset.xm_sub : subset.x_sub
	y_coords = staggered_axis === :y ? subset.ym_sub : subset.y_sub
	z_coords = staggered_axis === :z ? subset.zm_sub : subset.z_sub
	native = Float64.(field[subset.x_ixs, subset.y_ixs, subset.z_ixs])
	itp = interpolate((x_coords, y_coords, z_coords), native, Gridded(Linear()))
	return extrapolate(itp, NaN).(query.x, query.y, query.z)
end

"""
	_project_horizontal_wind!(projected, spec, input_fields, subset, query, grid)

Fill `projected` (shape `(n_r, n_θ, n_φ)`) with the horizontal wind of a projection spec,
projected onto the thermal's radial or azimuthal direction at every query point.

The query point at azimuth φ lies in the horizontal direction (cos φ, sin φ) from the
thermal's axis (see the query-point construction in `_normalize_one_file`). The radial
component is therefore `u cos φ + v sin φ`, and the azimuthal component
`-u sin φ + v cos φ`. On the axis itself (r = 0, or θ = 0 or π) the direction is undefined;
there the ring's points coincide, so the azimuthal mean of either component is zero.
"""
function _project_horizontal_wind!(projected, spec, input_fields, subset, query, grid)
	u_name, v_name = spec.inputs
	# Each component interpolated on its own staggered axis
	u_at_query = _interpolate_input_to_query(
		input_fields[u_name],
		subset,
		_staggered_axis(u_name),
		query,
	)
	v_at_query = _interpolate_input_to_query(
		input_fields[v_name],
		subset,
		_staggered_axis(v_name),
		query,
	)
	# Horizontal direction of each query point from the axis, broadcast over (r, θ)
	cos_phi = reshape(cos.(grid.phi_vals), 1, 1, :)
	sin_phi = reshape(sin.(grid.phi_vals), 1, 1, :)
	if spec.projection === :radial
		projected .= u_at_query .* cos_phi .+ v_at_query .* sin_phi
	elseif spec.projection === :azimuthal
		projected .= .-u_at_query .* sin_phi .+ v_at_query .* cos_phi
	else
		error("unknown projection :$(spec.projection) for composite '$(spec.name)'")
	end
	return projected
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

"""
	_azimuthal_mean_std(field, volume_weights)

Volume-weighted azimuthal mean and standard deviation of an `(r, θ, φ)` field, for a thermal
whose radius depends on direction.

At a fixed normalized radius, a grid point in direction (θ, φ) stands for a physical volume
proportional to R(θ, φ)³ r_norm² sin θ dr dθ dφ. Along one ring of fixed (r, θ), the factor that
varies with φ is R(θ, φ)³, given here as `volume_weights` with shape `(n_theta, n_phi)`. The
weighted mean is therefore the average over the ring's physical volume, which is what a volume
integral of the azimuthally averaged field needs. With equal weights it equals the plain mean.
NaN samples are ignored, as in the unweighted version.
"""
function _azimuthal_mean_std(field::Array{Float64, 3}, volume_weights::AbstractMatrix{Float64})
	n_r, n_theta, n_phi = size(field)
	mean_rt = Array{Float64}(undef, n_r, n_theta)
	std_rt = Array{Float64}(undef, n_r, n_theta)
	@inbounds for j in 1:n_theta, i in 1:n_r
		# Pass 1: weighted mean over the valid (non-NaN) φ samples
		sum_weighted = 0.0
		sum_weights = 0.0
		for k in 1:n_phi
			v = field[i, j, k]
			isnan(v) && continue
			sum_weighted += volume_weights[j, k] * v
			sum_weights += volume_weights[j, k]
		end
		if sum_weights == 0.0
			mean_rt[i, j] = NaN
			std_rt[i, j] = NaN
			continue
		end
		m = sum_weighted / sum_weights
		mean_rt[i, j] = m
		# Pass 2: weighted population variance about the mean
		sum_sq_dev = 0.0
		for k in 1:n_phi
			v = field[i, j, k]
			isnan(v) && continue
			d = v - m
			sum_sq_dev += volume_weights[j, k] * d * d
		end
		std_rt[i, j] = sqrt(sum_sq_dev / sum_weights)
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

`output` and `output_std` are FUNCTIONS from a composite name to its dense
`(dt, thermal, r, θ[, φ])` array, not stored arrays. The caller builds each dense array on
demand from compact per-file results, so only one variable's dense array exists at a time.
Holding every variable dense at once took ~57 GB for 45 variables, because most
(dt, thermal) slots are empty: thermals have different lifetimes.

`compression_level` (0–9) sets LOSSLESS NetCDF deflate (zlib) compression with the byte
shuffle filter; 0 writes uncompressed. Every value, NaN included, reads back bit-identical.
The payoff is large because ~90% of the dense (dt, thermal) layout is NaN fill, and level 1
already captures most of it at little extra write time. Composited fields are chunked one
thermal at a time — all dt for one thermal, (r, θ) whole (plus one dt per chunk when φ is
kept) — so a chunk holds one thermal's history, and its empty dt slots compress to almost
nothing.
"""
function _write_normalize_output(
	path::String,
	output::Function,
	grid::SphericalGrid,
	all_dt_seconds::Vector{Float64},
	tracks::Vector{ThermalTrack},
	time_seconds_2d::Matrix{Float64},
	composited_var_names::Vector{String},
	extras_2d::Dict{String, Matrix{Float64}},
	extra_names::Vector{String},
	extra_attrs::Dict{String, Dict{String, Any}};
	azimuthal_average::Bool,
	output_std::Union{Nothing, Function}=nothing,
	compression_level::Int=1,
	angular_variables::Dict{String, Any}=Dict{String, Any}(),
	global_attributes::Dict{String, Any}=Dict{String, Any}(),
)
	thermal_ixs = [t.thermal_ix for t in tracks]

	r2d = grid.r_vals * ones(length(grid.theta_vals))'
	t2d = ones(length(grid.r_vals)) * grid.theta_vals'
	x_norm = r2d .* sin.(t2d)
	z_norm = r2d .* cos.(t2d)

	# Lossless compression keywords for defVar; empty when compression is off
	compression_kwargs =
		compression_level > 0 ?
		(shuffle=true, deflatelevel=compression_level) : NamedTuple()

	NCDataset(path, "c"; attrib=global_attributes) do ds
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
			compression_kwargs...,
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
				compression_kwargs...,
				attrib=extra_attrs[name],
			)
			v[:, :] = extras_2d[name]
		end

		# Per-sample quantities that depend on direction (θ, or θ and φ) but not on r_norm,
		# stored (dt, thermal, θ[, φ]) like the fields; each entry is (array, attributes)
		for (name, (values, attributes)) in angular_variables
			angular_dims =
				ndims(values) == 3 ? ("theta", "thermal_ix", "dt_thermal_seconds") :
				("phi", "theta", "thermal_ix", "dt_thermal_seconds")
			v = defVar(
				ds,
				name,
				Float64,
				angular_dims;
				fillvalue=NaN,
				compression_kwargs...,
				attrib=attributes,
			)
			v[ntuple(_ -> Colon(), ndims(values))...] =
				permutedims(values, reverse(ntuple(identity, ndims(values))))
		end

		defVar(ds, "x_norm", permutedims(x_norm), ("theta", "r_norm"))
		defVar(ds, "z_norm", permutedims(z_norm), ("theta", "r_norm"))

		# Composited-field dimensions in defVar (reversed) order, and the matching permutedims
		# order that maps our (dt, thermal, r, θ[, φ]) storage into it. Averaged fields drop φ.
		# Chunk sizes follow the same defVar order: one thermal per chunk (see the docstring).
		n_dt = length(all_dt_seconds)
		if azimuthal_average
			field_dims = ("theta", "r_norm", "thermal_ix", "dt_thermal_seconds")
			storage_to_defvar_perm = (4, 3, 2, 1)
			field_chunk_sizes = (grid.n_theta, grid.n_r, 1, n_dt)
		else
			field_dims =
				("phi", "theta", "r_norm", "thermal_ix", "dt_thermal_seconds")
			storage_to_defvar_perm = (5, 4, 3, 2, 1)
			# Keeping φ makes one thermal's whole history ~n_phi× larger, so one dt per chunk
			field_chunk_sizes = (grid.n_phi, grid.n_theta, grid.n_r, 1, 1)
		end
		field_compression_kwargs =
			compression_level > 0 ?
			(compression_kwargs..., chunksizes=field_chunk_sizes) : NamedTuple()
		all_colons = ntuple(_ -> Colon(), length(field_dims))

		for var in composited_var_names
			println("  Writing $var...")
			v = defVar(
				ds,
				var,
				Float64,
				field_dims;
				fillvalue=NaN,
				field_compression_kwargs...,
			)
			v[all_colons...] = permutedims(output(var), storage_to_defvar_perm)

			# Azimuthal-asymmetry companion field (φ-std), same shape as the averaged field.
			if azimuthal_average && output_std !== nothing
				vstd = defVar(
					ds,
					"$(var)_phi_std",
					Float64,
					field_dims;
					fillvalue=NaN,
					field_compression_kwargs...,
					attrib=Dict(
						"long_name" => "azimuthal (phi) standard deviation of $var",
					),
				)
				vstd[all_colons...] =
					permutedims(output_std(var), storage_to_defvar_perm)
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
						 lite=true, azimuthal_average=true, include_azimuthal_std=true,
						 nonnegative_vars=String[], n_workers=1, compression_level=1)

Build a composite dataset in thermal-normalized spherical coordinates. For each
thermal at each timestep, linearly interpolates RAMS data from the Cartesian
grid onto an `(r_norm, theta, phi)` grid centered on the thermal where
`r_norm = 1` is the thermal edge. `r_norm_max` sets how far out (in thermal radii)
the grid extends; the Cartesian subset is padded to match, so the whole grid is
sampled (see the `compute_spatial_subset` call).

`composited_vars` is a vector whose entries are each either a RAMS variable name (a
`String`, interpolated as-is) or a `CompositeSpec` built with `derived`/`derived_sum`
(a quantity computed on the native RAMS grid before interpolation — see `CompositeSpec`).

An entry built with `horizontal_wind_component` composites the horizontal wind projected onto
the thermal's radial or azimuthal direction. Its u and v inputs are interpolated separately and
projected at the query points before any azimuthal average (see `_project_horizontal_wind!`).

`lite` selects which RAMS files to composite: lite (`a-L-`) or full-analysis (`a-A-`).
Only thermal_ds times with a matching RAMS file are composited (see the availability check).

`azimuthal_average` (default `true`) collapses the φ axis to its mean during accumulation,
so the composited fields are `(dt_thermal_seconds, thermal_ix, r_norm, theta)` — the
per-thermal azimuthal profiles, at 1/n_phi the memory of the full `(…, phi)` field. Set it
`false` to keep the full φ axis (e.g. to inspect the azimuthal structure itself), at the
cost of an n_phi× larger array. When averaging, `include_azimuthal_std` (default `true`)
additionally writes a `<var>_phi_std` field holding the azimuthal standard deviation, a
per-`(r, θ)` measure of how much azimuthal variation the mean is hiding.

`nonnegative_vars` lists RAMS variables to clip at zero on the native grid before a spec
combines its inputs (and so before interpolation). Lossy (ZFP) compression turns true zeros
into small values of either sign; clipping restores them. A derived total is summed from
clipped species.

`compression_level` (0–9, default 1) sets lossless deflate compression of the output file;
0 writes it uncompressed. Values are bit-identical either way (see `_write_normalize_output`).

# Direction-dependent boundaries

By default every thermal is a sphere of radius `scale_var`, so `r_norm = 1` is that sphere.
With `boundary_path`, the thermal's radius instead depends on direction: the file gives, for each
(thermal, time), the distance R(θ, φ) from the center to the thermal's edge along every grid
direction (see `read_boundary_radii`). The query point at normalized radius r in direction (θ, φ)
then lies r R(θ, φ) from the center, so `r_norm = 1` is the boundary itself in every direction.
Three things follow:

- A (thermal, time) without radii in the file is not composited.
- With `azimuthal_average`, each φ is weighted by its volume, R(θ, φ)³, so the φ-mean is the mean
  over the ring's physical volume (`_azimuthal_mean_std` with weights).
- The output gains `boundary_radius_cubed_phi_mean(dt, thermal, θ)`, the φ-mean of R³. A volume
  integral of a φ-averaged field f is 2π Σ f ⟨R³⟩ r_norm² sin θ Δr Δθ. Without φ averaging, the
  full `boundary_radius(dt, thermal, θ, φ)` is written instead.

`scale_var` is still read with a boundary file: a timestep whose `scale_var` is NaN is skipped.

# How the work is organized (and why it is fast)

The loop runs over RAMS FILES, not over thermals. Each file is opened once. For each
composited variable, the union of HDF5 chunks that any thermal alive at that time needs is
decompressed once, and each thermal's box is then cut from memory. The lite files store
full-column chunks that HDF5 must decompress whole, so the older thermal-major loop spent
most of its time decompressing the same chunks again for every thermal, timestep, and
variable. Each thermal's box also now spans only that timestep's sphere, not the thermal's
whole lifetime path. Neither change alters any output value (see `_normalize_one_file`).

`n_workers > 1` processes files in parallel on that many worker processes (Distributed),
which are started here and removed at the end. Threads would not help: NCDatasets serializes
every NetCDF call behind one global lock. On a worker, each spec's `combine` function must
be callable. The package's `composite`/`derived_sum` specs and anonymous functions defined in
`Main` are fine, because Distributed serializes them by value. A combine function that calls
another user-defined NAMED function or reads a global of the calling session will fail on
the workers; use `n_workers=1` for those, or define them on the workers first.
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
	scale_var::AbstractString="sphere_radius_azimuthal_smoothed",
	nonnegative_vars=String[],
	n_workers::Int=1,
	compression_level::Int=1,
	boundary_path::Union{Nothing, AbstractString}=nothing,
)
	# Normalize each requested variable into a CompositeSpec: bare strings become pass-through
	# specs; CompositeSpec entries (from derived/derived_sum) are kept as-is.
	specs = CompositeSpec[
		el isa CompositeSpec ? el : composite(String(el)) for
		el in composited_vars
	]
	thermal_ds_path = String(thermal_ds_path)
	rams_output_dir = String(rams_output_dir)
	rams_grid_path = String(rams_grid_path)
	output_path = String(output_path)

	grid = build_spherical_grid(n_r, n_theta, n_phi, Float64(r_norm_max))

	# Direction-dependent boundary radii, when given
	directional = boundary_path !== nothing
	boundary_lookup = nothing
	if directional
		println("Reading direction-dependent boundary radii from $boundary_path")
		boundary_lookup = read_boundary_radii(String(boundary_path), grid)
		println("  $(length(boundary_lookup)) (thermal, time) samples with a boundary")
	end

	println("Reading thermal tracks from $thermal_ds_path")
	tracks, extras, extra_names, extra_attrs = read_thermal_tracks(
		thermal_ds_path;
		scale_var=String(scale_var),
		boundary_lookup=boundary_lookup,
	)
	println("  Found $(length(tracks)) thermals")
	if !isempty(extra_names)
		println(
			"  Propagating $(length(extra_names)) extra variable(s): " *
			join(extra_names, ", "),
		)
	end

	println("Reading RAMS grid from $rams_grid_path")
	gc = read_rams_grid(rams_grid_path)

	println(
		"  Spherical grid: r=$(grid.n_r) × θ=$(grid.n_theta) × φ=$(grid.n_phi)",
	)

	# Determine which thermal-track timesteps have a corresponding RAMS output file.
	# Tracking is typically run on high-frequency lite output, while normalization may be
	# run against lower-frequency full-analysis output (lite=false). In that case most
	# thermal_ds times have no matching RAMS file, so we composite only the times where
	# both the thermal tracks and the RAMS output are present. The existence check is done
	# once here (and cached) so the hot loop below neither re-stats files nor warns per-file.
	unique_track_times = sort(
		collect(Set(ts.time for track in tracks for ts in track.timesteps)),
	)
	rams_file_available = Dict{DateTime, Bool}()
	for track_time in unique_track_times
		rams_file_available[track_time] = isfile(
			datetime_to_rams_path(rams_output_dir, track_time, lite=lite),
		)
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
	first_available_time =
		first(t for t in unique_track_times if rams_file_available[t])
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
				"$first_path: " *
				join(missing_inputs, ", "),
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

	# Every (thermal, timestep) with a RAMS file is one work item. Items are grouped by the
	# RAMS file they read, so each file is opened and its chunks decompressed only once.
	# `item_slot` and `item_dt_ix` record where each item lands in the (dt, thermal) layout.
	item_timesteps = ThermalTimeStep[]
	item_slot = Int[]
	item_dt_ix = Int[]
	item_ts_ix = Int[]
	item_indices_by_time = Dict{DateTime, Vector{Int}}()
	for (thermal_slot, track) in enumerate(tracks),
		(ts_ix, ts) in enumerate(track.timesteps)
		# Skip timesteps with no matching RAMS file (already reported in the warning above)
		rams_file_available[ts.time] || continue
		push!(item_timesteps, ts)
		push!(item_slot, thermal_slot)
		push!(item_dt_ix, dt_to_ix[ts.dt_thermal_seconds])
		push!(item_ts_ix, ts_ix)
		push!(
			get!(item_indices_by_time, ts.time, Int[]),
			length(item_timesteps),
		)
	end

	# Per-(dt, thermal) metadata needs no RAMS data, so it is filled here directly
	extras_2d = Dict{String, Matrix{Float64}}(
		name => fill(NaN, n_dt, n_thermals) for name in extra_names
	)
	ref_time = DateTime(2000, 1, 1)
	time_seconds_2d = fill(NaN, n_dt, n_thermals)
	for item_index in eachindex(item_timesteps)
		dt_ix = item_dt_ix[item_index]
		thermal_slot = item_slot[item_index]
		time_seconds_2d[dt_ix, thermal_slot] =
			Dates.value(
				Millisecond(item_timesteps[item_index].time - ref_time),
			) / 1000.0
		for name in extra_names
			extras_2d[name][dt_ix, thermal_slot] =
				extras[name][thermal_slot][item_ts_ix[item_index]]
		end
	end

	# Direction-dependent radii of every composited sample: the φ-mean of R³ per θ (for volume
	# integrals of φ-averaged fields), or the full R(θ, φ) when φ is kept
	angular_variables = Dict{String, Any}()
	if directional
		if azimuthal_average
			cubed_phi_mean = fill(NaN, n_dt, n_thermals, grid.n_theta)
			for item_index in eachindex(item_timesteps)
				radius = item_timesteps[item_index].boundary_radius
				cubed_phi_mean[item_dt_ix[item_index], item_slot[item_index], :] .=
					vec(sum(radius .^ 3; dims=2)) ./ size(radius, 2)
			end
			angular_variables["boundary_radius_cubed_phi_mean"] = (
				cubed_phi_mean,
				Dict{String, Any}(
					"units" => "m3",
					"long_name" => "azimuthal (phi) mean of the cubed boundary radius R(theta, phi)",
				),
			)
		else
			full_radius = fill(NaN, n_dt, n_thermals, grid.n_theta, grid.n_phi)
			for item_index in eachindex(item_timesteps)
				full_radius[item_dt_ix[item_index], item_slot[item_index], :, :] .=
					item_timesteps[item_index].boundary_radius
			end
			angular_variables["boundary_radius"] = (
				full_radius,
				Dict{String, Any}(
					"units" => "m",
					"long_name" => "distance from the thermal center to its boundary along (theta, phi)",
				),
			)
		end
	end
	global_attributes = Dict{String, Any}(
		"normalization" =>
			directional ? "direction-dependent boundary radius from $(boundary_path)" :
			"sphere of radius $(scale_var)",
		"azimuthal_mean_weighting" =>
			directional && azimuthal_average ? "volume (boundary_radius cubed)" : "none",
	)

	# One job per RAMS file, in time order: (file path, the thermal timesteps it serves)
	job_times = sort(collect(keys(item_indices_by_time)))
	job_item_indices = [item_indices_by_time[t] for t in job_times]
	jobs = [
		(
			datetime_to_rams_path(rams_output_dir, t, lite=lite),
			item_timesteps[item_indices_by_time[t]],
		) for t in job_times
	]

	# Everything a worker needs besides its job. Composite specs travel inside it; a derived
	# spec's combine function must therefore be callable on a worker (see the docstring).
	context = (
		specs=specs,
		grid=grid,
		rams_grid=gc,
		r_norm_max=Float64(r_norm_max),
		nonnegative_vars=Set{String}(String.(collect(nonnegative_vars))),
		azimuthal_average=azimuthal_average,
		include_azimuthal_std=include_azimuthal_std,
	)
	process_job = job -> _normalize_one_file(job[1], job[2], context)
	progress = Progress(length(jobs); output=stdout)

	println(
		"Processing $(length(item_timesteps)) thermal timesteps in " *
		"$(length(jobs)) RAMS files" *
		(n_workers <= 1 ? ", in this process..." : ", on $n_workers worker processes..."),
	)
	results = _map_over_workers(
		process_job,
		jobs;
		n_workers=n_workers,
		progress=progress,
	)
	# Release this process's field buffers (only the in-process path fills them here)
	empty!(_FIELD_BUFFERS)

	# Stored per-thermal shape: (r, θ) after the φ reduction, or the full (r, θ, φ)
	stored_shape =
		azimuthal_average ? (grid.n_r, grid.n_theta) :
		(grid.n_r, grid.n_theta, grid.n_phi)
	stored_colons = ntuple(_ -> Colon(), length(stored_shape))

	# Scatter one composite's per-file results into a dense NaN-filled (dt, thermal, …) array.
	# `part` is 1 for the mean (or the full φ field) and 2 for the φ spread. Slots that no
	# item fills (a thermal not alive at that dt) stay NaN, as in the original accumulator.
	function assemble_dense(name::String, part::Int)
		dense = fill(NaN, n_dt, n_thermals, stored_shape...)
		for (item_indices, file_result) in zip(job_item_indices, results)
			file_field = file_result[part][name]
			for (local_index, item_index) in enumerate(item_indices)
				dense[item_dt_ix[item_index], item_slot[item_index], stored_colons...] .=
					selectdim(file_field, ndims(file_field), local_index)
			end
		end
		return dense
	end

	println("Writing output to $output_path")
	_write_normalize_output(
		output_path,
		name -> assemble_dense(name, 1),
		grid,
		all_dt_seconds,
		tracks,
		time_seconds_2d,
		spec_names,
		extras_2d,
		extra_names,
		extra_attrs;
		azimuthal_average=azimuthal_average,
		output_std=azimuthal_average && include_azimuthal_std ?
				   (name -> assemble_dense(name, 2)) : nothing,
		compression_level=compression_level,
		angular_variables=angular_variables,
		global_attributes=global_attributes,
	)
	println("Done.")
	return nothing
end
