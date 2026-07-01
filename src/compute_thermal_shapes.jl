using ProgressMeter: Progress, next!

# ──────────────────────────────────────────────────────────────────
# Core radius finder — 2D disk zero-crossing at the w-maximum level
# (port of the Python notebook's `w_tr_integral_vs_radius` +
#  `first_zero_crossing_coord`, replacing the old 3D spherical cumsum)
# ──────────────────────────────────────────────────────────────────

"""
Zero-crossing radius of the `w_tr` cumulative *area* integral on the horizontal
slab at the thermal's w-maximum height.

This ports the radius-finding method from the `ellipsoid_tracking` notebook:
`w_tr.interp(z=z_smoothed)` (a single 2-D x–y slice at the center height) →
`integrate_over_disk` (cumulative ∫∫ w_tr dA over disks of growing radius, with
each grid cell split into `edge_cell_bins_per_dim²` piecewise-constant sub-cells
so boundary cells contribute fractionally) → `first_zero_crossing_coord` (the
radius at the first sign change of the cumulative integral, returned without
sub-step interpolation, matching `target_sign=None`).

Inputs are for the *subset* already cropped to max_radius around the center:
- `data_sub`: 3D `w_tr` array in Julia (x, y, z) order (WC already de-propagated)
- `dx`, `dy`: uniform xy spacing
- `dz_sub`: per-cell z spacing (length nz_sub) — unused now, kept for signature stability
- `x_coords_sub`, `y_coords_sub`: 1-D coordinate vectors
- `zc_sub`: cell-center z coordinates (from Python `0.5·(z[:-1]+z[1:])`)
- `center_*`: thermal center
- `max_radius`: largest disk radius swept (sub-cells beyond this are dropped)
- `edge_cell_bins_per_dim`: sub-cell sampling per axis (4 → 16 sub-cells/cell)

Returns the zero-crossing radius in meters, or `NaN` if no sign change found.
"""
function _find_zero_radius(
	data_sub::AbstractArray{T, 3},
	center_x::Float64,
	center_y::Float64,
	center_z::Float64,
	x_coords_sub::Vector{Float64},
	y_coords_sub::Vector{Float64},
	zc_sub::Vector{Float64},
	dz_sub::Vector{Float64},
	dx::Float64,
	dy::Float64;
	max_radius::Float64=2000.0,
	edge_cell_bins_per_dim::Int=4,
)::Float64 where {T}
	nx, ny, nz = size(data_sub)
	(nx == 0 || ny == 0 || nz == 0) && return NaN

	# ── Linearly interpolate w_tr onto the horizontal slab at the w-max height ──
	# Matches the notebook's `w_tr.interp(z=z_smoothed)`: a single 2-D (x, y) slice
	# at the thermal center height rather than a 3-D spherical integral.
	# Locate the two cell-center z levels that bracket center_z
	k_lower = searchsortedlast(zc_sub, center_z)
	# Clamp so we always have a valid [k_lower, k_lower+1] bracket (center should be
	# inside the cropped box, so this only guards the degenerate edge cases)
	k_lower = clamp(k_lower, 1, nz - 1)
	k_upper = k_lower + 1
	z_lower = zc_sub[k_lower]
	z_upper = zc_sub[k_upper]
	# Linear interpolation weight on the upper level (0 at z_lower, 1 at z_upper)
	z_weight_upper =
		(z_upper == z_lower) ? 0.0 : (center_z - z_lower) / (z_upper - z_lower)

	# Build the 2-D w_tr slab (nx, ny) at the interpolated height
	w_slab = Matrix{Float64}(undef, nx, ny)
	@inbounds for j in 1:ny, i in 1:nx
		w_lower = Float64(data_sub[i, j, k_lower])
		w_upper = Float64(data_sub[i, j, k_upper])
		w_slab[i, j] = w_lower + z_weight_upper * (w_upper - w_lower)
	end

	# ── Cumulative area integral of w_tr over disks of growing radius ──
	# Each grid cell is split into edge_cell_bins_per_dim² sub-cells (piecewise
	# constant) so partially-covered boundary cells contribute fractionally; this
	# mirrors `_cumulative_integral_by_entry_value` in ellipsoid_tracking.py.

	# Sub-cell center offsets within a parent cell, as a fraction of cell width
	# (matches Python `(arange(n)+0.5)/n - 0.5`; e.g. n=4 → [-0.375,-0.125,0.125,0.375])
	edge_cell_bin_fractional_offsets =
		[((b - 0.5) / edge_cell_bins_per_dim) - 0.5 for b in 1:edge_cell_bins_per_dim]

	# Each sub-cell carries area dA / edge_cell_bins_per_dim²
	subsampled_cell_dA = dx * dy / edge_cell_bins_per_dim^2

	# Per-sub-cell radial entry value (2-D distance to center) and replicated w_tr value
	n_subcells = nx * ny * edge_cell_bins_per_dim * edge_cell_bins_per_dim
	subcell_radius = Vector{Float64}(undef, n_subcells)
	subcell_w = Vector{Float64}(undef, n_subcells)
	count_inside = 0
	@inbounds for j in 1:ny
		y_cell_center = y_coords_sub[j]
		for bj in 1:edge_cell_bins_per_dim
			# Sub-cell y center and its offset from the region center
			subsampled_y = y_cell_center + edge_cell_bin_fractional_offsets[bj] * dy
			dy_from_center = subsampled_y - center_y
			for i in 1:nx
				x_cell_center = x_coords_sub[i]
				w_here = w_slab[i, j]
				for bi in 1:edge_cell_bins_per_dim
					# Sub-cell x center and its offset from the region center
					subsampled_x =
						x_cell_center + edge_cell_bin_fractional_offsets[bi] * dx
					dx_from_center = subsampled_x - center_x
					# For a disk, the entry radius is just the 2-D distance to center
					r = sqrt(dx_from_center*dx_from_center + dy_from_center*dy_from_center)
					# Drop sub-cells outside the swept radius range
					r > max_radius && continue
					count_inside += 1
					subcell_radius[count_inside] = r
					subcell_w[count_inside] = w_here
				end
			end
		end
	end
	count_inside == 0 && return NaN

	# Sort sub-cells by radius and accumulate the cumulative area integral
	r_order = sortperm(@view subcell_radius[1:count_inside])
	r_sorted = subcell_radius[1:count_inside][r_order]
	w_sorted = subcell_w[1:count_inside][r_order]
	integrated = cumsum(w_sorted) .* subsampled_cell_dA

	# ── First radius at which the cumulative integral changes sign ──
	# Matches `first_zero_crossing_coord` with target_sign=None: the first index i
	# where sign(integrated[i]) ≠ sign(integrated[i+1]) (np.diff(np.sign(...)) ≠ 0),
	# returning the radius coord *at* that index (the value before the flip), with
	# no sub-step interpolation.
	@inbounds for i in 1:(length(integrated) - 1)
		if sign(integrated[i]) != sign(integrated[i+1])
			return r_sorted[i]
		end
	end
	return NaN
end

# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

"""Rolling mean with window 3, centered, NaN-aware (min_periods=1)."""
function _rolling_mean_3(arr::AbstractVector{Float64})
	n = length(arr)
	out = Vector{Float64}(undef, n)
	@inbounds for i in 1:n
		s = 0.0
		c = 0
		for j in max(1, i-1):min(n, i+1)
			v = arr[j]
			if !isnan(v)
				s += v
				c += 1
			end
		end
		out[i] = c == 0 ? NaN : s / c
	end
	return out
end

# ──────────────────────────────────────────────────────────────────
# Trajectory input (per-thermal view)
# ──────────────────────────────────────────────────────────────────

struct _TrajInput
	thermal_ix::Int
	valid::BitVector                  # length n_time, true where thermal exists
	times::Vector{DateTime}           # n_time, for looking up RAMS files
	x::Vector{Float64}                # n_time, NaN outside valid
	y::Vector{Float64}
	z::Vector{Float64}
	z_prop::Vector{Float64}
end

# ──────────────────────────────────────────────────────────────────
# Reader for thermal_trajectories.nc
# ──────────────────────────────────────────────────────────────────

function _read_trajectories(path::String)
	NCDataset(path, "r") do ds
		thermal_ixs = Array(ds["thermal_ix"])
		times = Array(ds["time"])

		# Float and int variables were written with _FillValue=NaN / _FillValue=-1,
		# so NCDatasets returns Matrix{Union{Missing, T}} here. Collapse the Missing
		# branch with coalesce, which leaves a concrete-eltype matrix with NaN / -1
		# as the sentinel — matching the in-memory NaN-convention used everywhere
		# else in the pipeline.
		function read2df(name)
			a = coalesce.(Array(ds[name]), NaN)
			size(a) == (length(times), length(thermal_ixs)) ||
				error("Unexpected shape for $name: $(size(a))")
			a
		end
		function read2di(name)
			a = coalesce.(Array(ds[name]), -1)
			size(a) == (length(times), length(thermal_ixs)) ||
				error("Unexpected shape for $name: $(size(a))")
			a
		end

		(
			thermal_ixs=thermal_ixs,
			times=times,
			x=read2df("x"),
			y=read2df("y"),
			z=read2df("z"),
			x_ix=read2di("x_ix"),
			y_ix=read2di("y_ix"),
			z_ix=read2di("z_ix"),
			w_max=read2df("w_max"),
			x_smoothed=read2df("x_smoothed"),
			y_smoothed=read2df("y_smoothed"),
			z_smoothed=read2df("z_smoothed"),
			x_prop=read2df("x_propagation_speed"),
			y_prop=read2df("y_propagation_speed"),
			z_prop=read2df("z_propagation_speed"),
			time_ix=read2di("time_ix"),
			t_minutes=read2df("t_minutes"),
			t_seconds=read2df("t_seconds"),
			dt_thermal_seconds=read2df("dt_thermal_seconds"),
		)
	end
end

# ──────────────────────────────────────────────────────────────────
# Per-thermal shape computation
# ──────────────────────────────────────────────────────────────────

# HDF5 / libnetcdf are not guaranteed to be thread-safe — concurrent opens from
# parallel tasks can segfault the kernel. Serialize NetCDF reads through this lock;
# _find_zero_radius (the expensive part) still runs in parallel.
const _NC_IO_LOCK = ReentrantLock()

function _compute_sphere_radius_for_thermal!(
	sphere_radius::AbstractVector{Float64},    # length n_time, prefilled with NaN
	traj::_TrajInput,
	rams_output_dir::String,
	gc,                                        # NamedTuple(x, y, z) — full grid
	zc_full::Vector{Float64},
	dz_full::Vector{Float64},
	dx::Float64,
	dy::Float64,
	max_radius::Float64,
)
	for ti in 1:length(traj.times)
		traj.valid[ti] || continue
		center_x = traj.x[ti]
		center_y = traj.y[ti]
		center_z = traj.z[ti]
		z_prop = traj.z_prop[ti]
		(isnan(center_x) || isnan(center_y) || isnan(center_z) || isnan(z_prop)) &&
			continue

		# Bounding-box subsetting — Chebyshev max_radius around the center
		x_mask = abs.(gc.x .- center_x) .<= max_radius
		y_mask = abs.(gc.y .- center_y) .<= max_radius
		z_mask = abs.(zc_full .- center_z) .<= max_radius
		x_ixs = findall(x_mask)
		y_ixs = findall(y_mask)
		z_ixs = findall(z_mask)
		(isempty(x_ixs) || isempty(y_ixs) || isempty(z_ixs)) && continue

		rams_path = datetime_to_rams_path(rams_output_dir, traj.times[ti])
		isfile(rams_path) || (@warn "RAMS file not found: $rams_path"; continue)

		wc_sub = lock(_NC_IO_LOCK) do
			NCDataset(rams_path, "r") do ds
				Float32.(ds["WC"][x_ixs, y_ixs, z_ixs])
			end
		end

		w_field = wc_sub .- Float32(z_prop)

		sphere_radius[ti] = _find_zero_radius(
			w_field,
			center_x,
			center_y,
			center_z,
			gc.x[x_ixs],
			gc.y[y_ixs],
			zc_full[z_ixs],
			dz_full[z_ixs],
			dx,
			dy;
			max_radius=max_radius,
		)
	end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
	compute_thermal_shapes(; thermal_trajectories_path, rams_output_dir,
							 rams_grid_path, output_path, max_radius=3000.0)

For each thermal/timestep, compute the volume-weighted zero-crossing radius of
`(WC − z_propagation_speed)` around the thermal center. Also produces a rolling-
mean `sphere_radius_smoothed` and an `out_of_domain` flag. Drops any thermal
that is out-of-domain at any valid timestep.
"""
function compute_thermal_shapes(;
	thermal_trajectories_path::AbstractString,
	rams_output_dir::AbstractString,
	rams_grid_path::AbstractString,
	output_path::AbstractString,
	max_radius::Real=3000.0,
)
	println("Reading trajectories from $thermal_trajectories_path")
	tr = _read_trajectories(String(thermal_trajectories_path))
	n_time = length(tr.times)
	n_thermals = length(tr.thermal_ixs)
	println("  $n_thermals thermals × $n_time times")

	println("Reading RAMS grid from $rams_grid_path")
	gc = read_rams_grid(String(rams_grid_path))
	dx = gc.x[2] - gc.x[1]
	dy = gc.y[2] - gc.y[1]
	dz_full = diff(gc.z)
	zc_full = 0.5 .* (gc.z[1:(end-1)] .+ gc.z[2:end])

	# Domain bounds (match Python: use storm_ds[d].min/max, which is RAMS grid extent)
	x_lo, x_hi = gc.x[1], gc.x[end]
	y_lo, y_hi = gc.y[1], gc.y[end]
	z_lo, z_hi = gc.z[1], gc.z[end]

	# Build per-thermal _TrajInput views
	trajs = Vector{_TrajInput}(undef, n_thermals)
	for ti in 1:n_thermals
		x_col = tr.x[:, ti];
		y_col = tr.y[:, ti];
		z_col = tr.z[:, ti];
		zp_col = tr.z_prop[:, ti]
		valid =
			.!isnan.(x_col) .& .!isnan.(y_col) .& .!isnan.(z_col) .& .!isnan.(zp_col)
		trajs[ti] =
			_TrajInput(tr.thermal_ixs[ti], valid, tr.times, x_col, y_col, z_col, zp_col)
	end

	# Allocate sphere_radius matrix (time, thermal_ix)
	sphere_radius = fill(NaN, n_time, n_thermals)

	# Verify RAMS dim order on the first RAMS file we encounter
	# (pick the first valid time of the first thermal)
	any_path = ""
	for t in trajs, (i, v) in enumerate(t.valid)
		v || continue
		p = datetime_to_rams_path(String(rams_output_dir), t.times[i])
		if isfile(p)
			;
			any_path = p;
			break;
		end
	end
	any_path != "" || error("No RAMS files found for any trajectory timestep")
	verify_rams_dims(any_path, gc.x, gc.y, gc.z, "WC")

	# Parallel over thermals
	println("Computing sphere_radius on $(Threads.nthreads()) threads…")
	pbar = Progress(n_thermals; output=stdout)
	pbar_lock = ReentrantLock()

	@tasks for ti in 1:n_thermals
		sr_col = view(sphere_radius, :, ti)
		_compute_sphere_radius_for_thermal!(
			sr_col,
			trajs[ti],
			String(rams_output_dir),
			gc,
			zc_full,
			dz_full,
			dx,
			dy,
			Float64(max_radius),
		)
		lock(pbar_lock) do
			next!(pbar)
		end
	end

	# sphere_radius_smoothed + out_of_domain + drop fully-out-of-domain thermals
	sphere_radius_smoothed = similar(sphere_radius)
	out_of_domain = falses(n_time, n_thermals)
	keep_mask = trues(n_thermals)

	for ti in 1:n_thermals
		sr_col = sphere_radius[:, ti]
		sphere_radius_smoothed[:, ti] = _rolling_mean_3(sr_col)
		for i in 1:n_time
			trajs[ti].valid[i] || continue
			sr = sr_col[i]
			isnan(sr) && continue
			oob =
				(tr.x[i, ti] - sr < x_lo) ||
				(tr.x[i, ti] + sr > x_hi) ||
				(tr.y[i, ti] - sr < y_lo) ||
				(tr.y[i, ti] + sr > y_hi) ||
				(tr.z[i, ti] - sr < z_lo) ||
				(tr.z[i, ti] + sr > z_hi)
			if oob
				out_of_domain[i, ti] = true
				keep_mask[ti] = false
			end
		end
	end
	n_dropped = n_thermals - count(keep_mask)
	println("  Dropping $n_dropped thermals that are out-of-domain at some timestep")

	# Subset to surviving thermals
	keep_ixs = findall(keep_mask)
	println("  $(length(keep_ixs)) thermals remaining")

	_write_thermal_ds(
		String(output_path),
		tr,
		keep_ixs,
		sphere_radius[:, keep_ixs],
		sphere_radius_smoothed[:, keep_ixs],
		out_of_domain[:, keep_ixs],
	)
	println("Wrote $(output_path)")
	return nothing
end

# ──────────────────────────────────────────────────────────────────
# Write thermal_ds.nc (schema: thermal_trajectories.nc + sphere_radius{,_smoothed} + out_of_domain)
# ──────────────────────────────────────────────────────────────────

function _write_thermal_ds(
	path::String,
	tr::NamedTuple,
	keep_ixs::Vector{Int},
	sphere_radius::Matrix{Float64},           # (n_time, n_kept)
	sphere_radius_smoothed::Matrix{Float64},
	out_of_domain::BitMatrix,
)
	n_kept = length(keep_ixs)
	n_time = length(tr.times)

	# Reindex thermal_ix to be contiguous 0..n_kept-1 (matches Python notebook convention)
	new_thermal_ixs = collect(0:(n_kept-1))

	ref_time = DateTime(2000, 1, 1)
	time_seconds =
		Float64[Dates.value(Millisecond(t - ref_time)) / 1000.0 for t in tr.times]

	isfile(path) && rm(path)
	NCDataset(path, "c") do ds
		defDim(ds, "thermal_ix", n_kept)
		defDim(ds, "time", n_time)

		defVar(ds, "thermal_ix", new_thermal_ixs, ("thermal_ix",))
		v = defVar(
			ds,
			"time",
			Float64,
			("time",);
			attrib=Dict(
				"units" => "seconds since 2000-01-01 00:00:00",
				"calendar" => "standard",
			),
		)
		v[:] = time_seconds

		julia_dims = ("time", "thermal_ix")
		subs(a) = a[:, keep_ixs]

		defVar(ds, "x", subs(tr.x), julia_dims; fillvalue=NaN)
		defVar(ds, "y", subs(tr.y), julia_dims; fillvalue=NaN)
		defVar(ds, "z", subs(tr.z), julia_dims; fillvalue=NaN)
		defVar(ds, "x_ix", subs(tr.x_ix), julia_dims; fillvalue=-1)
		defVar(ds, "y_ix", subs(tr.y_ix), julia_dims; fillvalue=-1)
		defVar(ds, "z_ix", subs(tr.z_ix), julia_dims; fillvalue=-1)
		defVar(ds, "w_max", subs(tr.w_max), julia_dims; fillvalue=NaN)
		defVar(ds, "x_smoothed", subs(tr.x_smoothed), julia_dims; fillvalue=NaN)
		defVar(ds, "y_smoothed", subs(tr.y_smoothed), julia_dims; fillvalue=NaN)
		defVar(ds, "z_smoothed", subs(tr.z_smoothed), julia_dims; fillvalue=NaN)
		defVar(ds, "x_propagation_speed", subs(tr.x_prop), julia_dims; fillvalue=NaN)
		defVar(ds, "y_propagation_speed", subs(tr.y_prop), julia_dims; fillvalue=NaN)
		defVar(ds, "z_propagation_speed", subs(tr.z_prop), julia_dims; fillvalue=NaN)
		defVar(ds, "time_ix", subs(tr.time_ix), julia_dims; fillvalue=-1)
		defVar(ds, "t_minutes", subs(tr.t_minutes), julia_dims; fillvalue=NaN)
		defVar(ds, "t_seconds", subs(tr.t_seconds), julia_dims; fillvalue=NaN)
		defVar(
			ds,
			"dt_thermal_seconds",
			subs(tr.dt_thermal_seconds),
			julia_dims;
			fillvalue=NaN,
		)

		defVar(ds, "sphere_radius", sphere_radius, julia_dims; fillvalue=NaN)
		defVar(
			ds,
			"sphere_radius_smoothed",
			sphere_radius_smoothed,
			julia_dims;
			fillvalue=NaN,
		)
		# BitMatrix → Int8 for NetCDF (NetCDF doesn't support boolean natively)
		defVar(ds, "out_of_domain", Int8.(out_of_domain), julia_dims)
	end
end
