using ProgressMeter: Progress, next!
using Peaks: findminima, peakproms

# ══════════════════════════════════════════════════════════════════
# Shape fitting — ports of the `ellipsoid_tracking` notebook methods
# (src/thermals/ellipsoid_tracking.py + notebook cell `61d87d3d`):
#   • disk radius        : `w_tr` disk integral at the w-max level → first sign change
#   • half-ellipse b     : `w_tr` half-ellipse integral (fixed a) → integral crossing
#                          and second-derivative crossing (first prominent min of dI/db)
# The `w_tr` field lives on the RAMS grid levels `gc.z` (WC's native z), and the
# thermal center z also lives on `gc.z` (chain_w_maxima records `z_coords[ix_z]`),
# so all interpolation/integration here uses grid-level z coordinates.
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# Generic sub-cell cumulative area integrator
# (port of `_cumulative_integral_by_entry_value` in ellipsoid_tracking.py)
# ──────────────────────────────────────────────────────────────────

"""
Cumulative area integral of a 2-D piecewise-constant field over a region
parametrized by a single swept scalar (radius for a disk; vertical semi-axis for
an ellipse). Direct port of `_cumulative_integral_by_entry_value`.

`slab` is 2-D in (dim1, dim2) order matching (`coords1`, `coords2`); `d1`, `d2`
are the uniform spacings; `(center1, center2)` is the region center. Each grid
cell is split into `edge_cell_bins_per_dim²` piecewise-constant sub-cells;
`entry_value_fn(Δ1, Δ2)` returns the smallest value of the swept parameter at
which that sub-cell enters the region (may be `Inf` for never-inside cells).
Sub-cells whose entry value exceeds `max_entry_value` are dropped.

Returns `(entry_value_sorted, integrated)`: the cumulative integral as a function
of the ascending entry value.
"""
function _cumulative_area_integral_by_entry_value(
	slab::AbstractMatrix{<:Real},
	coords1::AbstractVector{Float64},
	coords2::AbstractVector{Float64},
	d1::Float64,
	d2::Float64,
	center1::Float64,
	center2::Float64,
	entry_value_fn,
	max_entry_value::Float64;
	edge_cell_bins_per_dim::Int=4,
)
	n1, n2 = size(slab)
	(n1 == 0 || n2 == 0) && return (Float64[], Float64[])

	# Sub-cell center offsets within a parent cell, as a fraction of cell width
	# (matches Python `(arange(n)+0.5)/n - 0.5`; e.g. n=4 → [-0.375,-0.125,0.125,0.375])
	edge_cell_bin_fractional_offsets =
		[((b - 0.5) / edge_cell_bins_per_dim) - 0.5 for b in 1:edge_cell_bins_per_dim]

	# Each sub-cell carries area dA / edge_cell_bins_per_dim²
	subsampled_cell_dA = d1 * d2 / edge_cell_bins_per_dim^2

	# Per-sub-cell entry value + replicated (piecewise-constant) field value; drop
	# sub-cells beyond the swept range (or permanently outside the region, entry = Inf)
	n_subcells = n1 * n2 * edge_cell_bins_per_dim * edge_cell_bins_per_dim
	entry_value = Vector{Float64}(undef, n_subcells)
	subcell_field = Vector{Float64}(undef, n_subcells)
	count_inside = 0
	@inbounds for i2 in 1:n2
		cell_center_2 = coords2[i2]
		for b2 in 1:edge_cell_bins_per_dim
			# Sub-cell dim-2 center and its offset from the region center
			subsampled_2 = cell_center_2 + edge_cell_bin_fractional_offsets[b2] * d2
			delta2 = subsampled_2 - center2
			for i1 in 1:n1
				cell_center_1 = coords1[i1]
				field_here = Float64(slab[i1, i2])
				for b1 in 1:edge_cell_bins_per_dim
					# Sub-cell dim-1 center and its offset from the region center
					subsampled_1 =
						cell_center_1 + edge_cell_bin_fractional_offsets[b1] * d1
					delta1 = subsampled_1 - center1
					entry = entry_value_fn(delta1, delta2)
					# Keep only sub-cells that enter the region within the swept range
					entry <= max_entry_value || continue
					count_inside += 1
					entry_value[count_inside] = entry
					subcell_field[count_inside] = field_here
				end
			end
		end
	end
	count_inside == 0 && return (Float64[], Float64[])

	# Sort sub-cells by entry value and accumulate the cumulative area integral
	entry_order = sortperm(@view entry_value[1:count_inside])
	entry_value_sorted = entry_value[1:count_inside][entry_order]
	integrated =
		cumsum(subcell_field[1:count_inside][entry_order]) .* subsampled_cell_dA
	return (entry_value_sorted, integrated)
end

# ──────────────────────────────────────────────────────────────────
# Zero-crossing / prominent-extremum coordinate finders
# (ports of `first_zero_crossing_coord` and `first_extremum_coord`)
# ──────────────────────────────────────────────────────────────────

"""
Port of `first_zero_crossing_coord`: the `coord` value at the first sign change of
`values`, or `NaN`. `target_sign` selects which crossings count (matching the
Python `np.diff(np.sign(values))` convention):
- `0`  → any sign change (Python `target_sign=None`)
- `>0` → only changes toward a more-positive sign (`sign[i+1]-sign[i] > 0`)
- `<0` → only changes toward a more-negative sign

The returned coord is the value *at* the pre-flip index `i` (no interpolation).
"""
function _first_zero_crossing(
	coord::AbstractVector{Float64},
	values::AbstractVector{Float64};
	target_sign::Int=0,
)::Float64
	n = length(values)
	n < 2 && return NaN
	@inbounds for i in 1:(n - 1)
		sign_change = sign(values[i + 1]) - sign(values[i])
		hit =
			target_sign == 0 ? (sign_change != 0) :
			target_sign > 0 ? (sign_change > 0) :
			(sign_change < 0)
		hit && return coord[i]
	end
	return NaN
end

"""
Port of `first_extremum_coord(..., extremum_type="min")`: the `coord` of the first
(lowest-index) local minimum of `values` whose topographic prominence is at least
`min_prominence_fraction · (maximum(values) − minimum(values))`, or `NaN` if none.

Uses Peaks.jl (`findminima` + `peakproms`); a minimum's valley prominence there
equals scipy `find_peaks(-values, prominence=…)` (verified), so this matches the
notebook's `find_peaks`-based extraction.
"""
function _first_prominent_min_coord(
	coord::AbstractVector{Float64},
	values::AbstractVector{Float64};
	min_prominence_fraction::Float64=0.5,
)::Float64
	# Need at least one interior sample to host a local minimum
	length(values) < 3 && return NaN
	value_range = maximum(values) - minimum(values)
	# A flat curve has no prominent extremum
	value_range == 0 && return NaN
	min_prominence = min_prominence_fraction * value_range

	# Local minima, then keep only those with sufficient prominence (leftmost first)
	minima = peakproms(findminima(values); min=min_prominence)
	isempty(minima.indices) && return NaN
	return coord[minima.indices[1]]
end

# ──────────────────────────────────────────────────────────────────
# Cross-section interpolation onto a plane through the thermal center
# ──────────────────────────────────────────────────────────────────

"""
Linear interpolation of a 3-D field (x, y, z) onto the horizontal (x, y) slab at
height `target_z`, using the z-level coordinates `z_coords` (ascending); returns an
(nx, ny) matrix. Matches the notebook's `w_tr.interp(z=z_smoothed)`. `target_z` is
bracketed and clamped into the coordinate range.
"""
function _interpolate_z_slab(
	data::AbstractArray{T, 3},
	z_coords::AbstractVector{Float64},
	target_z::Float64,
) where {T}
	nx, ny, nz = size(data)
	k_lower = clamp(searchsortedlast(z_coords, target_z), 1, nz - 1)
	k_upper = k_lower + 1
	z_lower = z_coords[k_lower]
	z_upper = z_coords[k_upper]
	weight_upper = (z_upper == z_lower) ? 0.0 : (target_z - z_lower) / (z_upper - z_lower)

	slab = Matrix{Float64}(undef, nx, ny)
	@inbounds for j in 1:ny, i in 1:nx
		lower = Float64(data[i, j, k_lower])
		upper = Float64(data[i, j, k_upper])
		slab[i, j] = lower + weight_upper * (upper - lower)
	end
	return slab
end

"""
Linear interpolation of a 3-D field (x, y, z) onto the vertical (x, z) cross-section
at `target_y`, using the y coordinates `y_coords` (ascending); returns an (nx, nz)
matrix. The xz analog of `_interpolate_z_slab` (the notebook's `w_tr.interp(y=y_smoothed)`).
"""
function _interpolate_y_slab(
	data::AbstractArray{T, 3},
	y_coords::AbstractVector{Float64},
	target_y::Float64,
) where {T}
	nx, ny, nz = size(data)
	j_lower = clamp(searchsortedlast(y_coords, target_y), 1, ny - 1)
	j_upper = j_lower + 1
	y_lower = y_coords[j_lower]
	y_upper = y_coords[j_upper]
	weight_upper = (y_upper == y_lower) ? 0.0 : (target_y - y_lower) / (y_upper - y_lower)

	slab = Matrix{Float64}(undef, nx, nz)
	@inbounds for k in 1:nz, i in 1:nx
		lower = Float64(data[i, j_lower, k])
		upper = Float64(data[i, j_upper, k])
		slab[i, k] = lower + weight_upper * (upper - lower)
	end
	return slab
end

# ──────────────────────────────────────────────────────────────────
# Core radius finder — 2D disk zero-crossing at the w-maximum level
# (port of the notebook's `w_tr_integral_vs_radius` + `first_zero_crossing_coord`)
# ──────────────────────────────────────────────────────────────────

"""
Zero-crossing radius of the `w_tr` cumulative *area* integral on the horizontal slab
at the thermal's w-maximum height: `w_tr.interp(z=z_center)` → cumulative ∫∫ w_tr dA
over disks of growing radius → first sign change of the integral (no interpolation,
matching `first_zero_crossing_coord` with `target_sign=None`).

Inputs are for the *subset* already cropped around the center:
- `data_sub`: 3-D `w_tr` array in Julia (x, y, z) order (WC already de-propagated)
- `x_coords_sub`, `y_coords_sub`: 1-D coordinate vectors
- `z_coords_sub`: WC z-level coordinates (RAMS grid levels, matching `center_z`)
- `dx`, `dy`: uniform xy spacing
- `center_*`: thermal center
- `max_radius`: largest disk radius swept (sub-cells beyond this are dropped)

Returns the zero-crossing radius in meters, or `NaN` if no sign change found.
"""
function _find_zero_radius(
	data_sub::AbstractArray{T, 3},
	center_x::Float64,
	center_y::Float64,
	center_z::Float64,
	x_coords_sub::Vector{Float64},
	y_coords_sub::Vector{Float64},
	z_coords_sub::Vector{Float64},
	dx::Float64,
	dy::Float64;
	max_radius::Float64=2000.0,
	edge_cell_bins_per_dim::Int=4,
)::Float64 where {T}
	nx, ny, nz = size(data_sub)
	(nx == 0 || ny == 0 || nz == 0) && return NaN

	# Horizontal w_tr slab at the w-max height (w_tr.interp(z=z_smoothed))
	w_slab = _interpolate_z_slab(data_sub, z_coords_sub, center_z)

	# Cumulative area integral over disks of growing radius; a disk's entry value is
	# just the 2-D distance from the center
	disk_entry_value_fn(dx_from_center, dy_from_center) =
		sqrt(dx_from_center * dx_from_center + dy_from_center * dy_from_center)
	radius_sorted, integrated = _cumulative_area_integral_by_entry_value(
		w_slab,
		x_coords_sub,
		y_coords_sub,
		dx,
		dy,
		center_x,
		center_y,
		disk_entry_value_fn,
		max_radius;
		edge_cell_bins_per_dim=edge_cell_bins_per_dim,
	)
	isempty(integrated) && return NaN

	# First radius at which the cumulative integral changes sign (any direction)
	return _first_zero_crossing(radius_sorted, integrated; target_sign=0)
end

# ──────────────────────────────────────────────────────────────────
# Half-ellipse vertical semi-axis finder
# (ports of `integrate_over_ellipse_with_fixed_r_axis` + the per-half integral /
#  second-derivative extraction in notebook cell `61d87d3d`)
# ──────────────────────────────────────────────────────────────────

"""`np.gradient` on a uniformly spaced grid (2nd-order central interior, 1st-order edges)."""
function _gradient_uniform(values::AbstractVector{Float64}, h::Float64)
	n = length(values)
	gradient = Vector{Float64}(undef, n)
	if n < 2
		n == 1 && (gradient[1] = 0.0)
		return gradient
	end
	@inbounds begin
		gradient[1] = (values[2] - values[1]) / h
		gradient[n] = (values[n] - values[n - 1]) / h
		for i in 2:(n - 1)
			gradient[i] = (values[i + 1] - values[i - 1]) / (2h)
		end
	end
	return gradient
end

"""
Linear interpolation of `(x_sorted, y)` (x strictly increasing) at the query points
`x_new`; queries outside the range clamp to the endpoints. Mirrors the `np.interp`
linear behavior used for the uniform-grid resample.
"""
function _interp_to_uniform(
	x_sorted::AbstractVector{Float64},
	y::AbstractVector{Float64},
	x_new::AbstractVector{Float64},
)
	n = length(x_sorted)
	out = Vector{Float64}(undef, length(x_new))
	@inbounds for (k, xq) in enumerate(x_new)
		if xq <= x_sorted[1]
			out[k] = y[1]
		elseif xq >= x_sorted[n]
			out[k] = y[n]
		else
			# x_sorted[j] <= xq < x_sorted[j+1]
			j = searchsortedlast(x_sorted, xq)
			t = (xq - x_sorted[j]) / (x_sorted[j + 1] - x_sorted[j])
			out[k] = y[j] + t * (y[j + 1] - y[j])
		end
	end
	return out
end

"""`np.arange(lo, hi, step)`: evenly spaced values from `lo` (inclusive) up to but excluding `hi`."""
function _arange_exclusive(lo::Float64, hi::Float64, step::Float64)
	n = max(0, Int(ceil((hi - lo) / step)))
	return [lo + k * step for k in 0:(n - 1)]
end

"""
Collapse ties in an ascending coordinate, keeping the *last* value in each run of
equal coordinates — matches the notebook's `groupby("z_semi_axis").last()` used to
de-duplicate the cumulative curve before differentiating.
"""
function _dedup_keep_last(x_sorted::AbstractVector{Float64}, y::AbstractVector{Float64})
	n = length(x_sorted)
	n == 0 && return (Float64[], Float64[])
	x_unique = Float64[]
	y_last = Float64[]
	@inbounds for i in 1:n
		# Skip this sample if the next shares the same coordinate (so we keep the last)
		if i < n && x_sorted[i + 1] == x_sorted[i]
			continue
		end
		push!(x_unique, x_sorted[i])
		push!(y_last, y[i])
	end
	return (x_unique, y_last)
end

"""
Vertical semi-axis `b` of one half-ellipse (`half ∈ (:upper, :lower)`) with fixed
horizontal semi-axis `a`, by two methods, returned as `(b_integral, b_2nd_deriv)`:

- `b_integral`: first `b` where the cumulative `w_tr` half-ellipse integral crosses
  to negative (`first_zero_crossing_coord(..., target_sign=-1)`).
- `b_2nd_deriv`: resample the integral onto a uniform grid, take the first derivative,
  and locate its first prominent local minimum (a zero-crossing of the 2nd derivative)
  via `_first_prominent_min_coord`.

`w_xz_slab` is the (x, z) `w_tr` cross-section through the thermal center;
`x_coords`/`z_coords` are its coordinate vectors; `dx`/`dz` the (uniform) spacings.
"""
function _ellipse_b_for_half(
	w_xz_slab::AbstractMatrix{<:Real},
	x_coords::AbstractVector{Float64},
	z_coords::AbstractVector{Float64},
	dx::Float64,
	dz::Float64,
	r_center::Float64,
	z_center::Float64,
	a::Float64;
	half::Symbol,
	max_z_semi_axis::Float64,
	min_prominence_fraction::Float64,
	uniform_grid_spacing::Float64,
	edge_cell_bins_per_dim::Int,
)
	# Restrict the cross-section to the requested vertical half (mirrors the Python
	# `.sel(z=slice(...))` crop); pad by one cell diagonal so boundary sub-cells are covered
	cell_diagonal = hypot(dx, dz)
	if half === :upper
		z_lo, z_hi = z_center, z_center + max_z_semi_axis + cell_diagonal
	elseif half === :lower
		z_lo, z_hi = z_center - max_z_semi_axis - cell_diagonal, z_center
	else
		error("half must be :upper or :lower, got $half")
	end
	z_ixs = findall(z -> z_lo <= z <= z_hi, z_coords)
	# Also crop horizontally to the ellipse footprint ±a (+pad) so we don't sort far-field cells
	x_ixs = findall(
		x -> (r_center - a - cell_diagonal) <= x <= (r_center + a + cell_diagonal),
		x_coords,
	)
	(isempty(z_ixs) || isempty(x_ixs)) && return (NaN, NaN)
	slab_half = w_xz_slab[x_ixs, z_ixs]
	x_sub = collect(Float64, x_coords[x_ixs])
	z_sub = collect(Float64, z_coords[z_ixs])

	# Ellipse (fixed a) entry value: the smallest b at which a sub-cell at (Δr, Δz)
	# enters, i.e. b ≥ |Δz| / sqrt(1 − (Δr/a)²); never inside the footprint → Inf
	a_squared = a * a
	function ellipse_entry_value_fn(dr_from_center, dz_from_center)
		abs(dr_from_center) < a || return Inf
		return abs(dz_from_center) /
			   sqrt(1 - (dr_from_center * dr_from_center) / a_squared)
	end

	z_semi_axis_sorted, integrated = _cumulative_area_integral_by_entry_value(
		slab_half,
		x_sub,
		z_sub,
		dx,
		dz,
		r_center,
		z_center,
		ellipse_entry_value_fn,
		max_z_semi_axis;
		edge_cell_bins_per_dim=edge_cell_bins_per_dim,
	)
	isempty(integrated) && return (NaN, NaN)

	# De-duplicate ties in b (groupby-last), so the coordinate is strictly increasing
	b_unique, integrated_unique = _dedup_keep_last(z_semi_axis_sorted, integrated)

	# Method A: first zero-crossing of the integral to negative
	b_integral = _first_zero_crossing(b_unique, integrated_unique; target_sign=-1)

	# Method B: resample onto a uniform grid, differentiate once, then take the first
	# prominent minimum of the derivative (= zero-crossing of the 2nd derivative)
	b_2nd_deriv = NaN
	if length(b_unique) >= 2
		uniform_b = _arange_exclusive(b_unique[1], b_unique[end], uniform_grid_spacing)
		if length(uniform_b) >= 3
			integrated_on_uniform =
				_interp_to_uniform(b_unique, integrated_unique, uniform_b)
			first_derivative =
				_gradient_uniform(integrated_on_uniform, uniform_grid_spacing)
			b_2nd_deriv = _first_prominent_min_coord(
				uniform_b,
				first_derivative;
				min_prominence_fraction=min_prominence_fraction,
			)
		end
	end

	return (b_integral, b_2nd_deriv)
end

"""
Four half-ellipse vertical semi-axes at one (thermal, time): upper and lower, each
by the integral-crossing and second-derivative methods. `w_field_box` is the 3-D
`w_tr` subset (x, y, z); `a` is the fixed horizontal semi-axis (= disk radius).
Returns `(upper_integral, lower_integral, upper_2nd, lower_2nd)`.
"""
function _ellipse_semi_axes_for_timestep(
	w_field_box::AbstractArray{<:Real, 3},
	x_coords_sub::AbstractVector{Float64},
	y_coords_sub::AbstractVector{Float64},
	z_coords_sub::AbstractVector{Float64},
	dx::Float64,
	dz::Float64,
	center_x::Float64,
	center_y::Float64,
	center_z::Float64,
	a::Float64;
	max_z_semi_axis::Float64,
	min_prominence_fraction::Float64,
	uniform_grid_spacing::Float64,
	edge_cell_bins_per_dim::Int,
)
	# Vertical (x, z) cross-section through y = center_y (w_tr.interp(y=y_smoothed))
	w_xz_slab = _interpolate_y_slab(w_field_box, y_coords_sub, center_y)

	upper_integral, upper_2nd = _ellipse_b_for_half(
		w_xz_slab,
		x_coords_sub,
		z_coords_sub,
		dx,
		dz,
		center_x,
		center_z,
		a;
		half=:upper,
		max_z_semi_axis=max_z_semi_axis,
		min_prominence_fraction=min_prominence_fraction,
		uniform_grid_spacing=uniform_grid_spacing,
		edge_cell_bins_per_dim=edge_cell_bins_per_dim,
	)
	lower_integral, lower_2nd = _ellipse_b_for_half(
		w_xz_slab,
		x_coords_sub,
		z_coords_sub,
		dx,
		dz,
		center_x,
		center_z,
		a;
		half=:lower,
		max_z_semi_axis=max_z_semi_axis,
		min_prominence_fraction=min_prominence_fraction,
		uniform_grid_spacing=uniform_grid_spacing,
		edge_cell_bins_per_dim=edge_cell_bins_per_dim,
	)
	return (
		upper_integral=upper_integral,
		lower_integral=lower_integral,
		upper_2nd=upper_2nd,
		lower_2nd=lower_2nd,
	)
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

function _compute_shapes_for_thermal!(
	sphere_radius::AbstractVector{Float64},        # length n_time, prefilled with NaN
	z_semi_axis_upper_integral::AbstractVector{Float64},   # ↳ four half-ellipse b's, likewise
	z_semi_axis_lower_integral::AbstractVector{Float64},
	z_semi_axis_upper_second_derivative::AbstractVector{Float64},
	z_semi_axis_lower_second_derivative::AbstractVector{Float64},
	traj::_TrajInput,
	rams_output_dir::String,
	gc,                                        # NamedTuple(x, y, z) — full grid
	dx::Float64,
	dy::Float64,
	dz::Float64,                               # uniform z spacing (WC z is grid-level gc.z)
	max_radius::Float64,
	max_z_semi_axis::Float64,
	min_prominence_fraction::Float64,
	uniform_z_semi_axis_grid_spacing::Float64,
	edge_cell_bins_per_dim::Int,
)
	# The box must cover the disk radius horizontally (±max_radius) and the full
	# vertical ellipse sweep (±max_z_semi_axis); pad z by one cell for boundary sub-cells
	z_half_extent = max_z_semi_axis + dz
	for ti in 1:length(traj.times)
		traj.valid[ti] || continue
		center_x = traj.x[ti]
		center_y = traj.y[ti]
		center_z = traj.z[ti]
		z_prop = traj.z_prop[ti]
		(isnan(center_x) || isnan(center_y) || isnan(center_z) || isnan(z_prop)) &&
			continue

		# Bounding-box subsetting around the center. WC lives on the RAMS grid levels
		# gc.z (as does center_z, from chain_w_maxima), so index/label z with gc.z.
		x_mask = abs.(gc.x .- center_x) .<= max_radius
		y_mask = abs.(gc.y .- center_y) .<= max_radius
		z_mask = abs.(gc.z .- center_z) .<= z_half_extent
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

		# w in the thermal's translating frame: w_tr = WC − z_propagation_speed
		w_field = wc_sub .- Float32(z_prop)
		x_coords_sub = gc.x[x_ixs]
		y_coords_sub = gc.y[y_ixs]
		z_coords_sub = gc.z[z_ixs]

		# Horizontal disk radius a = fixed horizontal semi-axis of the half-ellipses
		a = _find_zero_radius(
			w_field,
			center_x,
			center_y,
			center_z,
			x_coords_sub,
			y_coords_sub,
			z_coords_sub,
			dx,
			dy;
			max_radius=max_radius,
			edge_cell_bins_per_dim=edge_cell_bins_per_dim,
		)
		sphere_radius[ti] = a

		# Vertical half-ellipse semi-axes are only meaningful with a finite radius a
		isnan(a) && continue
		semi_axes = _ellipse_semi_axes_for_timestep(
			w_field,
			x_coords_sub,
			y_coords_sub,
			z_coords_sub,
			dx,
			dz,
			center_x,
			center_y,
			center_z,
			a;
			max_z_semi_axis=max_z_semi_axis,
			min_prominence_fraction=min_prominence_fraction,
			uniform_grid_spacing=uniform_z_semi_axis_grid_spacing,
			edge_cell_bins_per_dim=edge_cell_bins_per_dim,
		)
		z_semi_axis_upper_integral[ti] = semi_axes.upper_integral
		z_semi_axis_lower_integral[ti] = semi_axes.lower_integral
		z_semi_axis_upper_second_derivative[ti] = semi_axes.upper_2nd
		z_semi_axis_lower_second_derivative[ti] = semi_axes.lower_2nd
	end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
	compute_thermal_shapes(; thermal_trajectories_path, rams_output_dir,
							 rams_grid_path, output_path, max_radius=3000.0,
							 max_z_semi_axis=3000.0, min_prominence_fraction=0.05,
							 uniform_z_semi_axis_grid_spacing=50.0, edge_cell_bins_per_dim=4)

For each thermal/timestep, fit `w_tr = (WC − z_propagation_speed)` around the
smoothed thermal center and record several boundary estimates:
- `sphere_radius`: horizontal disk zero-crossing radius `a` (+ `sphere_radius_smoothed`)
- `z_semi_axis_{upper,lower}_integral_crossing`: vertical half-ellipse semi-axis by the
  integral zero-crossing method
- `z_semi_axis_{upper,lower}_second_derivative`: vertical half-ellipse semi-axis by the
  second-derivative (first prominent min of dI/db) method

All thermals are kept; usability is expressed via one `out_of_domain_*` flag per
fitted quantity (true ⟺ that shape exits the RAMS domain). Fitting anchors on the
smoothed centers (`x/y/z_smoothed`), matching the ellipsoid_tracking notebook.
"""
function compute_thermal_shapes(;
	thermal_trajectories_path::AbstractString,
	rams_output_dir::AbstractString,
	rams_grid_path::AbstractString,
	output_path::AbstractString,
	max_radius::Real=3000.0,
	max_z_semi_axis::Real=3000.0,
	min_prominence_fraction::Real=0.05,
	uniform_z_semi_axis_grid_spacing::Real=50.0,
	edge_cell_bins_per_dim::Integer=4,
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
	# WC lives on the RAMS grid levels gc.z. The ellipse area element assumes a uniform
	# vertical grid (matching the notebook's ct.spacing, which errors on non-uniform z).
	dz = gc.z[2] - gc.z[1]
	if !all(isapprox.(diff(gc.z), dz; rtol=1e-3))
		@warn "RAMS z grid is not uniform; ellipse dz uses gc.z[2]-gc.z[1]=$dz m (area element may be biased)"
	end

	# Domain bounds (RAMS grid extent)
	x_lo, x_hi = gc.x[1], gc.x[end]
	y_lo, y_hi = gc.y[1], gc.y[end]
	z_lo, z_hi = gc.z[1], gc.z[end]

	# Build per-thermal _TrajInput views anchored on the smoothed centers (the fit anchor)
	trajs = Vector{_TrajInput}(undef, n_thermals)
	for ti in 1:n_thermals
		x_col = tr.x_smoothed[:, ti]
		y_col = tr.y_smoothed[:, ti]
		z_col = tr.z_smoothed[:, ti]
		zp_col = tr.z_prop[:, ti]
		valid =
			.!isnan.(x_col) .& .!isnan.(y_col) .& .!isnan.(z_col) .& .!isnan.(zp_col)
		trajs[ti] =
			_TrajInput(tr.thermal_ixs[ti], valid, tr.times, x_col, y_col, z_col, zp_col)
	end

	# Allocate the fitted-quantity matrices (time, thermal_ix)
	sphere_radius = fill(NaN, n_time, n_thermals)
	z_semi_axis_upper_integral = fill(NaN, n_time, n_thermals)
	z_semi_axis_lower_integral = fill(NaN, n_time, n_thermals)
	z_semi_axis_upper_second_derivative = fill(NaN, n_time, n_thermals)
	z_semi_axis_lower_second_derivative = fill(NaN, n_time, n_thermals)

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
	println("Computing thermal shapes on $(Threads.nthreads()) threads…")
	pbar = Progress(n_thermals; output=stdout)
	pbar_lock = ReentrantLock()

	@tasks for ti in 1:n_thermals
		_compute_shapes_for_thermal!(
			view(sphere_radius, :, ti),
			view(z_semi_axis_upper_integral, :, ti),
			view(z_semi_axis_lower_integral, :, ti),
			view(z_semi_axis_upper_second_derivative, :, ti),
			view(z_semi_axis_lower_second_derivative, :, ti),
			trajs[ti],
			String(rams_output_dir),
			gc,
			dx,
			dy,
			dz,
			Float64(max_radius),
			Float64(max_z_semi_axis),
			Float64(min_prominence_fraction),
			Float64(uniform_z_semi_axis_grid_spacing),
			Int(edge_cell_bins_per_dim),
		)
		lock(pbar_lock) do
			next!(pbar)
		end
	end

	# Rolling-mean smoothed radius (unchanged)
	sphere_radius_smoothed = similar(sphere_radius)
	for ti in 1:n_thermals
		sphere_radius_smoothed[:, ti] = _rolling_mean_3(view(sphere_radius, :, ti))
	end

	# Per-quantity out-of-domain flags — keep all thermals; downstream filters. Each flag
	# is true iff that fitted shape exits the RAMS domain, evaluated at the smoothed center.
	# The horizontal disk extent (±a) is shared by every shape; the half-ellipses add their
	# vertical tip (upper pokes out the top cz+b>z_hi; lower the bottom cz−b<z_lo).
	out_of_domain_sphere_radius = falses(n_time, n_thermals)
	out_of_domain_upper_integral = falses(n_time, n_thermals)
	out_of_domain_lower_integral = falses(n_time, n_thermals)
	out_of_domain_upper_second_derivative = falses(n_time, n_thermals)
	out_of_domain_lower_second_derivative = falses(n_time, n_thermals)

	for ti in 1:n_thermals, i in 1:n_time
		trajs[ti].valid[i] || continue
		a = sphere_radius[i, ti]
		isnan(a) && continue
		center_x = tr.x_smoothed[i, ti]
		center_y = tr.y_smoothed[i, ti]
		center_z = tr.z_smoothed[i, ti]

		# Horizontal disk extent (shared by every fitted shape via the fixed semi-axis a)
		horizontal_oob =
			(center_x - a < x_lo) ||
			(center_x + a > x_hi) ||
			(center_y - a < y_lo) ||
			(center_y + a > y_hi)
		out_of_domain_sphere_radius[i, ti] = horizontal_oob

		b_upper_integral = z_semi_axis_upper_integral[i, ti]
		b_lower_integral = z_semi_axis_lower_integral[i, ti]
		b_upper_2nd = z_semi_axis_upper_second_derivative[i, ti]
		b_lower_2nd = z_semi_axis_lower_second_derivative[i, ti]
		out_of_domain_upper_integral[i, ti] =
			horizontal_oob || (isfinite(b_upper_integral) && center_z + b_upper_integral > z_hi)
		out_of_domain_lower_integral[i, ti] =
			horizontal_oob || (isfinite(b_lower_integral) && center_z - b_lower_integral < z_lo)
		out_of_domain_upper_second_derivative[i, ti] =
			horizontal_oob || (isfinite(b_upper_2nd) && center_z + b_upper_2nd > z_hi)
		out_of_domain_lower_second_derivative[i, ti] =
			horizontal_oob || (isfinite(b_lower_2nd) && center_z - b_lower_2nd < z_lo)
	end

	_write_thermal_ds(
		String(output_path),
		tr,
		sphere_radius,
		sphere_radius_smoothed,
		z_semi_axis_upper_integral,
		z_semi_axis_lower_integral,
		z_semi_axis_upper_second_derivative,
		z_semi_axis_lower_second_derivative,
		out_of_domain_sphere_radius,
		out_of_domain_upper_integral,
		out_of_domain_lower_integral,
		out_of_domain_upper_second_derivative,
		out_of_domain_lower_second_derivative,
	)
	println("Wrote $(output_path)")
	return nothing
end

# ──────────────────────────────────────────────────────────────────
# Write thermal_ds.nc (schema: thermal_trajectories.nc + sphere_radius{,_smoothed}
# + 4 half-ellipse semi-axes + 5 per-quantity out_of_domain flags)
# ──────────────────────────────────────────────────────────────────

function _write_thermal_ds(
	path::String,
	tr::NamedTuple,
	sphere_radius::Matrix{Float64},           # (n_time, n_thermals) — all thermals kept
	sphere_radius_smoothed::Matrix{Float64},
	z_semi_axis_upper_integral::Matrix{Float64},
	z_semi_axis_lower_integral::Matrix{Float64},
	z_semi_axis_upper_second_derivative::Matrix{Float64},
	z_semi_axis_lower_second_derivative::Matrix{Float64},
	out_of_domain_sphere_radius::BitMatrix,
	out_of_domain_upper_integral::BitMatrix,
	out_of_domain_lower_integral::BitMatrix,
	out_of_domain_upper_second_derivative::BitMatrix,
	out_of_domain_lower_second_derivative::BitMatrix,
)
	n_thermals = length(tr.thermal_ixs)
	n_time = length(tr.times)

	ref_time = DateTime(2000, 1, 1)
	time_seconds =
		Float64[Dates.value(Millisecond(t - ref_time)) / 1000.0 for t in tr.times]

	isfile(path) && rm(path)
	NCDataset(path, "c") do ds
		defDim(ds, "thermal_ix", n_thermals)
		defDim(ds, "time", n_time)

		# Keep the original (0-based) thermal ids — no thermals are dropped anymore
		defVar(ds, "thermal_ix", tr.thermal_ixs, ("thermal_ix",))
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

		defVar(ds, "x", tr.x, julia_dims; fillvalue=NaN)
		defVar(ds, "y", tr.y, julia_dims; fillvalue=NaN)
		defVar(ds, "z", tr.z, julia_dims; fillvalue=NaN)
		defVar(ds, "x_ix", tr.x_ix, julia_dims; fillvalue=-1)
		defVar(ds, "y_ix", tr.y_ix, julia_dims; fillvalue=-1)
		defVar(ds, "z_ix", tr.z_ix, julia_dims; fillvalue=-1)
		defVar(ds, "w_max", tr.w_max, julia_dims; fillvalue=NaN)
		defVar(ds, "x_smoothed", tr.x_smoothed, julia_dims; fillvalue=NaN)
		defVar(ds, "y_smoothed", tr.y_smoothed, julia_dims; fillvalue=NaN)
		defVar(ds, "z_smoothed", tr.z_smoothed, julia_dims; fillvalue=NaN)
		defVar(ds, "x_propagation_speed", tr.x_prop, julia_dims; fillvalue=NaN)
		defVar(ds, "y_propagation_speed", tr.y_prop, julia_dims; fillvalue=NaN)
		defVar(ds, "z_propagation_speed", tr.z_prop, julia_dims; fillvalue=NaN)
		defVar(ds, "time_ix", tr.time_ix, julia_dims; fillvalue=-1)
		defVar(ds, "t_minutes", tr.t_minutes, julia_dims; fillvalue=NaN)
		defVar(ds, "t_seconds", tr.t_seconds, julia_dims; fillvalue=NaN)
		defVar(ds, "dt_thermal_seconds", tr.dt_thermal_seconds, julia_dims; fillvalue=NaN)

		# Fitted boundary quantities: horizontal disk radius a + four half-ellipse semi-axes
		defVar(ds, "sphere_radius", sphere_radius, julia_dims; fillvalue=NaN)
		defVar(ds, "sphere_radius_smoothed", sphere_radius_smoothed, julia_dims; fillvalue=NaN)
		defVar(ds, "z_semi_axis_upper_integral_crossing", z_semi_axis_upper_integral, julia_dims; fillvalue=NaN)
		defVar(ds, "z_semi_axis_lower_integral_crossing", z_semi_axis_lower_integral, julia_dims; fillvalue=NaN)
		defVar(ds, "z_semi_axis_upper_second_derivative", z_semi_axis_upper_second_derivative, julia_dims; fillvalue=NaN)
		defVar(ds, "z_semi_axis_lower_second_derivative", z_semi_axis_lower_second_derivative, julia_dims; fillvalue=NaN)

		# One out-of-domain flag per fitted quantity (BitMatrix → Int8; NetCDF has no bool)
		defVar(ds, "out_of_domain_sphere_radius", Int8.(out_of_domain_sphere_radius), julia_dims)
		defVar(ds, "out_of_domain_upper_integral_crossing", Int8.(out_of_domain_upper_integral), julia_dims)
		defVar(ds, "out_of_domain_lower_integral_crossing", Int8.(out_of_domain_lower_integral), julia_dims)
		defVar(ds, "out_of_domain_upper_second_derivative", Int8.(out_of_domain_upper_second_derivative), julia_dims)
		defVar(ds, "out_of_domain_lower_second_derivative", Int8.(out_of_domain_lower_second_derivative), julia_dims)
	end
end
