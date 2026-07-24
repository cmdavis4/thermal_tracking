using ProgressMeter: Progress, next!
using Peaks: findminima, peakproms

# ══════════════════════════════════════════════════════════════════
# Shape fitting — ports/extensions of the `ellipsoid_tracking` notebook methods.
# Three sphere radii and four half-ellipse semi-axes are fit to the thermal-relative
# vertical velocity  w_tr = WC − z_propagation_speed  around each (smoothed) center:
#
#   sphere_radius_disk       : w_tr disk AREA integral on the horizontal slab at the
#                              w-max level → first zero-crossing (the 2-D disk method)
#   sphere_radius_3d         : w_tr VOLUME integral over the raw 3-D grid cells, swept by
#                              spherical radius → first zero-crossing (the full-3-D method)
#   sphere_radius_azimuthal  : same volume zero-crossing, but on the AZIMUTHAL AVERAGE
#                              w_tr(r, z) (φ-averaged about the vertical axis)
#   z_semi_axis_{upper,lower}_{integral_crossing,second_derivative} : half-ellipse vertical
#                              semi-axes fit on the azimuthal average w_tr(r, z), with fixed
#                              horizontal semi-axis a = sphere_radius_azimuthal
#
# All spheres/ellipses that integrate an axisymmetric (r, z) field weight each cell by the
# annulus volume element 2π·r, so they are true volume integrals of a body of revolution —
# consistent with the raw 3-D volume method. The disk method keeps its 2-D area weight.
# WC lives on the RAMS grid levels gc.z (as does the tracked center_z), so all z work is on
# grid-level z coordinates.
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# Generic sub-cell cumulative integrator over a 2-D (dim1, dim2) field
# (port of `_cumulative_integral_by_entry_value` in ellipsoid_tracking.py, extended with
#  an optional per-cell weight so it can integrate an axisymmetric body of revolution)
# ──────────────────────────────────────────────────────────────────

"""
Cumulative integral of a 2-D piecewise-constant field over a region parametrized by a
single swept scalar (radius for a disk/sphere; vertical semi-axis for an ellipse).

`slab` is 2-D in (dim1, dim2) order matching (`coords1`, `coords2`); `d1`, `d2` are the
uniform spacings; `(center1, center2)` is the region center. Each grid cell is split into
`edge_cell_bins_per_dim²` piecewise-constant sub-cells; `entry_value_fn(Δ1, Δ2)` returns the
smallest value of the swept parameter at which that sub-cell enters the region (may be `Inf`).
Sub-cells whose entry value exceeds `max_entry_value` (or whose field value is NaN) are dropped.

`weight_fn(coord1, coord2)` (optional) multiplies each sub-cell's contribution by a weight
evaluated at the sub-cell's *absolute* coordinates — pass `(r, z) -> 2π*r` to turn the plain
area integral (`weight_fn=nothing`) into the volume integral of an axisymmetric body about the
`coord1 = 0` axis.

Returns `(entry_value_sorted, integrated)`: the cumulative integral vs the ascending entry value.
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
	weight_fn=nothing,
)
	n1, n2 = size(slab)
	(n1 == 0 || n2 == 0) && return (Float64[], Float64[])

	# Sub-cell center offsets within a parent cell, as a fraction of cell width
	# (matches Python `(arange(n)+0.5)/n - 0.5`; e.g. n=4 → [-0.375,-0.125,0.125,0.375])
	edge_cell_bin_fractional_offsets = [
		((b - 0.5) / edge_cell_bins_per_dim) - 0.5 for
		b in 1:edge_cell_bins_per_dim
	]

	# Each sub-cell carries base area dA / edge_cell_bins_per_dim²
	subsampled_cell_dA = d1 * d2 / edge_cell_bins_per_dim^2

	# Per-sub-cell entry value + replicated (piecewise-constant) weighted field value; drop
	# sub-cells beyond the swept range, permanently outside the region (entry = Inf), or NaN.
	n_subcells = n1 * n2 * edge_cell_bins_per_dim * edge_cell_bins_per_dim
	entry_value = Vector{Float64}(undef, n_subcells)
	subcell_field = Vector{Float64}(undef, n_subcells)
	count_inside = 0
	@inbounds for i2 in 1:n2
		cell_center_2 = coords2[i2]
		for b2 in 1:edge_cell_bins_per_dim
			# Sub-cell dim-2 center and its offset from the region center
			subsampled_2 =
				cell_center_2 + edge_cell_bin_fractional_offsets[b2] * d2
			delta2 = subsampled_2 - center2
			for i1 in 1:n1
				cell_center_1 = coords1[i1]
				field_here = Float64(slab[i1, i2])
				# NaN = no data here (e.g. an azimuthal ring that partly left the box): skip.
				isnan(field_here) && continue
				for b1 in 1:edge_cell_bins_per_dim
					# Sub-cell dim-1 center and its offset from the region center
					subsampled_1 =
						cell_center_1 +
						edge_cell_bin_fractional_offsets[b1] * d1
					delta1 = subsampled_1 - center1
					entry = entry_value_fn(delta1, delta2)
					# Keep only sub-cells that enter the region within the swept range
					entry <= max_entry_value || continue
					count_inside += 1
					entry_value[count_inside] = entry
					subcell_field[count_inside] =
						weight_fn === nothing ? field_here :
						field_here * weight_fn(subsampled_1, subsampled_2)
				end
			end
		end
	end
	count_inside == 0 && return (Float64[], Float64[])

	# Sort sub-cells by entry value and accumulate the cumulative integral
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
	@inbounds for i in 1:(n-1)
		sign_change = sign(values[i+1]) - sign(values[i])
		hit =
			target_sign == 0 ? (sign_change != 0) :
			target_sign > 0 ? (sign_change > 0) : (sign_change < 0)
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
# Horizontal-slab interpolation (used by the disk-at-wmax method)
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
	weight_upper =
		(z_upper == z_lower) ? 0.0 : (target_z - z_lower) / (z_upper - z_lower)

	slab = Matrix{Float64}(undef, nx, ny)
	@inbounds for j in 1:ny, i in 1:nx
		lower = Float64(data[i, j, k_lower])
		upper = Float64(data[i, j, k_upper])
		slab[i, j] = lower + weight_upper * (upper - lower)
	end
	return slab
end

"""
Bilinear interpolation of one horizontal level `k` of a 3-D field at `(xq, yq)`, using the
ascending coordinate vectors `x_coords`, `y_coords`. Returns `NaN` for queries outside the
grid (so partial-coverage azimuthal rings drop the missing samples).
"""
function _bilinear_xy(
	data::AbstractArray{T, 3},
	x_coords::AbstractVector{Float64},
	y_coords::AbstractVector{Float64},
	xq::Float64,
	yq::Float64,
	k::Int,
) where {T}
	nx = length(x_coords)
	ny = length(y_coords)
	(xq < x_coords[1] || xq > x_coords[nx] || yq < y_coords[1] || yq > y_coords[ny]) &&
		return NaN
	i0 = clamp(searchsortedlast(x_coords, xq), 1, nx - 1)
	j0 = clamp(searchsortedlast(y_coords, yq), 1, ny - 1)
	tx = (xq - x_coords[i0]) / (x_coords[i0+1] - x_coords[i0])
	ty = (yq - y_coords[j0]) / (y_coords[j0+1] - y_coords[j0])
	@inbounds begin
		v00 = Float64(data[i0, j0, k])
		v10 = Float64(data[i0+1, j0, k])
		v01 = Float64(data[i0, j0+1, k])
		v11 = Float64(data[i0+1, j0+1, k])
	end
	return (1 - tx) * (1 - ty) * v00 +
		   tx * (1 - ty) * v10 +
		   (1 - tx) * ty * v01 +
		   tx * ty * v11
end

# ──────────────────────────────────────────────────────────────────
# Sphere method 1 — 2-D disk zero-crossing at the w-maximum level
# (port of the notebook's `w_tr_integral_vs_radius` + `first_zero_crossing_coord`)
# ──────────────────────────────────────────────────────────────────

"""
Zero-crossing radius of the `w_tr` cumulative *area* integral on the horizontal slab at the
thermal's w-maximum height: `w_tr.interp(z=center_z)` → cumulative ∫∫ w_tr dA over disks of
growing radius → first sign change (no interpolation, `first_zero_crossing_coord` target=None).
Returns the radius in meters, or `NaN` if no sign change is found.
"""
function _find_zero_crossing_disk_radius(
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

	# Horizontal w_tr slab at the w-max height (w_tr.interp(z=center_z))
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
	return _first_zero_crossing(radius_sorted, integrated; target_sign=0)
end

# ──────────────────────────────────────────────────────────────────
# Sphere method 2 — full 3-D volume zero-crossing over the raw grid cells
# (restore of the pre-`ddb67e2` volume-weighted radial cumsum, on grid-level z)
# ──────────────────────────────────────────────────────────────────

"""
Zero-crossing radius of the `w_tr` cumulative *volume* integral over the raw 3-D grid cells:
each cell contributes `w_tr · dx·dy·dz`, cells are ordered by their 3-D distance to the
center, and the first sign change of the running sum gives the radius (no interpolation,
matching the disk/azimuthal methods). Cells beyond `max_radius` are excluded from the sweep.
Returns the radius in meters, or `NaN` if no sign change is found.
"""
function _find_zero_crossing_sphere_radius_3d(
	data_sub::AbstractArray{T, 3},
	center_x::Float64,
	center_y::Float64,
	center_z::Float64,
	x_coords_sub::Vector{Float64},
	y_coords_sub::Vector{Float64},
	z_coords_sub::Vector{Float64},
	dx::Float64,
	dy::Float64,
	dz::Float64;
	max_radius::Float64=2000.0,
)::Float64 where {T}
	nx, ny, nz = size(data_sub)
	(nx == 0 || ny == 0 || nz == 0) && return NaN
	cell_volume = dx * dy * dz

	radial_distance = Vector{Float64}(undef, nx * ny * nz)
	volume_weighted = Vector{Float64}(undef, nx * ny * nz)
	count_inside = 0
	@inbounds for k in 1:nz
		dk = z_coords_sub[k] - center_z
		for j in 1:ny
			dj = y_coords_sub[j] - center_y
			dj2_plus_dk2 = dj * dj + dk * dk
			for i in 1:nx
				di = x_coords_sub[i] - center_x
				r = sqrt(di * di + dj2_plus_dk2)
				r <= max_radius || continue
				value = Float64(data_sub[i, j, k])
				isnan(value) && continue
				count_inside += 1
				radial_distance[count_inside] = r
				volume_weighted[count_inside] = value * cell_volume
			end
		end
	end
	count_inside == 0 && return NaN

	order = sortperm(@view radial_distance[1:count_inside])
	radius_sorted = radial_distance[1:count_inside][order]
	integrated = cumsum(volume_weighted[1:count_inside][order])
	return _first_zero_crossing(radius_sorted, integrated; target_sign=0)
end

# ──────────────────────────────────────────────────────────────────
# Azimuthal average of w_tr about the vertical axis through the center
# ──────────────────────────────────────────────────────────────────

"""
Azimuthally average the 3-D `w_tr` field about the vertical axis through
`(center_x, center_y)`, returning the axisymmetric field `w_tr(r, z)` as an
`(n_r, nz)` matrix on the radial grid `r_coords` (cell centers) and the field's own z
levels. For each `(r, z)` node the value is the mean over `n_azimuth` points around the
circle of radius `r` (bilinear in x/y at that z level); points that fall outside the box
are dropped, and an `(r, z)` with no valid azimuth becomes `NaN`.
"""
function _azimuthal_average_w_tr(
	data_sub::AbstractArray{T, 3},
	x_coords_sub::AbstractVector{Float64},
	y_coords_sub::AbstractVector{Float64},
	center_x::Float64,
	center_y::Float64,
	r_coords::AbstractVector{Float64},
	n_azimuth::Int,
) where {T}
	nz = size(data_sub, 3)
	n_r = length(r_coords)
	cos_phi = [cos(2π * (m - 1) / n_azimuth) for m in 1:n_azimuth]
	sin_phi = [sin(2π * (m - 1) / n_azimuth) for m in 1:n_azimuth]

	w_rz = fill(NaN, n_r, nz)
	@inbounds for k in 1:nz
		for ir in 1:n_r
			r = r_coords[ir]
			sum_val = 0.0
			n_valid = 0
			for m in 1:n_azimuth
				xq = center_x + r * cos_phi[m]
				yq = center_y + r * sin_phi[m]
				v = _bilinear_xy(data_sub, x_coords_sub, y_coords_sub, xq, yq, k)
				isnan(v) && continue
				sum_val += v
				n_valid += 1
			end
			w_rz[ir, k] = n_valid == 0 ? NaN : sum_val / n_valid
		end
	end
	return w_rz
end

# ──────────────────────────────────────────────────────────────────
# Sphere method 3 — volume zero-crossing on the azimuthal average
# ──────────────────────────────────────────────────────────────────

"""
Zero-crossing radius of the `w_tr(r, z)` cumulative *volume* integral (annulus weight `2π·r`)
over spheres of growing radius centered on the axis at height `center_z`. Returns the radius
in meters, or `NaN` if no sign change is found.
"""
function _find_zero_crossing_sphere_radius_azimuthal(
	w_rz::AbstractMatrix{<:Real},
	r_coords::AbstractVector{Float64},
	z_coords::AbstractVector{Float64},
	dr::Float64,
	dz::Float64,
	center_z::Float64,
	max_radius::Float64;
	edge_cell_bins_per_dim::Int=4,
)::Float64
	# A cell at (r, z) enters the sphere at radius sqrt(r² + (z − center_z)²)
	sphere_entry_value_fn(dr_from_axis, dz_from_center) =
		sqrt(dr_from_axis * dr_from_axis + dz_from_center * dz_from_center)
	radius_sorted, integrated = _cumulative_area_integral_by_entry_value(
		w_rz,
		collect(Float64, r_coords),
		collect(Float64, z_coords),
		dr,
		dz,
		0.0,             # the axis (r = 0) is the region center in the radial direction
		center_z,
		sphere_entry_value_fn,
		max_radius;
		edge_cell_bins_per_dim=edge_cell_bins_per_dim,
		weight_fn=(r, z) -> 2π * r,
	)
	isempty(integrated) && return NaN
	return _first_zero_crossing(radius_sorted, integrated; target_sign=0)
end

# ──────────────────────────────────────────────────────────────────
# Half-ellipse vertical semi-axis on the azimuthal average
# (integral zero-crossing + second-derivative prominent-min, volume weighted 2π·r)
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
		gradient[n] = (values[n] - values[n-1]) / h
		for i in 2:(n-1)
			gradient[i] = (values[i+1] - values[i-1]) / (2h)
		end
	end
	return gradient
end

"""
Linear interpolation of `(x_sorted, y)` (x strictly increasing) at the query points
`x_new`; queries outside the range clamp to the endpoints. Mirrors `np.interp`.
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
			j = searchsortedlast(x_sorted, xq)
			t = (xq - x_sorted[j]) / (x_sorted[j+1] - x_sorted[j])
			out[k] = y[j] + t * (y[j+1] - y[j])
		end
	end
	return out
end

"""`np.arange(lo, hi, step)`: evenly spaced values from `lo` (inclusive) up to but excluding `hi`."""
function _arange_exclusive(lo::Float64, hi::Float64, step::Float64)
	n = max(0, Int(ceil((hi - lo) / step)))
	return [lo + k * step for k in 0:(n-1)]
end

"""
Collapse ties in an ascending coordinate, keeping the *last* value in each run of
equal coordinates — matches the notebook's `groupby("z_semi_axis").last()`.
"""
function _dedup_keep_last(
	x_sorted::AbstractVector{Float64},
	y::AbstractVector{Float64},
)
	n = length(x_sorted)
	n == 0 && return (Float64[], Float64[])
	x_unique = Float64[]
	y_last = Float64[]
	@inbounds for i in 1:n
		if i < n && x_sorted[i+1] == x_sorted[i]
			continue
		end
		push!(x_unique, x_sorted[i])
		push!(y_last, y[i])
	end
	return (x_unique, y_last)
end

"""
Vertical semi-axis `b` of one half-ellipse (`half ∈ (:upper, :lower)`) fit on the azimuthal
average `w_rz(r, z)` with fixed horizontal semi-axis `a`, by two methods, returned as
`(b_integral, b_2nd_deriv)`:

- `b_integral`: first `b` where the cumulative volume integral (weight `2π·r`) over the
  half-ellipsoid of revolution crosses to negative.
- `b_2nd_deriv`: resample that integral onto a uniform grid, take the first derivative, and
  locate its first prominent local minimum (a zero-crossing of the 2nd derivative).
"""
function _ellipse_b_for_half_azimuthal(
	w_rz::AbstractMatrix{<:Real},
	r_coords::AbstractVector{Float64},
	z_coords::AbstractVector{Float64},
	dr::Float64,
	dz::Float64,
	z_center::Float64,
	a::Float64;
	half::Symbol,
	max_z_semi_axis::Float64,
	min_prominence_fraction::Float64,
	uniform_grid_spacing::Float64,
	edge_cell_bins_per_dim::Int,
)
	# Restrict to the requested vertical half; pad by one cell diagonal for boundary sub-cells
	cell_diagonal = hypot(dr, dz)
	if half === :upper
		z_lo, z_hi = z_center, z_center + max_z_semi_axis + cell_diagonal
	elseif half === :lower
		z_lo, z_hi = z_center - max_z_semi_axis - cell_diagonal, z_center
	else
		error("half must be :upper or :lower, got $half")
	end
	z_ixs = findall(z -> z_lo <= z <= z_hi, z_coords)
	# Crop radially to the ellipse footprint (r ∈ [0, a] + pad); r_coords start at the axis
	r_ixs = findall(r -> r <= a + cell_diagonal, r_coords)
	(isempty(z_ixs) || isempty(r_ixs)) && return (NaN, NaN)
	slab_half = w_rz[r_ixs, z_ixs]
	r_sub = collect(Float64, r_coords[r_ixs])
	z_sub = collect(Float64, z_coords[z_ixs])

	# Ellipse (fixed a) entry value: smallest b at which a cell at (r, Δz) enters, i.e.
	# b ≥ |Δz| / sqrt(1 − (r/a)²); outside the radial footprint → Inf
	a_squared = a * a
	function ellipse_entry_value_fn(dr_from_axis, dz_from_center)
		dr_from_axis < a || return Inf
		return abs(dz_from_center) /
			   sqrt(1 - (dr_from_axis * dr_from_axis) / a_squared)
	end

	z_semi_axis_sorted, integrated = _cumulative_area_integral_by_entry_value(
		slab_half,
		r_sub,
		z_sub,
		dr,
		dz,
		0.0,
		z_center,
		ellipse_entry_value_fn,
		max_z_semi_axis;
		edge_cell_bins_per_dim=edge_cell_bins_per_dim,
		weight_fn=(r, z) -> 2π * r,
	)
	isempty(integrated) && return (NaN, NaN)

	b_unique, integrated_unique =
		_dedup_keep_last(z_semi_axis_sorted, integrated)

	# Method A: first zero-crossing of the integral to negative
	b_integral =
		_first_zero_crossing(b_unique, integrated_unique; target_sign=-1)

	# Method B: resample onto a uniform grid, differentiate once, then take the first
	# prominent minimum of the derivative (= zero-crossing of the 2nd derivative)
	b_2nd_deriv = NaN
	if length(b_unique) >= 2
		uniform_b =
			_arange_exclusive(b_unique[1], b_unique[end], uniform_grid_spacing)
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
Four half-ellipse vertical semi-axes at one (thermal, time) from the azimuthal average
`w_rz(r, z)`: upper and lower, each by the integral-crossing and second-derivative methods.
`a` is the fixed horizontal semi-axis (= `sphere_radius_azimuthal`).
Returns `(upper_integral, lower_integral, upper_2nd, lower_2nd)`.
"""
function _ellipse_semi_axes_azimuthal(
	w_rz::AbstractMatrix{<:Real},
	r_coords::AbstractVector{Float64},
	z_coords::AbstractVector{Float64},
	dr::Float64,
	dz::Float64,
	z_center::Float64,
	a::Float64;
	max_z_semi_axis::Float64,
	min_prominence_fraction::Float64,
	uniform_grid_spacing::Float64,
	edge_cell_bins_per_dim::Int,
)
	upper_integral, upper_2nd = _ellipse_b_for_half_azimuthal(
		w_rz, r_coords, z_coords, dr, dz, z_center, a;
		half=:upper, max_z_semi_axis=max_z_semi_axis,
		min_prominence_fraction=min_prominence_fraction,
		uniform_grid_spacing=uniform_grid_spacing,
		edge_cell_bins_per_dim=edge_cell_bins_per_dim,
	)
	lower_integral, lower_2nd = _ellipse_b_for_half_azimuthal(
		w_rz, r_coords, z_coords, dr, dz, z_center, a;
		half=:lower, max_z_semi_axis=max_z_semi_axis,
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
# Reader for the (filtered) thermal_trajectories dataset
# ──────────────────────────────────────────────────────────────────

function _read_trajectories(path::String)
	NCDataset(path, "r") do ds
		thermal_ixs = Array(ds["thermal_ix"])
		times = Array(ds["time"])

		# Float and int variables were written with _FillValue=NaN / _FillValue=-1,
		# so NCDatasets returns Matrix{Union{Missing, T}} here. Collapse the Missing
		# branch with coalesce, matching the in-memory NaN-convention used everywhere else.
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

# HDF5 / libnetcdf are not guaranteed to be thread-safe — concurrent opens from parallel
# tasks can segfault the kernel. Serialize NetCDF reads through this lock; the fitting
# (the expensive part) still runs in parallel.
const _NC_IO_LOCK = ReentrantLock()

"""
Fit all boundary quantities for one thermal, writing into the per-thermal output views in
`out` (a NamedTuple of length-n_time vectors: `disk`, `sphere3d`, `azimuthal`, and the four
half-ellipse semi-axes `up_int`, `lo_int`, `up_2nd`, `lo_2nd`).
"""
function _compute_shapes_for_thermal!(
	out::NamedTuple,
	traj::_TrajInput,
	rams_output_dir::String,
	gc,
	dx::Float64,
	dy::Float64,
	dz::Float64,
	r_coords::Vector{Float64},
	dr::Float64,
	n_azimuth::Int,
	max_radius::Float64,
	max_z_semi_axis::Float64,
	min_prominence_fraction::Float64,
	uniform_z_semi_axis_grid_spacing::Float64,
	edge_cell_bins_per_dim::Int,
)
	# The box must cover the largest sphere (±max_radius in every direction) and the full
	# ellipse sweep (±max_z_semi_axis vertically); pad z by one cell for boundary sub-cells.
	z_half_extent = max(max_radius, max_z_semi_axis) + dz
	for ti in 1:length(traj.times)
		traj.valid[ti] || continue
		center_x = traj.x[ti]
		center_y = traj.y[ti]
		center_z = traj.z[ti]
		z_prop = traj.z_prop[ti]
		(
			isnan(center_x) ||
			isnan(center_y) ||
			isnan(center_z) ||
			isnan(z_prop)
		) && continue

		# Bounding-box subset around the center. WC lives on the RAMS grid levels gc.z.
		x_ixs = findall(abs.(gc.x .- center_x) .<= max_radius)
		y_ixs = findall(abs.(gc.y .- center_y) .<= max_radius)
		z_ixs = findall(abs.(gc.z .- center_z) .<= z_half_extent)
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

		# ── Sphere 1: 2-D disk zero-crossing at the w-max level ──
		out.disk[ti] = _find_zero_crossing_disk_radius(
			w_field, center_x, center_y, center_z,
			x_coords_sub, y_coords_sub, z_coords_sub, dx, dy;
			max_radius=max_radius, edge_cell_bins_per_dim=edge_cell_bins_per_dim,
		)

		# ── Sphere 2: full 3-D volume zero-crossing over the raw grid cells ──
		out.sphere3d[ti] = _find_zero_crossing_sphere_radius_3d(
			w_field, center_x, center_y, center_z,
			x_coords_sub, y_coords_sub, z_coords_sub, dx, dy, dz;
			max_radius=max_radius,
		)

		# ── Azimuthal average w_tr(r, z), shared by sphere 3 and the ellipses ──
		w_rz = _azimuthal_average_w_tr(
			w_field, x_coords_sub, y_coords_sub, center_x, center_y, r_coords, n_azimuth,
		)

		# ── Sphere 3: volume zero-crossing on the azimuthal average ──
		r_azimuthal = _find_zero_crossing_sphere_radius_azimuthal(
			w_rz, r_coords, z_coords_sub, dr, dz, center_z, max_radius;
			edge_cell_bins_per_dim=edge_cell_bins_per_dim,
		)
		out.azimuthal[ti] = r_azimuthal

		# ── Ellipses on the azimuthal average, fixed a = sphere_radius_azimuthal ──
		isnan(r_azimuthal) && continue
		semi_axes = _ellipse_semi_axes_azimuthal(
			w_rz, r_coords, z_coords_sub, dr, dz, center_z, r_azimuthal;
			max_z_semi_axis=max_z_semi_axis,
			min_prominence_fraction=min_prominence_fraction,
			uniform_grid_spacing=uniform_z_semi_axis_grid_spacing,
			edge_cell_bins_per_dim=edge_cell_bins_per_dim,
		)
		out.up_int[ti] = semi_axes.upper_integral
		out.lo_int[ti] = semi_axes.lower_integral
		out.up_2nd[ti] = semi_axes.upper_2nd
		out.lo_2nd[ti] = semi_axes.lower_2nd
	end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
	compute_thermal_shapes(; thermal_trajectories_path, rams_output_dir, rams_grid_path,
							 output_path, max_radius=3000.0, max_z_semi_axis=3000.0,
							 min_prominence_fraction=0.05, uniform_z_semi_axis_grid_spacing=50.0,
							 edge_cell_bins_per_dim=4, n_azimuth=72,
							 azimuthal_radial_spacing=nothing)

For each thermal/timestep, fit `w_tr = (WC − z_propagation_speed)` around the *smoothed*
thermal center and record several boundary estimates:
- `sphere_radius_disk`      : 2-D disk zero-crossing at the w-max level (+ `_smoothed`)
- `sphere_radius_3d`        : full 3-D volume zero-crossing over raw cells (+ `_smoothed`)
- `sphere_radius_azimuthal` : volume zero-crossing on the azimuthal average (+ `_smoothed`)
- `z_semi_axis_{upper,lower}_{integral_crossing,second_derivative}` : half-ellipse semi-axes
  on the azimuthal average, fixed horizontal semi-axis a = `sphere_radius_azimuthal`

All thermals are kept; usability is a per-quantity `out_of_domain_*` flag (true ⟺ that shape
exits the RAMS domain). `n_azimuth` sets the azimuthal-average resolution;
`azimuthal_radial_spacing` its radial cell size (defaults to the horizontal grid spacing).
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
	n_azimuth::Integer=72,
	azimuthal_radial_spacing::Union{Nothing, Real}=nothing,
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
	# WC lives on the RAMS grid levels gc.z. The volume/area elements assume a uniform
	# vertical grid (matching the notebook's ct.spacing, which errors on non-uniform z).
	dz = gc.z[2] - gc.z[1]
	if !all(isapprox.(diff(gc.z), dz; rtol=1e-3))
		@warn "RAMS z grid is not uniform; dz uses gc.z[2]-gc.z[1]=$dz m (volume element may be biased)"
	end

	# Radial grid for the azimuthal average (cell centers from dr/2 so 2π·r weights stay > 0)
	dr = azimuthal_radial_spacing === nothing ? dx : Float64(azimuthal_radial_spacing)
	r_coords = collect(dr / 2 : dr : Float64(max_radius))
	println(
		"  Azimuthal average: n_azimuth=$n_azimuth, dr=$(round(dr; digits=1)) m, " *
		"n_r=$(length(r_coords))",
	)

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
		trajs[ti] = _TrajInput(
			tr.thermal_ixs[ti], valid, tr.times, x_col, y_col, z_col, zp_col,
		)
	end

	# Allocate the fitted-quantity matrices (time, thermal_ix)
	sphere_radius_disk = fill(NaN, n_time, n_thermals)
	sphere_radius_3d = fill(NaN, n_time, n_thermals)
	sphere_radius_azimuthal = fill(NaN, n_time, n_thermals)
	z_semi_axis_upper_integral = fill(NaN, n_time, n_thermals)
	z_semi_axis_lower_integral = fill(NaN, n_time, n_thermals)
	z_semi_axis_upper_second_derivative = fill(NaN, n_time, n_thermals)
	z_semi_axis_lower_second_derivative = fill(NaN, n_time, n_thermals)

	# Verify RAMS dim order on the first RAMS file we encounter
	any_path = ""
	for t in trajs, (i, v) in enumerate(t.valid)
		v || continue
		p = datetime_to_rams_path(String(rams_output_dir), t.times[i])
		if isfile(p)
			any_path = p
			break
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
			(
				disk=view(sphere_radius_disk, :, ti),
				sphere3d=view(sphere_radius_3d, :, ti),
				azimuthal=view(sphere_radius_azimuthal, :, ti),
				up_int=view(z_semi_axis_upper_integral, :, ti),
				lo_int=view(z_semi_axis_lower_integral, :, ti),
				up_2nd=view(z_semi_axis_upper_second_derivative, :, ti),
				lo_2nd=view(z_semi_axis_lower_second_derivative, :, ti),
			),
			trajs[ti],
			String(rams_output_dir),
			gc,
			dx,
			dy,
			dz,
			r_coords,
			dr,
			Int(n_azimuth),
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

	# Rolling-mean smoothed radius for each sphere method
	smooth_columns(mat) = begin
		out = similar(mat)
		for ti in 1:n_thermals
			out[:, ti] = _rolling_mean_3(view(mat, :, ti))
		end
		out
	end
	sphere_radius_disk_smoothed = smooth_columns(sphere_radius_disk)
	sphere_radius_3d_smoothed = smooth_columns(sphere_radius_3d)
	sphere_radius_azimuthal_smoothed = smooth_columns(sphere_radius_azimuthal)

	# `_rolling_mean_3` skips NaNs, so at a gap/edge timestep it averages in the radii
	# of adjacent timesteps and emits a value even where the raw radius was NaN. That can
	# produce a smoothed radius at a timestep whose smoothed center (x/y/z) is missing —
	# a radius with no center to anchor its disk, which is meaningless and breaks the
	# normalize stage. Re-mask every smoothed radius to NaN wherever the center is absent,
	# so the smoothed radius is defined on exactly the timesteps the center is.
	center_missing =
		isnan.(tr.x_smoothed) .| isnan.(tr.y_smoothed) .| isnan.(tr.z_smoothed)
	sphere_radius_disk_smoothed[center_missing] .= NaN
	sphere_radius_3d_smoothed[center_missing] .= NaN
	sphere_radius_azimuthal_smoothed[center_missing] .= NaN

	# Per-quantity out-of-domain flags — keep all thermals; downstream filters. Each flag is
	# true iff that fitted shape exits the RAMS domain, evaluated at the smoothed center.
	oob_disk = falses(n_time, n_thermals)
	oob_3d = falses(n_time, n_thermals)
	oob_azimuthal = falses(n_time, n_thermals)
	oob_upper_integral = falses(n_time, n_thermals)
	oob_lower_integral = falses(n_time, n_thermals)
	oob_upper_second_derivative = falses(n_time, n_thermals)
	oob_lower_second_derivative = falses(n_time, n_thermals)

	# A sphere of radius R centered at (cx, cy, cz) leaves the domain if it pokes out any face.
	sphere_oob(cx, cy, cz, R) =
		isfinite(R) && (
			cx - R < x_lo || cx + R > x_hi ||
			cy - R < y_lo || cy + R > y_hi ||
			cz - R < z_lo || cz + R > z_hi
		)
	# The disk lives at a single level, so only its horizontal extent can leave the domain.
	horizontal_oob(cx, cy, R) =
		isfinite(R) &&
		(cx - R < x_lo || cx + R > x_hi || cy - R < y_lo || cy + R > y_hi)

	for ti in 1:n_thermals, i in 1:n_time
		trajs[ti].valid[i] || continue
		cx = tr.x_smoothed[i, ti]
		cy = tr.y_smoothed[i, ti]
		cz = tr.z_smoothed[i, ti]

		oob_disk[i, ti] = horizontal_oob(cx, cy, sphere_radius_disk[i, ti])
		oob_3d[i, ti] = sphere_oob(cx, cy, cz, sphere_radius_3d[i, ti])
		oob_azimuthal[i, ti] = sphere_oob(cx, cy, cz, sphere_radius_azimuthal[i, ti])

		# The ellipses share horizontal semi-axis a = sphere_radius_azimuthal; each half adds
		# its vertical tip (upper pokes out the top cz+b>z_hi, lower the bottom cz−b<z_lo).
		a = sphere_radius_azimuthal[i, ti]
		ellipse_horizontal_oob = horizontal_oob(cx, cy, a)
		b_up_int = z_semi_axis_upper_integral[i, ti]
		b_lo_int = z_semi_axis_lower_integral[i, ti]
		b_up_2nd = z_semi_axis_upper_second_derivative[i, ti]
		b_lo_2nd = z_semi_axis_lower_second_derivative[i, ti]
		oob_upper_integral[i, ti] =
			ellipse_horizontal_oob || (isfinite(b_up_int) && cz + b_up_int > z_hi)
		oob_lower_integral[i, ti] =
			ellipse_horizontal_oob || (isfinite(b_lo_int) && cz - b_lo_int < z_lo)
		oob_upper_second_derivative[i, ti] =
			ellipse_horizontal_oob || (isfinite(b_up_2nd) && cz + b_up_2nd > z_hi)
		oob_lower_second_derivative[i, ti] =
			ellipse_horizontal_oob || (isfinite(b_lo_2nd) && cz - b_lo_2nd < z_lo)
	end

	_write_thermal_ds(
		String(output_path),
		tr,
		(
			sphere_radius_disk=sphere_radius_disk,
			sphere_radius_disk_smoothed=sphere_radius_disk_smoothed,
			sphere_radius_3d=sphere_radius_3d,
			sphere_radius_3d_smoothed=sphere_radius_3d_smoothed,
			sphere_radius_azimuthal=sphere_radius_azimuthal,
			sphere_radius_azimuthal_smoothed=sphere_radius_azimuthal_smoothed,
			z_semi_axis_upper_integral_crossing=z_semi_axis_upper_integral,
			z_semi_axis_lower_integral_crossing=z_semi_axis_lower_integral,
			z_semi_axis_upper_second_derivative=z_semi_axis_upper_second_derivative,
			z_semi_axis_lower_second_derivative=z_semi_axis_lower_second_derivative,
		),
		(
			out_of_domain_sphere_radius_disk=oob_disk,
			out_of_domain_sphere_radius_3d=oob_3d,
			out_of_domain_sphere_radius_azimuthal=oob_azimuthal,
			out_of_domain_upper_integral_crossing=oob_upper_integral,
			out_of_domain_lower_integral_crossing=oob_lower_integral,
			out_of_domain_upper_second_derivative=oob_upper_second_derivative,
			out_of_domain_lower_second_derivative=oob_lower_second_derivative,
		),
	)
	println("Wrote $(output_path)")
	return nothing
end

# ──────────────────────────────────────────────────────────────────
# Write thermal_ds.nc (schema: trajectory pass-through + fitted boundary quantities
# `fitted` + per-quantity out-of-domain flags `oob`)
# ──────────────────────────────────────────────────────────────────

function _write_thermal_ds(
	path::String,
	tr::NamedTuple,
	fitted::NamedTuple,     # name => (n_time, n_thermals) Float64 matrix
	oob::NamedTuple,        # name => (n_time, n_thermals) BitMatrix
)
	n_thermals = length(tr.thermal_ixs)
	n_time = length(tr.times)

	ref_time = DateTime(2000, 1, 1)
	time_seconds = Float64[
		Dates.value(Millisecond(t - ref_time)) / 1000.0 for t in tr.times
	]

	isfile(path) && rm(path)
	NCDataset(path, "c") do ds
		defDim(ds, "thermal_ix", n_thermals)
		defDim(ds, "time", n_time)

		# Keep the original (0-based) thermal ids — no thermals are dropped here
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

		# Trajectory pass-through columns
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
		defVar(
			ds, "dt_thermal_seconds", tr.dt_thermal_seconds, julia_dims; fillvalue=NaN,
		)

		# Fitted boundary quantities (three sphere radii + smoothed, four ellipse semi-axes)
		for (name, mat) in pairs(fitted)
			defVar(ds, String(name), mat, julia_dims; fillvalue=NaN)
		end

		# One out-of-domain flag per fitted-boundary method (BitMatrix → Int8; NetCDF has no bool)
		for (name, flag) in pairs(oob)
			defVar(ds, String(name), Int8.(flag), julia_dims)
		end
	end
end
