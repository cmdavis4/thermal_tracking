"""
	local_maxima_3d(image; min_distance, threshold_abs)

Port of `skimage.feature.peak_local_max(image, min_distance, threshold_abs)`
for 3-D inputs. Returns a `Vector{NTuple{3,Int}}` of CartesianIndex-style
`(i1, i2, i3)` tuples using 1-based Julia indexing.

Algorithm:
  1. Dilate with a Chebyshev-`min_distance` box (max-filter, footprint `2·md+1`).
  2. A point is a candidate if `image[p] == dilated[p]` AND `image[p] ≥ threshold_abs`.
  3. Greedy suppression: sort candidates by value descending, accept one if no
	 already-accepted peak lies within Euclidean distance `< min_distance`.

`min_distance` is interpreted in *pixels*, isotropically — matching the Python
behavior on stretched-z grids.

Step 1 runs only on the bounding box of the above-threshold cells, padded by `md` on every
side and clipped to the array. Only above-threshold cells can become candidates. The
dilation window of such a cell reaches at most `md` cells in each direction, and that
window lies inside the padded box, so every candidate sees the same dilated value as a
full-array dilation. Updrafts above threshold fill a small part of a RAMS domain (in the
tracer_soundings run, about 17% of the volume, and only the lowest ~40% of levels), so the
crop cuts the dilation cost several-fold.
"""
function local_maxima_3d(
	image::AbstractArray{T, 3};
	min_distance::Int,
	threshold_abs::Real,
)::Vector{NTuple{3, Int}} where {T}
	md = min_distance

	# Bounding box of the cells that meet the threshold; with none, there are no peaks
	thr = T(threshold_abs)
	sz = size(image)
	box_lo = [typemax(Int), typemax(Int), typemax(Int)]
	box_hi = [0, 0, 0]
	@inbounds for k in 1:sz[3], j in 1:sz[2], i in 1:sz[1]
		image[i, j, k] >= thr || continue
		for (dim, index) in enumerate((i, j, k))
			box_lo[dim] = min(box_lo[dim], index)
			box_hi[dim] = max(box_hi[dim], index)
		end
	end
	box_hi[1] == 0 && return NTuple{3, Int}[]

	# Pad the box by the dilation half-width so every candidate's window lies inside it
	crop_ranges = ntuple(
		dim -> max(1, box_lo[dim] - md):min(sz[dim], box_hi[dim] + md),
		3,
	)
	cropped_image = image[crop_ranges...]
	# Offset that converts a cropped-array index back to a full-array index
	crop_offsets = ntuple(dim -> first(crop_ranges[dim]) - 1, 3)

	# Step 1: separable 1D max filter along each axis (Lemire monotonic-deque,
	# amortized O(N) per line), on the cropped box only.
	dilated = _dilate_3d(cropped_image, md)

	# Step 2: collect (value, i, j, k) for plateau-top pixels above threshold, in full-array
	# indices. The loop visits cells in the same (k, j, i) order as a full-array scan, so
	# the candidate order, and hence the tie order in the sort below, is unchanged.
	cropped_size = size(cropped_image)
	candidates = Tuple{T, Int, Int, Int}[]
	@inbounds for k in 1:cropped_size[3], j in 1:cropped_size[2], i in 1:cropped_size[1]
		v = cropped_image[i, j, k]
		if v >= thr && v == dilated[i, j, k]
			push!(
				candidates,
				(v, i + crop_offsets[1], j + crop_offsets[2], k + crop_offsets[3]),
			)
		end
	end

	isempty(candidates) && return NTuple{3, Int}[]

	# Step 3: greedy Euclidean-distance suppression, highest-value first
	sort!(candidates; by=c -> -c[1])

	md2 = Float64(md)^2
	accepted = NTuple{3, Int}[]
	@inbounds for c in candidates
		_, i, j, k = c
		ok = true
		for a in accepted
			di = i - a[1]
			dj = j - a[2]
			dk = k - a[3]
			if di*di + dj*dj + dk*dk < md2
				ok = false
				break
			end
		end
		ok && push!(accepted, (i, j, k))
	end

	return accepted
end

# ──────────────────────────────────────────────────────────────────
# Separable Lemire max filter over a half-width `md` centered box
# ──────────────────────────────────────────────────────────────────

"""
1D max filter along a single line, centered window of half-width `md`
(width 2·md+1), boundaries clipped. Lemire's monotonic-deque algorithm:
amortized O(N) comparisons. `q` is a reusable index buffer; `head` tracks
the logical front to avoid O(N) `popfirst!`.
"""
@inline function _max_line!(
	out::AbstractVector{T},
	v::AbstractVector{T},
	md::Int,
	q::Vector{Int},
) where {T}
	resize!(q, 0)
	head = 1
	N = length(v)
	@inbounds for i in 1:N
		vi = v[i]
		while length(q) >= head && v[q[end]] <= vi
			pop!(q)
		end
		push!(q, i)
		o = i - md
		if 1 <= o <= N
			lo = i - 2 * md
			while head <= length(q) && q[head] < lo
				head += 1
			end
			out[o] = v[q[head]]
		end
	end
	# Tail: windows clipped at the right boundary (right edge fixed at N)
	@inbounds for o in max(1, N - md + 1):N
		lo = o - md
		while head <= length(q) && q[head] < lo
			head += 1
		end
		out[o] = v[q[head]]
	end
	return nothing
end

function _max_filter_axis!(
	out::Array{T, 3},
	arr::AbstractArray{T, 3},
	md::Int,
	dim::Int,
) where {T}
	sz = size(arr)
	if dim == 1
		Threads.@threads for k in 1:sz[3]
			q = Int[]
			sizehint!(q, 2 * md + 2)
			for j in 1:sz[2]
				_max_line!(view(out, :, j, k), view(arr, :, j, k), md, q)
			end
		end
	elseif dim == 2
		Threads.@threads for k in 1:sz[3]
			q = Int[]
			sizehint!(q, 2 * md + 2)
			for i in 1:sz[1]
				_max_line!(view(out, i, :, k), view(arr, i, :, k), md, q)
			end
		end
	else
		Threads.@threads for j in 1:sz[2]
			q = Int[]
			sizehint!(q, 2 * md + 2)
			for i in 1:sz[1]
				_max_line!(view(out, i, j, :), view(arr, i, j, :), md, q)
			end
		end
	end
	return out
end

function _dilate_3d(arr::AbstractArray{T, 3}, md::Int) where {T}
	t1 = similar(arr)
	t2 = similar(arr)
	t3 = similar(arr)
	_max_filter_axis!(t1, arr, md, 1)
	_max_filter_axis!(t2, t1, md, 2)
	_max_filter_axis!(t3, t2, md, 3)
	return t3
end
