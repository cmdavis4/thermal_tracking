using NCDatasets
using ProgressMeter: Progress, progress_map, progress_pmap
using Distributed: addprocs, rmprocs, CachingPool, remotecall_eval
using Dates

# ──────────────────────────────────────────────────────────────────
# Print / println that force flushing to stdout (nicer under papermill / juliacall)
# ──────────────────────────────────────────────────────────────────

function println(x)
	Base.println(x)
	flush(stdout)
	flush(stderr)
end

function print(x)
	Base.print(x)
	flush(stdout)
	flush(stderr)
end

# ──────────────────────────────────────────────────────────────────
# Thermal track dataclasses (used by compute_thermal_shapes + normalize_thermals)
# ──────────────────────────────────────────────────────────────────

struct ThermalTimeStep
	center_x::Float64
	center_y::Float64
	center_z::Float64
	sphere_radius::Float64
	dt_thermal_seconds::Float64
	time::DateTime
end

struct ThermalTrack
	thermal_ix::Int   # 0-based, matching Python convention
	timesteps::Vector{ThermalTimeStep}
end

# ──────────────────────────────────────────────────────────────────
# Composited-variable specs (used by normalize_thermals)
# ──────────────────────────────────────────────────────────────────

"""
Specification for one variable to composite onto the normalized spherical grid.

- `name`    — output variable name written to the normalized dataset.
- `inputs`  — RAMS variable(s) read from the output file to build it.
- `combine` — maps a `Dict{String, Array{Float64,3}}` (each input's native-grid subset)
              to the combined native-grid field, evaluated *before* interpolation.

Derived quantities are computed on the native RAMS grid and interpolated once. This is the
physically correct order for nonlinear quantities (a nonlinear function of interpolated
fields ≠ the interpolated nonlinear field); for linear sums the two orders agree, but doing
it here also cuts memory/runtime by storing one output field instead of every input species.

All inputs of a spec must share the same Arakawa-C staggering (all scalar, or all on the
same momentum axis), since they're combined pointwise on one grid — see `_staggered_axis`.
"""
struct CompositeSpec
	name::String
	inputs::Vector{String}
	combine::Function
end

"""Pass-through composite: interpolate RAMS variable `name` as-is."""
function composite(name::AbstractString)::CompositeSpec
	nm = String(name)
	CompositeSpec(nm, [nm], native -> native[nm])
end

"""
Derived composite `name` computed on the native RAMS grid from `inputs` before
interpolation. `combine` receives a `Dict` mapping each input name to its native-grid
(subset) array and returns the combined native-grid array, e.g.
`derived("RH", ["THETA", "PI", "RV"], native -> compute_rh(native["THETA"], ...))`.
"""
function derived(name::AbstractString, inputs, combine::Function)::CompositeSpec
	CompositeSpec(String(name), String.(collect(inputs)), combine)
end

"""
Convenience derived composite that sums `inputs` into one field (e.g. total ice mixing
ratio from the individual ice-species mixing ratios): `derived_sum("R_ice", [...])`.
"""
function derived_sum(name::AbstractString, inputs)::CompositeSpec
	input_names = String.(collect(inputs))
	CompositeSpec(String(name), input_names, native -> sum(native[v] for v in input_names))
end

# ──────────────────────────────────────────────────────────────────
# RAMS grid / path helpers
# ──────────────────────────────────────────────────────────────────

"""
Derive Arakawa-C momentum levels from thermodynamic levels as the midpoints between
consecutive T-levels (top level extrapolated with the last spacing). Exact on a uniform
grid (`m[k] = t[k] + dz/2`); only approximate on a stretched grid, where the true `zmn`
levels should be read from the RAMS header instead. Used only as a fallback for grid
files written before momentum levels were stored.
"""
function _derive_momentum_levels(t_levels::Vector{Float64})::Vector{Float64}
	n = length(t_levels)
	n == 1 && return copy(t_levels)
	m = similar(t_levels)
	for k in 1:(n-1)
		m[k] = 0.5 * (t_levels[k] + t_levels[k+1])
	end
	# Top momentum level: extrapolate one half-step above the last thermo level
	m[n] = t_levels[n] + 0.5 * (t_levels[n] - t_levels[n-1])
	return m
end

"""
Read the RAMS grid coordinates. Returns thermodynamic (scalar) levels `x, y, z` and the
Arakawa-C momentum levels `xm, ym, zm` (where staggered variables live: U on `xm`, V on
`ym`, W on `zm`). Momentum levels are read directly if present; for older grid files that
predate them they are derived from the thermo levels (see `_derive_momentum_levels`).
"""
function read_rams_grid(path::String)
	NCDataset(path, "r") do ds
		x = Float64.(ds["x"][:])
		y = Float64.(ds["y"][:])
		z = Float64.(ds["z"][:])
		if haskey(ds, "xm") && haskey(ds, "ym") && haskey(ds, "zm")
			xm = Float64.(ds["xm"][:])
			ym = Float64.(ds["ym"][:])
			zm = Float64.(ds["zm"][:])
		else
			@warn "rams_grid.nc has no momentum levels (xm/ym/zm); deriving them from the " *
				  "thermo levels (exact only for a uniform grid). Regenerate with " *
				  "write_rams_grid to store the true RAMS momentum levels."
			xm = _derive_momentum_levels(x)
			ym = _derive_momentum_levels(y)
			zm = _derive_momentum_levels(z)
		end
		(x=x, y=y, z=z, xm=xm, ym=ym, zm=zm)
	end
end

function datetime_to_rams_path(
	rams_dir::String,
	dt::DateTime;
	lite::Bool=true,
)::String
	datestr = Dates.format(dt, dateformat"yyyy-mm-dd-HHMMSS")
	joinpath(rams_dir, "a-$(lite ? "L" : "A")-$(datestr)-g1.h5")
end

"""
Parse the datetime encoded in a RAMS filename of the form `a-L-YYYY-mm-dd-HHMMSS-g1.h5`
(lite) or `a-A-YYYY-mm-dd-HHMMSS-g1.h5` (full-analysis). Both output kinds are accepted so
the pipeline can be driven off either via a single `lite` flag.
"""
function rams_path_to_datetime(path::String)::DateTime
	name = basename(path)
	# name = a-{L,A}-YYYY-mm-dd-HHMMSS-g1.h5 (lite or full-analysis)
	m = match(r"^a-[LA]-(\d{4})-(\d{2})-(\d{2})-(\d{6})-g1\.h5$", name)
	m === nothing &&
		error("RAMS filename does not match expected pattern: $name")
	y, mo, d, hms = m.captures
	DateTime(
		parse(Int, y),
		parse(Int, mo),
		parse(Int, d),
		parse(Int, hms[1:2]),
		parse(Int, hms[3:4]),
		parse(Int, hms[5:6]),
	)
end

"""
Enumerate RAMS files in `dir`, sorted by datetime. `lite=true` (default) lists the
high-frequency lite files `a-L-*-g1.h5`; `lite=false` lists the full-analysis files
`a-A-*-g1.h5`. Selecting the kind here lets a single top-level flag drive which output
the whole tracking pipeline reads.
"""
function list_rams_files(dir::String; lite::Bool=true)::Vector{String}
	all_files = readdir(dir; join=true)
	file_kind_prefix = lite ? "a-L-" : "a-A-"
	rams = filter(
		p -> occursin(Regex("^$(file_kind_prefix).*-g1\\.h5\$"), basename(p)),
		all_files,
	)
	sort!(rams; by=rams_path_to_datetime)
	return rams
end

"""
Read one `__<name>` block from an open RAMS header file. Advances `io` past the block.
Returns `nothing` if the marker isn't found.
"""
function _read_header_block(
	io::IO,
	marker::String,
)::Union{Nothing, Vector{Float64}}
	seekstart(io)
	for line in eachline(io)
		if strip(line) == marker
			n = parse(Int, strip(readline(io)))
			return [parse(Float64, strip(readline(io))) for _ in 1:n]
		end
	end
	return nothing
end

"""
Write `rams_grid.nc` with shifted x/y coords matching `xarrams.calculate_derived_variables`
(which does `storm_ds["x"] -= min(storm_ds["x"])` so the grid starts at zero).

Stores both the thermodynamic (scalar) levels `x, y, z` (from `__[xyz]tn01`) and the
Arakawa-C momentum levels `xm, ym, zm` (from `__[xyz]mn01`), so staggered variables (U on
`xm`, V on `ym`, W on `zm`) can be placed on their true grid downstream.

Coord values come from the `-head.txt` file paired with the first RAMS `.h5` file —
RAMS HDF5 outputs only carry phony dimensions, not coord arrays.
"""
function write_rams_grid(
	rams_output_dir::AbstractString,
	output_path::AbstractString;
	lite::Bool=true,
)
	rams_output_dir = String(rams_output_dir)
	output_path = String(output_path)

	# The grid coords come from the `-head.txt` paired with the first RAMS file; `lite`
	# only picks which file kind we look for so the search works when a run has just one
	# kind present (e.g. analysis-only). Both kinds share the same grid.
	files = list_rams_files(rams_output_dir; lite=lite)
	isempty(files) && error("No RAMS files found in $rams_output_dir")

	header_path = replace(files[1], r"g\d+\.h5$" => "head.txt")
	isfile(header_path) || error("Expected header file not found: $header_path")

	xt, yt, zt, xm, ym, zm = open(header_path, "r") do io
		xtn = _read_header_block(io, "__xtn01")
		ytn = _read_header_block(io, "__ytn01")
		ztn = _read_header_block(io, "__ztn01")
		xmn = _read_header_block(io, "__xmn01")
		ymn = _read_header_block(io, "__ymn01")
		zmn = _read_header_block(io, "__zmn01")
		(xtn === nothing || ytn === nothing || ztn === nothing) &&
			error("Missing __xtn01/__ytn01/__ztn01 block in $header_path")
		(xmn === nothing || ymn === nothing || zmn === nothing) && error(
			"Missing __xmn01/__ymn01/__zmn01 (momentum) block in $header_path",
		)
		(xtn, ytn, ztn, xmn, ymn, zmn)
	end

	# Thermo and momentum axes must be index-aligned (same length) so one bounding-box index
	# range applies to both — xm[k] is the U-point paired with scalar point xt[k], etc.
	(
		length(xm) == length(xt) &&
		length(ym) == length(yt) &&
		length(zm) == length(zt)
	) || error(
		"RAMS thermo/momentum level counts differ; cannot align staggered coordinates",
	)

	# Shift x/y so the thermo grid starts at zero. Apply the SAME offset to the momentum
	# levels so xm/xt (and ym/yt) stay in one frame — i.e. xm[k] = xt[k] + dx/2 is preserved.
	# z is not shifted.
	x_offset = minimum(xt)
	y_offset = minimum(yt)
	xt .-= x_offset
	yt .-= y_offset
	xm .-= x_offset
	ym .-= y_offset

	NCDataset(output_path, "c") do ds
		defDim(ds, "x", length(xt))
		defDim(ds, "y", length(yt))
		defDim(ds, "z", length(zt))
		defVar(ds, "x", xt, ("x",))
		defVar(ds, "y", yt, ("y",))
		defVar(ds, "z", zt, ("z",))
		# Momentum levels share the thermo dimensions (same length, index-aligned)
		defVar(ds, "xm", xm, ("x",))
		defVar(ds, "ym", ym, ("y",))
		defVar(ds, "zm", zm, ("z",))
	end
	return output_path
end

"""
Verify that the RAMS variable has the expected 3D shape (x, y, z) in Julia.

RAMS lite files have Python/ncdump dims (z, y, x). NCDatasets reverses this,
so Julia sees (x, y, z).
"""
function verify_rams_dims(
	rams_path::String,
	x_coords,
	y_coords,
	z_coords,
	test_var::String,
)
	NCDataset(rams_path, "r") do ds
		sz = size(ds[test_var])
		nx, ny, nz = length(x_coords), length(y_coords), length(z_coords)
		expected = (nx, ny, nz)
		if sz != expected
			error(
				"RAMS variable '$test_var' has Julia shape $sz, expected $expected (x,y,z).",
			)
		end
		@info "RAMS dimension check passed: $test_var has shape $sz = (nx=$nx, ny=$ny, nz=$nz)"
	end
end

# ──────────────────────────────────────────────────────────────────
# Spatial subsetting
# ──────────────────────────────────────────────────────────────────

"""Find the index range in `coords` (sorted ascending) covering [lo, hi]."""
function coord_range(coords::Vector{Float64}, lo::Float64, hi::Float64)
	i0 = max(1, searchsortedfirst(coords, lo) - 1)
	i1 = min(length(coords), searchsortedlast(coords, hi) + 1)
	return i0:i1
end

function compute_spatial_subset(
	track::ThermalTrack,
	x_coords::Vector{Float64},
	y_coords::Vector{Float64},
	z_coords::Vector{Float64};
	pad=1.5,
	xm_coords::Union{Nothing, Vector{Float64}}=nothing,
	ym_coords::Union{Nothing, Vector{Float64}}=nothing,
	zm_coords::Union{Nothing, Vector{Float64}}=nothing,
)
	# `pad` is the bounding-box half-extent in thermal radii. Callers should set it to the
	# normalized grid's r_norm_max so the whole (r_norm ≤ r_norm_max) query sphere is covered;
	# a smaller pad clips the outer shells and interpolates them to NaN.
	max_sr = maximum(ts.sphere_radius for ts in track.timesteps)

	x_lo = minimum(ts.center_x for ts in track.timesteps) - max_sr * pad
	x_hi = maximum(ts.center_x for ts in track.timesteps) + max_sr * pad
	y_lo = minimum(ts.center_y for ts in track.timesteps) - max_sr * pad
	y_hi = maximum(ts.center_y for ts in track.timesteps) + max_sr * pad
	z_lo = minimum(ts.center_z for ts in track.timesteps) - max_sr * pad
	z_hi = maximum(ts.center_z for ts in track.timesteps) + max_sr * pad

	# Bounding-box index range is computed on the thermo grid and reused for the momentum
	# grid (index-aligned, same length). The momentum coords carry the true staggered
	# heights; fall back to the thermo coords if none were supplied.
	x_ixs = coord_range(x_coords, x_lo, x_hi)
	y_ixs = coord_range(y_coords, y_lo, y_hi)
	z_ixs = coord_range(z_coords, z_lo, z_hi)

	xm = xm_coords === nothing ? x_coords : xm_coords
	ym = ym_coords === nothing ? y_coords : ym_coords
	zm = zm_coords === nothing ? z_coords : zm_coords

	(
		x_ixs=x_ixs,
		y_ixs=y_ixs,
		z_ixs=z_ixs,
		x_sub=x_coords[x_ixs],
		y_sub=y_coords[y_ixs],
		z_sub=z_coords[z_ixs],
		xm_sub=xm[x_ixs],
		ym_sub=ym[y_ixs],
		zm_sub=zm[z_ixs],
	)
end

# ──────────────────────────────────────────────────────────────────
# Chunk-aware reads and per-file parallel dispatch
# (shared by chain_w_maxima, compute_thermal_shapes, and normalize_thermals)
# ──────────────────────────────────────────────────────────────────

"""
	_chunk_aligned_blocks(subsets, chunk_size_x, chunk_size_y, nx, ny)

List the horizontal `(x_range, y_range)` index blocks to read so that every HDF5 chunk that
any box in `subsets` touches is decompressed exactly ONCE.

Why this matters: RAMS lite variables are stored in full-column chunks (all z levels × a
small x–y tile), and HDF5 must decompress a whole chunk to return any part of it. The older
thermal-major loop read each thermal's box separately, so a chunk shared by several thermals
(or by consecutive reads) was decompressed again every time. Here the union of needed chunks
is read once per file and variable, and each thermal's box is then cut from memory.

Blocks are chunk-aligned, so no read straddles a chunk it does not need. Needed tiles that
sit next to each other in x within one tile row (same y tile) merge into a single read,
which cuts per-call overhead without decompressing anything extra.
"""
function _chunk_aligned_blocks(
	subsets,
	chunk_size_x::Int,
	chunk_size_y::Int,
	nx::Int,
	ny::Int,
)
	# Chunk tiles (0-based tile indices) that at least one subset box overlaps
	needed_tiles = Set{Tuple{Int, Int}}()
	for subset in subsets
		tile_x_range =
			((first(subset.x_ixs) - 1) ÷ chunk_size_x):((last(subset.x_ixs) - 1) ÷ chunk_size_x)
		tile_y_range =
			((first(subset.y_ixs) - 1) ÷ chunk_size_y):((last(subset.y_ixs) - 1) ÷ chunk_size_y)
		for tile_x in tile_x_range, tile_y in tile_y_range
			push!(needed_tiles, (tile_x, tile_y))
		end
	end

	blocks = Tuple{UnitRange{Int}, UnitRange{Int}}[]
	for tile_y in sort(unique(tile[2] for tile in needed_tiles))
		# Native y indices covered by this tile row (the last row may be partial)
		y_range = (tile_y * chunk_size_y + 1):min((tile_y + 1) * chunk_size_y, ny)

		# Group this row's needed x tiles into runs of consecutive tiles
		row_tiles_x = sort([tile[1] for tile in needed_tiles if tile[2] == tile_y])
		run_start_tile = row_tiles_x[1]
		previous_tile = run_start_tile
		for tile_x in row_tiles_x[2:end]
			if tile_x != previous_tile + 1
				# The run ended at previous_tile: emit it and start a new run here
				push!(
					blocks,
					(
						(run_start_tile * chunk_size_x + 1):min((previous_tile + 1) * chunk_size_x, nx),
						y_range,
					),
				)
				run_start_tile = tile_x
			end
			previous_tile = tile_x
		end
		# Emit the row's final run
		push!(
			blocks,
			(
				(run_start_tile * chunk_size_x + 1):min((previous_tile + 1) * chunk_size_x, nx),
				y_range,
			),
		)
	end
	return blocks
end

"""
	_load_blocks(variable, blocks, field_buffer)

Read `blocks` of `variable` (every z level) into a full-domain array, and return that array.

`field_buffer` is reused when its element type matches what the variable returns; otherwise
(or when it is `nothing`) a new full-domain array is allocated. Reuse avoids allocating a
~300 MB array per variable per file. Cells outside `blocks` keep stale or uninitialized
values, which is safe because callers only slice boxes that the blocks cover.

Values are read through the same CF-variable indexing as the older per-box reads, so they
are bitwise identical to what that code saw.
"""
function _load_blocks(variable, blocks, field_buffer)
	for (x_range, y_range) in blocks
		# Full columns: the chunks span every z level, so a z subset would not save work
		block = variable[x_range, y_range, :]
		if field_buffer === nothing || eltype(field_buffer) != eltype(block)
			field_buffer = Array{eltype(block)}(undef, size(variable))
		end
		field_buffer[x_range, y_range, :] .= block
	end
	return field_buffer
end

"""
	_map_over_workers(process_job, jobs; n_workers, progress)

Apply `process_job` to every element of `jobs` and return the results in job order.

Every pipeline stage that reads RAMS files splits its work into one job per file, because
files are independent. With `n_workers <= 1` the jobs run in this process. Otherwise they
run on `n_workers` fresh Distributed worker processes, which are removed afterwards.

Why processes and not threads: NCDatasets serializes every NetCDF call behind one global
lock, so threads would read (and decompress) one file at a time. The tracking kernel is also
single-threaded, because PyCall segfaults in a multi-threaded kernel.

Workers inherit this process's environment (HDF5_PLUGIN_PATH, JULIA_DEPOT_PATH) and use the
same Julia project, and each one loads ThermalTracking. A `CachingPool` ships `process_job`,
and any context it captures, to each worker once instead of once per job. Anything the
closure calls must therefore be defined on the workers: package functions are, but a named
function or a global defined only in the calling session is not.

`progress` is a ProgressMeter `Progress` that advances once per finished job.
"""
function _map_over_workers(process_job, jobs; n_workers::Int, progress)
	# Serial path: run every job in this process
	n_workers <= 1 && return progress_map(process_job, jobs; progress=progress)

	# Parallel path: start single-threaded workers on the same project
	worker_ids = addprocs(
		n_workers;
		exeflags=["--project=$(Base.active_project())", "--threads=1"],
	)
	try
		remotecall_eval(Main, worker_ids, :(using ThermalTracking))
		return progress_pmap(
			process_job,
			CachingPool(worker_ids),
			jobs;
			progress=progress,
		)
	finally
		rmprocs(worker_ids)
	end
end
