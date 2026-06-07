using NCDatasets
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
# RAMS grid / path helpers
# ──────────────────────────────────────────────────────────────────

function read_rams_grid(path::String)
    NCDataset(path, "r") do ds
        (x=Float64.(ds["x"][:]), y=Float64.(ds["y"][:]), z=Float64.(ds["z"][:]))
    end
end

function datetime_to_rams_path(rams_dir::String, dt::DateTime)::String
    datestr = Dates.format(dt, dateformat"yyyy-mm-dd-HHMMSS")
    joinpath(rams_dir, "a-L-$(datestr)-g1.h5")
end

"""Parse the datetime encoded in a RAMS filename of the form `a-L-YYYY-mm-dd-HHMMSS-g1.h5`."""
function rams_path_to_datetime(path::String)::DateTime
    name = basename(path)
    # name = a-L-YYYY-mm-dd-HHMMSS-g1.h5
    m = match(r"^a-L-(\d{4})-(\d{2})-(\d{2})-(\d{6})-g1\.h5$", name)
    m === nothing && error("RAMS filename does not match expected pattern: $name")
    y, mo, d, hms = m.captures
    DateTime(parse(Int, y), parse(Int, mo), parse(Int, d),
             parse(Int, hms[1:2]), parse(Int, hms[3:4]), parse(Int, hms[5:6]))
end

"""Enumerate RAMS lite files `a-L-*-g1.h5` in `dir`, sorted by datetime."""
function list_rams_files(dir::String)::Vector{String}
    all_files = readdir(dir; join=true)
    rams = filter(p -> occursin(r"^a-L-.*-g1\.h5$", basename(p)), all_files)
    sort!(rams; by=rams_path_to_datetime)
    return rams
end

"""
Read one `__<name>` block from an open RAMS header file. Advances `io` past the block.
Returns `nothing` if the marker isn't found.
"""
function _read_header_block(io::IO, marker::String)::Union{Nothing,Vector{Float64}}
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

Coord values come from the `-head.txt` file paired with the first RAMS `.h5` file —
RAMS HDF5 outputs only carry phony dimensions, not coord arrays.
"""
function write_rams_grid(rams_output_dir::AbstractString, output_path::AbstractString)
    rams_output_dir = String(rams_output_dir)
    output_path = String(output_path)

    files = list_rams_files(rams_output_dir)
    isempty(files) && error("No RAMS files found in $rams_output_dir")

    header_path = replace(files[1], r"g\d+\.h5$" => "head.txt")
    isfile(header_path) || error("Expected header file not found: $header_path")

    x, y, z = open(header_path, "r") do io
        xv = _read_header_block(io, "__xtn01")
        yv = _read_header_block(io, "__ytn01")
        zv = _read_header_block(io, "__ztn01")
        (xv === nothing || yv === nothing || zv === nothing) &&
            error("Missing __xtn01/__ytn01/__ztn01 block in $header_path")
        (xv, yv, zv)
    end

    x .-= minimum(x)
    y .-= minimum(y)

    NCDataset(output_path, "c") do ds
        defDim(ds, "x", length(x))
        defDim(ds, "y", length(y))
        defDim(ds, "z", length(z))
        defVar(ds, "x", x, ("x",))
        defVar(ds, "y", y, ("y",))
        defVar(ds, "z", z, ("z",))
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
)
    max_sr = maximum(ts.sphere_radius for ts in track.timesteps)

    x_lo = minimum(ts.center_x for ts in track.timesteps) - max_sr * pad
    x_hi = maximum(ts.center_x for ts in track.timesteps) + max_sr * pad
    y_lo = minimum(ts.center_y for ts in track.timesteps) - max_sr * pad
    y_hi = maximum(ts.center_y for ts in track.timesteps) + max_sr * pad
    z_lo = minimum(ts.center_z for ts in track.timesteps) - max_sr * pad
    z_hi = maximum(ts.center_z for ts in track.timesteps) + max_sr * pad

    x_ixs = coord_range(x_coords, x_lo, x_hi)
    y_ixs = coord_range(y_coords, y_lo, y_hi)
    z_ixs = coord_range(z_coords, z_lo, z_hi)

    (
        x_ixs=x_ixs,
        y_ixs=y_ixs,
        z_ixs=z_ixs,
        x_sub=x_coords[x_ixs],
        y_sub=y_coords[y_ixs],
        z_sub=z_coords[z_ixs],
    )
end
