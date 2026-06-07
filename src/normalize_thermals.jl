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
function read_thermal_tracks(path::String)
	NCDataset(path, "r") do ds
		thermal_ixs = Array(ds["thermal_ix"])
		time_vals = Array(ds["time"])

		n_thermals = length(thermal_ixs)
		n_time = length(time_vals)

		x_all = Array(ds["x"])
		y_all = Array(ds["y"])
		z_all = Array(ds["z"])
		sr_all = Array(ds["sphere_radius"])
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
			"sphere_radius",
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
			(sz == (n_time, n_thermals) || sz == (n_thermals, n_time)) || continue
			push!(extra_names, name)
			extra_raw[name] = Array(ncvar)
			extra_attrs[name] = Dict{String, Any}(
				k => ncvar.attrib[k] for k in keys(ncvar.attrib) if !(k in skip_attrs)
			)
		end

		tracks = ThermalTrack[]
		extras = Dict(name => Vector{Vector{Float64}}() for name in extra_names)
		for ti in 1:n_thermals
			timesteps = ThermalTimeStep[]
			per_track_extras = Dict(name => Float64[] for name in extra_names)
			for di in 1:n_time
				sr = get_val(sr_all, di, ti)
				(ismissing(sr) || isnan(Float64(sr))) && continue
				push!(
					timesteps,
					ThermalTimeStep(
						Float64(get_val(x_all, di, ti)),
						Float64(get_val(y_all, di, ti)),
						Float64(get_val(z_all, di, ti)),
						Float64(sr),
						Float64(get_val(dt_all, di, ti)),
						time_vals[di],
					),
				)
				for name in extra_names
					val = get_val(extra_raw[name], di, ti)
					push!(per_track_extras[name], ismissing(val) ? NaN : Float64(val))
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

function interpolate_timestep!(
	result::Dict{String, Array{Float64, 3}},
	rams_path::String,
	vars::Vector{String},
	subset::NamedTuple,
	grid::SphericalGrid,
	ts::ThermalTimeStep,
)
	rho = grid.R .* ts.sphere_radius
	z_q = rho .* grid.cos_theta .+ ts.center_z
	x_q = rho .* grid.sin_theta_cos_phi .+ ts.center_x
	y_q = rho .* grid.sin_theta_sin_phi .+ ts.center_y

	NCDataset(rams_path, "r") do ds
		for var in vars
			data_sub = Float64.(ds[var][subset.x_ixs, subset.y_ixs, subset.z_ixs])

			itp = interpolate(
				(subset.x_sub, subset.y_sub, subset.z_sub),
				data_sub,
				Gridded(Linear()),
			)
			eitp = extrapolate(itp, NaN)

			result[var] .= eitp.(x_q, y_q, z_q)
		end
	end
end

# ──────────────────────────────────────────────────────────────────
# Output writer
# ──────────────────────────────────────────────────────────────────

"""
Write the normalized dataset to NetCDF.

Julia defVar dim tuples are the reverse of Python/ncdump/CDL order. Python dims
`(dt_thermal_seconds, thermal_ix, r_norm, theta, phi)` → defVar dims
`("phi", "theta", "r_norm", "thermal_ix", "dt_thermal_seconds")`.
"""
function _write_normalize_output(
	path::String,
	output::Dict{String, Array{Float64, 5}},
	grid::SphericalGrid,
	all_dt_seconds::Vector{Float64},
	tracks::Vector{ThermalTrack},
	time_seconds_2d::Matrix{Float64},
	composited_vars::Vector{String},
	extras_2d::Dict{String, Matrix{Float64}},
	extra_names::Vector{String},
	extra_attrs::Dict{String, Dict{String, Any}},
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
		defDim(ds, "phi", grid.n_phi)

		defVar(ds, "dt_thermal_seconds", all_dt_seconds, ("dt_thermal_seconds",))
		defVar(ds, "thermal_ix", thermal_ixs, ("thermal_ix",))
		defVar(ds, "r_norm", grid.r_vals, ("r_norm",))
		defVar(ds, "theta", grid.theta_vals, ("theta",))
		defVar(ds, "phi", grid.phi_vals, ("phi",))

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

		for var in composited_vars
			println("  Writing $var...")
			v = defVar(
				ds,
				var,
				Float64,
				("phi", "theta", "r_norm", "thermal_ix", "dt_thermal_seconds");
				fillvalue=NaN,
			)
			v[:, :, :, :, :] = permutedims(output[var], (5, 4, 3, 2, 1))
		end
	end
	println("Output written to $path")
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
	normalize_thermals(; thermal_ds_path, rams_output_dir, rams_grid_path, output_path,
						 composited_vars, n_r=20, n_theta=36, n_phi=72, r_norm_max=3.0)

Build a composite dataset in thermal-normalized spherical coordinates. For each
thermal at each timestep, linearly interpolates RAMS data from the Cartesian
grid onto an `(r_norm, theta, phi)` grid centered on the thermal where
`r_norm = 1` is the thermal edge.
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
	r_norm_max::Real=3.0,
)
	composited_vars = String.(collect(composited_vars))
	thermal_ds_path = String(thermal_ds_path)
	rams_output_dir = String(rams_output_dir)
	rams_grid_path = String(rams_grid_path)
	output_path = String(output_path)

	println("Reading thermal tracks from $thermal_ds_path")
	tracks, extras, extra_names, extra_attrs = read_thermal_tracks(thermal_ds_path)
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
	println("  Spherical grid: r=$(grid.n_r) × θ=$(grid.n_theta) × φ=$(grid.n_phi)")

	first_path = datetime_to_rams_path(rams_output_dir, tracks[1].timesteps[1].time)

	available_vars = NCDataset(first_path, "r") do ds
		Set(keys(ds))
	end
	missing_vars = [v for v in composited_vars if !(v in available_vars)]
	if !isempty(missing_vars)
		println(
			"  Skipping $(length(missing_vars)) variable(s) not present in " *
			"$first_path: " * join(missing_vars, ", "),
		)
		composited_vars = [v for v in composited_vars if v in available_vars]
	end
	if isempty(composited_vars)
		error("None of the requested composited_vars are present in $first_path")
	end

	verify_rams_dims(first_path, gc.x, gc.y, gc.z, composited_vars[1])

	all_dt_set = Set{Float64}()
	for track in tracks, ts in track.timesteps
		push!(all_dt_set, ts.dt_thermal_seconds)
	end
	all_dt_seconds = sort(collect(all_dt_set))
	dt_to_ix = Dict(dt => ix for (ix, dt) in enumerate(all_dt_seconds))

	n_dt = length(all_dt_seconds)
	n_thermals = length(tracks)
	println(
		"  Output: dt=$n_dt × thermals=$n_thermals × " *
		"r=$(grid.n_r) × θ=$(grid.n_theta) × φ=$(grid.n_phi) × vars=$(length(composited_vars))",
	)

	output = Dict(
		var => fill(NaN, n_dt, n_thermals, grid.n_r, grid.n_theta, grid.n_phi) for
		var in composited_vars
	)
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
		@local buf = Dict(
			var => Array{Float64}(undef, grid.n_r, grid.n_theta, grid.n_phi) for
			var in composited_vars
		)

		track = tracks[thermal_slot]
		subset = compute_spatial_subset(track, gc.x, gc.y, gc.z)

		for (ts_ix, ts) in enumerate(track.timesteps)
			rams_path = datetime_to_rams_path(rams_output_dir, ts.time)
			if !isfile(rams_path)
				@warn "RAMS file not found: $rams_path"
				continue
			end

			interpolate_timestep!(buf, rams_path, composited_vars, subset, grid, ts)

			dt_ix = dt_to_ix[ts.dt_thermal_seconds]
			for var in composited_vars
				output[var][dt_ix, thermal_slot, :, :, :] .= buf[var]
			end
			time_seconds_2d[dt_ix, thermal_slot] =
				Dates.value(Millisecond(ts.time - ref_time)) / 1000.0
			for name in extra_names
				extras_2d[name][dt_ix, thermal_slot] = extras[name][thermal_slot][ts_ix]
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
		composited_vars,
		extras_2d,
		extra_names,
		extra_attrs,
	)
	println("Done.")
	return nothing
end
