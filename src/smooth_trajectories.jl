using Polynomials: fit, derivative

# ──────────────────────────────────────────────────────────────────
# Per-cluster smoothing + trimming (mutates no external state)
# ──────────────────────────────────────────────────────────────────

struct _SmoothedTrajectory
	time::Vector{DateTime}
	time_ix::Vector{Int}
	t_minutes::Vector{Float64}
	t_seconds::Vector{Float64}
	dt_thermal_seconds::Vector{Float64}
	x::Vector{Float64}
	y::Vector{Float64}
	z::Vector{Float64}
	x_ix::Vector{Int}
	y_ix::Vector{Int}
	z_ix::Vector{Int}
	w_max::Vector{Float64}
	x_smoothed::Vector{Float64}
	y_smoothed::Vector{Float64}
	z_smoothed::Vector{Float64}
	x_propagation_speed::Vector{Float64}
	y_propagation_speed::Vector{Float64}
	z_propagation_speed::Vector{Float64}
end

function _idx_slice(t::_SmoothedTrajectory, idx)
	_SmoothedTrajectory(
		t.time[idx],
		t.time_ix[idx],
		t.t_minutes[idx],
		t.t_seconds[idx],
		t.dt_thermal_seconds[idx],
		t.x[idx],
		t.y[idx],
		t.z[idx],
		t.x_ix[idx],
		t.y_ix[idx],
		t.z_ix[idx],
		t.w_max[idx],
		t.x_smoothed[idx],
		t.y_smoothed[idx],
		t.z_smoothed[idx],
		t.x_propagation_speed[idx],
		t.y_propagation_speed[idx],
		t.z_propagation_speed[idx],
	)
end

function _smooth_cluster(
	cluster_peaks::NamedTuple;
	propagation_poly_degree::Int,
	min_point_vertical_propagation_speed::Float64,
	min_lifetime_vertical_propagation_speed::Float64,
	min_peaks::Int,
	apply_lifetime_threshold_after_trimming::Bool,
)::Union{Nothing, _SmoothedTrajectory}
	dt = cluster_peaks.dt_thermal_seconds

	# Fit polynomials per axis and evaluate + derivative
	poly_x = fit(dt, cluster_peaks.x, propagation_poly_degree)
	poly_y = fit(dt, cluster_peaks.y, propagation_poly_degree)
	poly_z = fit(dt, cluster_peaks.z, propagation_poly_degree)

	x_s = poly_x.(dt)
	y_s = poly_y.(dt)
	z_s = poly_z.(dt)

	dpoly_x = derivative(poly_x)
	dpoly_y = derivative(poly_y)
	dpoly_z = derivative(poly_z)

	x_p = dpoly_x.(dt)
	y_p = dpoly_y.(dt)
	z_p = dpoly_z.(dt)

	traj = _SmoothedTrajectory(
		cluster_peaks.time,
		cluster_peaks.time_ix,
		cluster_peaks.t_minutes,
		cluster_peaks.t_seconds,
		cluster_peaks.dt_thermal_seconds,
		cluster_peaks.x,
		cluster_peaks.y,
		cluster_peaks.z,
		cluster_peaks.x_ix,
		cluster_peaks.y_ix,
		cluster_peaks.z_ix,
		cluster_peaks.w_max,
		x_s,
		y_s,
		z_s,
		x_p,
		y_p,
		z_p,
	)

	# Edge-only filter: trim leading and trailing points where the vertical
	# propagation speed falls below the threshold. Interior points are
	# preserved so the trajectory stays contiguous in time.
	above = traj.z_propagation_speed .> min_point_vertical_propagation_speed
	any(above) || return nothing
	traj = _idx_slice(traj, findfirst(above):findlast(above))

	# Edge trim: peel from whichever end has the smaller w_max until mean vertical
	# propagation speed ≥ threshold.
	while length(traj.time) > 0 &&
		mean(traj.z_propagation_speed) < min_lifetime_vertical_propagation_speed
		if traj.w_max[1] > traj.w_max[end]
			traj = _idx_slice(traj, 1:(length(traj.time)-1))
		else
			traj = _idx_slice(traj, 2:length(traj.time))
		end
	end
	isempty(traj.time) && return nothing

	if apply_lifetime_threshold_after_trimming
		# Count-based, matching the pre-trim filter. Using the dt span here would
		# let a cluster pass with very few surviving points but a wide endpoint
		# gap (the per-point z_propagation_speed filter can drop interior points
		# when the polyfit derivative oscillates).
		if length(traj.time) < min_peaks
			return nothing
		end
	end

	return traj
end

# ──────────────────────────────────────────────────────────────────
# Flat-table reader
# ──────────────────────────────────────────────────────────────────

function _read_raw_wmax_clusters(path::String)
	NCDataset(path, "r") do ds
		cluster_id = Array(ds["cluster_id"])
		seq_ix = Array(ds["seq_ix"])
		time_ix = Array(ds["time_ix"])
		t_minutes = Array(ds["t_minutes"])
		x = Array(ds["x"])
		y = Array(ds["y"])
		z = Array(ds["z"])
		x_ix = Array(ds["x_ix"])
		y_ix = Array(ds["y_ix"])
		z_ix = Array(ds["z_ix"])
		w = Array(ds["w"])
		times = Array(ds["time"])   # NCDatasets decodes CF time → DateTime
		(;
			cluster_id,
			seq_ix,
			time_ix,
			t_minutes,
			x,
			y,
			z,
			x_ix,
			y_ix,
			z_ix,
			w,
			time=times,
		)
	end
end

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

"""
	smooth_trajectories(; raw_wmax_clusters_path, output_path,
						  propagation_poly_degree=3,
						  min_point_vertical_propagation_speed=0.1,
						  min_lifetime_vertical_propagation_speed=1.0,
						  min_thermal_lifetime_seconds,
						  apply_lifetime_threshold_after_trimming=true)

Polynomial-smooth each cluster's (x, y, z) trajectory, compute propagation speeds
from analytic derivatives, drop low-vertical-speed points, trim leading/trailing
edges, and discard clusters that are too short. Writes a (thermal_ix, time)
dataset to `output_path`.
"""
function smooth_trajectories(;
	raw_wmax_clusters_path::AbstractString,
	output_path::AbstractString,
	propagation_poly_degree::Int=3,
	min_point_vertical_propagation_speed::Real=0.1,
	min_lifetime_vertical_propagation_speed::Real=1.0,
	min_thermal_lifetime_seconds::Real,
	apply_lifetime_threshold_after_trimming::Bool=true,
)
	println("Reading raw clusters from $raw_wmax_clusters_path")
	flat = _read_raw_wmax_clusters(String(raw_wmax_clusters_path))
	n_peaks = length(flat.cluster_id)
	println("  $n_peaks total peaks")

	# Infer timestep from the set of unique times
	unique_times = sort!(unique(flat.time))
	length(unique_times) < 2 && error("Need at least 2 distinct timesteps to infer dt")
	timestep_seconds =
		Float64(Dates.value(Millisecond(unique_times[2] - unique_times[1]))) / 1000.0
	println("  Timestep: $(timestep_seconds) s, $(length(unique_times)) unique times")

	# Group peaks by cluster_id
	cluster_ids = sort!(unique(flat.cluster_id))
	clusters = Vector{NamedTuple}(undef, length(cluster_ids))
	for (ci, cid) in enumerate(cluster_ids)
		idx = findall(==(cid), flat.cluster_id)
		# Order within cluster by seq_ix
		order = sortperm(flat.seq_ix[idx])
		idx = idx[order]
		times = flat.time[idx]
		dt_seconds =
			Float64[Dates.value(Millisecond(t - times[1])) / 1000.0 for t in times]
		clusters[ci] = (
			cluster_id=cid,
			time=times,
			time_ix=Int.(flat.time_ix[idx]),
			t_minutes=Float64.(flat.t_minutes[idx]),
			t_seconds=Float64.(flat.t_minutes[idx]) .* 60.0,
			dt_thermal_seconds=dt_seconds,
			x=Float64.(flat.x[idx]),
			y=Float64.(flat.y[idx]),
			z=Float64.(flat.z[idx]),
			x_ix=Int.(flat.x_ix[idx]),
			y_ix=Int.(flat.y_ix[idx]),
			z_ix=Int.(flat.z_ix[idx]),
			w_max=Float64.(flat.w[idx]),
		)
	end

	# Pre-trim lifetime filter (matches Python's `min_cluster_length` step).
	# Bind to a new name rather than rebinding `clusters` — OhMyThreads' `@tasks`
	# box-capture check rejects closures over a rebound outer local.
	min_peaks = ceil(Int, min_thermal_lifetime_seconds / timestep_seconds)
	long_clusters = filter(c -> length(c.time) >= min_peaks, clusters)
	println(
		"  $(length(long_clusters)) clusters pass pre-trim lifetime filter (≥$min_peaks peaks)",
	)

	# Parallel smoothing
	println("Smoothing trajectories on $(Threads.nthreads()) threads…")
	smoothed = Vector{Union{Nothing, _SmoothedTrajectory}}(undef, length(long_clusters))
	@tasks for i in eachindex(long_clusters)
		smoothed[i] = _smooth_cluster(
			long_clusters[i];
			propagation_poly_degree=propagation_poly_degree,
			min_point_vertical_propagation_speed=Float64(
				min_point_vertical_propagation_speed,
			),
			min_lifetime_vertical_propagation_speed=Float64(
				min_lifetime_vertical_propagation_speed,
			),
			min_peaks=min_peaks,
			apply_lifetime_threshold_after_trimming=apply_lifetime_threshold_after_trimming,
		)
	end

	surviving = _SmoothedTrajectory[t for t in smoothed if t !== nothing]
	println("  $(length(surviving)) trajectories survive trimming")

	_write_thermal_trajectories(String(output_path), surviving)
	println("Wrote $(output_path)")
	return nothing
end

# ──────────────────────────────────────────────────────────────────
# Write (thermal_ix, time) dataset — time = union across all trajectories
# ──────────────────────────────────────────────────────────────────

function _write_thermal_trajectories(
	path::String,
	trajectories::Vector{_SmoothedTrajectory},
)
	n_thermals = length(trajectories)

	# Union of all times across trajectories
	all_times_set = Set{DateTime}()
	for t in trajectories, tt in t.time
		push!(all_times_set, tt)
	end
	all_times = sort!(collect(all_times_set))
	n_time = length(all_times)
	time_to_ix = Dict(t => i for (i, t) in enumerate(all_times))

	# Pre-allocate 2-D (time, thermal_ix) arrays — Julia defVar order is reversed
	# relative to Python dims (thermal_ix, time), so we allocate in (time, thermal_ix)
	# and hand that directly to defVar.
	Tf = (T) -> fill(T(NaN), n_time, n_thermals)
	Ti = () -> fill(-1, n_time, n_thermals)   # sentinel for integer NaN
	x = Tf(Float64);
	y = Tf(Float64);
	z = Tf(Float64)
	x_ix = Ti();
	y_ix = Ti();
	z_ix = Ti()
	w_max = Tf(Float64)
	x_smoothed = Tf(Float64);
	y_smoothed = Tf(Float64);
	z_smoothed = Tf(Float64)
	x_prop = Tf(Float64);
	y_prop = Tf(Float64);
	z_prop = Tf(Float64)
	time_ix_2d = Ti()
	t_minutes_2d = Tf(Float64)
	t_seconds_2d = Tf(Float64)
	dt_thermal_2d = Tf(Float64)

	for (ti, traj) in enumerate(trajectories)
		for (j, t) in enumerate(traj.time)
			i = time_to_ix[t]
			x[i, ti] = traj.x[j];
			y[i, ti] = traj.y[j];
			z[i, ti] = traj.z[j]
			x_ix[i, ti] = traj.x_ix[j];
			y_ix[i, ti] = traj.y_ix[j];
			z_ix[i, ti] = traj.z_ix[j]
			w_max[i, ti] = traj.w_max[j]
			x_smoothed[i, ti] = traj.x_smoothed[j]
			y_smoothed[i, ti] = traj.y_smoothed[j]
			z_smoothed[i, ti] = traj.z_smoothed[j]
			x_prop[i, ti] = traj.x_propagation_speed[j]
			y_prop[i, ti] = traj.y_propagation_speed[j]
			z_prop[i, ti] = traj.z_propagation_speed[j]
			time_ix_2d[i, ti] = traj.time_ix[j]
			t_minutes_2d[i, ti] = traj.t_minutes[j]
			t_seconds_2d[i, ti] = traj.t_seconds[j]
			dt_thermal_2d[i, ti] = traj.dt_thermal_seconds[j]
		end
	end

	ref_time = DateTime(2000, 1, 1)
	time_seconds =
		Float64[Dates.value(Millisecond(t - ref_time)) / 1000.0 for t in all_times]
	thermal_ix = collect(0:(n_thermals-1))

	isfile(path) && rm(path)
	NCDataset(path, "c") do ds
		# Python dims: (thermal_ix, time) → Julia defVar dims: ("time", "thermal_ix")
		defDim(ds, "thermal_ix", n_thermals)
		defDim(ds, "time", n_time)

		defVar(ds, "thermal_ix", thermal_ix, ("thermal_ix",))
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
		defVar(ds, "x", x, julia_dims; fillvalue=NaN)
		defVar(ds, "y", y, julia_dims; fillvalue=NaN)
		defVar(ds, "z", z, julia_dims; fillvalue=NaN)
		defVar(ds, "x_ix", x_ix, julia_dims; fillvalue=-1)
		defVar(ds, "y_ix", y_ix, julia_dims; fillvalue=-1)
		defVar(ds, "z_ix", z_ix, julia_dims; fillvalue=-1)
		defVar(ds, "w_max", w_max, julia_dims; fillvalue=NaN)
		defVar(ds, "x_smoothed", x_smoothed, julia_dims; fillvalue=NaN)
		defVar(ds, "y_smoothed", y_smoothed, julia_dims; fillvalue=NaN)
		defVar(ds, "z_smoothed", z_smoothed, julia_dims; fillvalue=NaN)
		defVar(ds, "x_propagation_speed", x_prop, julia_dims; fillvalue=NaN)
		defVar(ds, "y_propagation_speed", y_prop, julia_dims; fillvalue=NaN)
		defVar(ds, "z_propagation_speed", z_prop, julia_dims; fillvalue=NaN)
		defVar(ds, "time_ix", time_ix_2d, julia_dims; fillvalue=-1)
		defVar(ds, "t_minutes", t_minutes_2d, julia_dims; fillvalue=NaN)
		defVar(ds, "t_seconds", t_seconds_2d, julia_dims; fillvalue=NaN)
		defVar(ds, "dt_thermal_seconds", dt_thermal_2d, julia_dims; fillvalue=NaN)
	end
end
