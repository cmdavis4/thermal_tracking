module ThermalTracking

using NCDatasets
using Dates
using Statistics
using ProgressMeter
using OhMyThreads: @tasks, @set, @local

include("io.jl")
include("thermodynamics.jl")
include("peak_finding.jl")
include("chain_w_maxima.jl")
include("smooth_trajectories.jl")
include("filter_trajectories.jl")
include("compute_thermal_shapes.jl")
include("normalize_thermals.jl")

export chain_w_maxima,
    smooth_trajectories,
    filter_thermal_trajectories,
    compute_trajectory_quality_metrics,
    trajectory_quality_masks,
    compute_thermal_shapes,
    normalize_thermals,
    write_rams_grid,
    composite,
    derived,
    derived_sum,
    derived_supersaturation,
    derived_supersaturated_flag

end
