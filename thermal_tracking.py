from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count

from tqdm import tqdm
import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates
from skimage.feature import peak_local_max

from carlee_tools import spacing

# ---------------------------------------------------------------------------
# Internal dataclasses (not part of public API)
# ---------------------------------------------------------------------------


@dataclass
class _WMaximum:
    time_ix: int
    time: np.datetime64
    t_minutes: float
    coords: tuple[float, float, float]  # (z, y, x)
    coord_ixs: tuple[int, int, int]  # (z, y, x)
    w: float
    u: float
    v: float
    ix: int = field(default_factory=lambda: next(_WMaximum._counter), init=False)
    _counter: count = field(default=None, init=False, repr=False)

    def __eq__(self, other):
        if not isinstance(other, _WMaximum):
            return False
        return self.ix == other.ix

    def __hash__(self):
        return hash(self.ix)


# Class-level counter (reset per tracking run via _reset_counters)
_WMaximum._counter = count()


@dataclass
class _Match:
    maximum1: _WMaximum
    maximum2: _WMaximum
    distance: float


@dataclass
class _WMaxCluster:
    w_maxima: list[_WMaximum]
    matches: list[_Match]


# ---------------------------------------------------------------------------
# Helper geometry functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 1: Find updraft maxima at a single time step
# ---------------------------------------------------------------------------


def _find_w_maxima_at_time(
    time_ds,
    current_time_ix,
    current_time,
    zyx_grid_spacing,
    min_distance=500,
    threshold_abs=0.8,
):
    maxima_dim_order = ["z", "y", "x"]
    current_time_slice = time_ds.isel(time=-1)
    current_w = current_time_slice["WC"].transpose(*maxima_dim_order).values
    current_u = current_time_slice["UC"].transpose(*maxima_dim_order).values
    current_v = current_time_slice["VC"].transpose(*maxima_dim_order).values

    current_w_maxima_coords_ixs = peak_local_max(
        image=current_w,
        min_distance=int(min_distance / zyx_grid_spacing[1]),
        threshold_abs=threshold_abs,
    )

    maxima_ixs = [
        current_w_maxima_coords_ixs[:, i]
        for i in range(current_w_maxima_coords_ixs.shape[1])
    ]
    current_w_maxima_speeds = current_w[*maxima_ixs]
    current_u_at_maxima = current_u[*maxima_ixs]
    current_v_at_maxima = current_v[*maxima_ixs]
    current_w_maxima_coords = current_w_maxima_coords_ixs * zyx_grid_spacing

    return [
        _WMaximum(
            time_ix=current_time_ix,
            time=current_time,
            t_minutes=time_ds["t_minutes"].values,
            coords=current_w_maxima_coords[i],
            coord_ixs=current_w_maxima_coords_ixs[i],
            w=current_w_maxima_speeds[i],
            u=current_u_at_maxima[i],
            v=current_v_at_maxima[i],
        )
        for i in range(len(current_w_maxima_speeds))
    ]


# ---------------------------------------------------------------------------
# Step 2: Chain maxima across time into clusters
# ---------------------------------------------------------------------------


def chain_w_maxima(storm_ds, min_distance=500, threshold_abs=0.8):
    w_maxima_by_time = []
    w_clusters = []

    z_grid_spacing = spacing(storm_ds["z"]).values.item()
    xy_grid_spacing = spacing(storm_ds["x"]).values.item()
    zyx_grid_spacing = np.array([z_grid_spacing, xy_grid_spacing, xy_grid_spacing])
    timestep = (
        spacing(storm_ds["time"]).astype("timedelta64[s]").astype(int).values.item()
    )

    for current_time_ix, (current_time, current_time_ds) in tqdm(
        enumerate(storm_ds.groupby("time")),
        total=len(storm_ds["time"]),
    ):
        current_w_maxima = _find_w_maxima_at_time(
            time_ds=current_time_ds,
            current_time_ix=current_time_ix,
            current_time=current_time,
            zyx_grid_spacing=zyx_grid_spacing,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
        )
        w_maxima_by_time.append(current_w_maxima)

        if current_time_ix == 0:
            w_clusters = [_WMaxCluster([wm], matches=[]) for wm in current_w_maxima]
            continue

        last_time_maxima = w_maxima_by_time[-2]
        current_match_candidates = defaultdict(list)
        last_match_candidates = defaultdict(list)

        for this_current_max in current_w_maxima:
            for this_last_max in last_time_maxima:
                pair_is_z_candidate = (
                    (this_current_max.coords[0] >= this_last_max.coords[0])
                    & (
                        (
                            this_current_max.coords[0]
                            - (this_current_max.w * timestep * 1.5)
                        )
                        <= this_last_max.coords[0]
                    )
                    & (
                        (this_last_max.coords[0] + (this_last_max.w * timestep * 1.5))
                        >= this_current_max.coords[0]
                    )
                )
                if not pair_is_z_candidate:
                    continue

                # pair_is_xy_candidate = True
                # for coord_ix, wind_attr in [(1, "v"), (2, "u")]:
                #     current_wind = getattr(this_current_max, wind_attr)
                #     last_wind = getattr(this_last_max, wind_attr)
                #     pair_is_xy_candidate = pair_is_xy_candidate & (
                #         (
                #             (
                #                 this_current_max.coords[coord_ix]
                #                 - this_last_max.coords[coord_ix]
                #             )
                #             * (current_wind + last_wind)
                #             >= 0
                #         )
                #         & (
                #             (
                #                 this_current_max.coords[coord_ix]
                #                 - (current_wind * timestep * 1.5)
                #             )
                #             <= this_last_max.coords[coord_ix]
                #         )
                #         & (
                #             (
                #                 this_last_max.coords[coord_ix]
                #                 + (last_wind * timestep * 1.5)
                #             )
                #             >= this_current_max.coords[coord_ix]
                #         )
                #     )
                # if not pair_is_xy_candidate:
                #     continue

                last_match_candidates[this_last_max.ix].append(
                    _Match(
                        maximum1=this_last_max,
                        maximum2=this_current_max,
                        distance=np.linalg.norm(
                            this_last_max.coords
                            + np.array([0.5 * timestep * this_last_max.w, 0, 0])
                            - this_current_max.coords
                        ),
                    )
                )
                current_match_candidates[this_current_max.ix].append(
                    _Match(
                        maximum1=this_current_max,
                        maximum2=this_last_max,
                        distance=np.linalg.norm(
                            this_current_max.coords
                            - np.array([0.5 * timestep * this_current_max.w, 0, 0])
                            - this_last_max.coords
                        ),
                    )
                )

        def best_match(candidates_dict):
            return {
                ix: min(cands, key=lambda m: m.distance)
                for ix, cands in candidates_dict.items()
            }

        current_best = best_match(current_match_candidates)
        last_best = best_match(last_match_candidates)

        matches = []
        for last_max_ix, last_bm in last_best.items():
            if current_best[last_bm.maximum2.ix].maximum2 is last_bm.maximum1:
                matches.append(last_bm)

        for match in matches:
            parent = [c for c in w_clusters if match.maximum1 in c.w_maxima]
            if len(parent) != 1:
                raise ValueError("Expected exactly one cluster for previous maximum")
            parent[0].w_maxima.append(match.maximum2)
            parent[0].matches.append(match)

        matched_current = {m.maximum2 for m in matches}
        for wm in current_w_maxima:
            if wm not in matched_current:
                w_clusters.append(_WMaxCluster(w_maxima=[wm], matches=[]))

    return w_clusters


# ---------------------------------------------------------------------------
# Step 3: Compute smoothed trajectories and propagation speeds for a cluster
# ---------------------------------------------------------------------------


def wmax_cluster_to_dataset(cluster, propagation_poly_degree=3):
    wms = cluster.w_maxima
    ds = xr.Dataset(
        {
            "x": ("time", [wm.coords[2] for wm in wms]),
            "y": ("time", [wm.coords[1] for wm in wms]),
            "z": ("time", [wm.coords[0] for wm in wms]),
            "x_ix": ("time", [wm.coord_ixs[2] for wm in wms]),
            "y_ix": ("time", [wm.coord_ixs[1] for wm in wms]),
            "z_ix": ("time", [wm.coord_ixs[0] for wm in wms]),
            "w_max": ("time", [wm.w for wm in wms]),
        },
        coords={"time": [wm.time for wm in wms]},
    )
    ds = ds.assign_coords({
        "time_ix": ("time", [wm.time_ix for wm in wms]),
        "t_minutes": ("time", [wm.t_minutes[0] for wm in wms]),
        "t_seconds": ("time", [wm.t_minutes[0] * 60 for wm in wms]),
        "dt_thermal_seconds": (
            "time",
            [(wm.time - wms[0].time) / np.timedelta64(1, "s") for wm in wms],
        ),
    })

    # Fit polynomials to position and compute smoothed + propagation speed
    for dim in ["x", "y", "z"]:
        coeffs = ds[dim].polyfit(dim="dt_thermal_seconds", deg=propagation_poly_degree)[
            "polyfit_coefficients"
        ]
        smoothed = xr.zeros_like(ds[dim])
        propagation = xr.zeros_like(ds[dim])
        for degree in range(propagation_poly_degree + 1):
            c = coeffs.sel(degree=degree)
            smoothed = smoothed + c * ds["dt_thermal_seconds"] ** degree
            if degree > 0:
                propagation = propagation + degree * c * ds["dt_thermal_seconds"] ** (
                    degree - 1
                )
        ds[f"{dim}_smoothed"] = smoothed
        ds[f"{dim}_propagation_speed"] = propagation

    # Filter: drop points with vertical propagation speed < 0.1 m/s
    ds = ds.sel(time=ds["z_propagation_speed"] > 0.1)

    # Trim edges until mean vertical propagation speed >= 1 m/s
    while len(ds["time"]) > 0 and ds["z_propagation_speed"].mean().values < 1:
        if ds["w_max"].isel(time=0) > ds["w_max"].isel(time=-1):
            ds = ds.isel(time=slice(None, -1))
        else:
            ds = ds.isel(time=slice(1, None))

    return ds


# ---------------------------------------------------------------------------
# Step 4: Determine thermal shape (sphere_radius) at each time
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n):
    """Generate `n` approximately uniformly distributed unit vectors on a sphere."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = 2 * np.pi * i / golden_ratio
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    return np.column_stack([
        np.cos(phi),  # z
        np.sin(phi) * np.sin(theta),  # y
        np.sin(phi) * np.cos(theta),  # x
    ])


import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def _find_zero_radius(data, center, zyx_coords, max_radius=3000):
    """field: 3D array on (z, y, x); z can be non-uniform."""
    # Unpack thermal center and coordinate arrays
    center_z, center_y, center_x = center
    z_coords, y_coords, x_coords = zyx_coords
    # Get grid spacings, allowing stretched z grid
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    dz = np.diff(z_coords)
    # Get cell-center z coordinates
    # FIXME: Need to handle grid stagger
    zc = 0.5 * (z_coords[:-1] + z_coords[1:])

    # Subset to a bounding box around the center within max_radius
    if max_radius:
        x_mask = np.abs(x_coords - center_x) <= max_radius
        y_mask = np.abs(y_coords - center_y) <= max_radius
        z_mask = np.abs(zc - center_z) <= max_radius
        x_coords = x_coords[x_mask]
        y_coords = y_coords[y_mask]
        zc = zc[z_mask]
        dz = dz[z_mask]
        data_subsetted = data[np.ix_(z_mask, y_mask, x_mask)]
    else:
        data_subsetted = data

    # Create meshgrid of cell centers (z, y, x order to match data layout)
    z_grid, y_grid, x_grid = np.meshgrid(zc, y_coords, x_coords, indexing="ij")
    # Calculate radial distances from the thermal center on the grid we're considering
    radial_distances = np.sqrt(
        (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 + (z_grid - center_z) ** 2
    )

    # Calculate cell volumes, accounting for stretched grid
    DZ = dz[:, np.newaxis, np.newaxis]
    vol = np.resize(dx * dy * DZ, data_subsetted.shape)

    # Volume-weight the data
    data_volume_weighted = (data_subsetted * vol).ravel()
    # Find the order that sorts the radial distances, so we can get the cumulative
    # value of the volume-weighted data as we increase the radius
    radial_distances_flattened = radial_distances.ravel()
    radial_distances_order = np.argsort(radial_distances_flattened)
    radial_distances_sorted = radial_distances_flattened[radial_distances_order]
    cumu_volume_weighted_data = np.cumsum(data_volume_weighted[radial_distances_order])

    # Find the index of the radius where the sign flips
    sign_changes = np.where(np.diff(np.sign(cumu_volume_weighted_data)))[0]
    if len(sign_changes) == 0:
        return None
    zero_crossing_lb_ix = sign_changes[0]
    # The sign flips betwen zero_crossing_lb_ix and zero_crossing_lb_ix+1;
    # interpolate to the exact zero crossing
    return radial_distances_sorted[zero_crossing_lb_ix] - cumu_volume_weighted_data[
        zero_crossing_lb_ix
    ] * (
        radial_distances_sorted[zero_crossing_lb_ix + 1]
        - radial_distances_sorted[zero_crossing_lb_ix]
    ) / (
        cumu_volume_weighted_data[zero_crossing_lb_ix + 1]
        - cumu_volume_weighted_data[zero_crossing_lb_ix]
    )


# def _find_zero_radius(
#     data,
#     center,
#     max_sphere_radius,
#     grid_spacing,
#     bin_step=50,
#     n_shell_points=500,
# ):
#     if np.isscalar(grid_spacing):
#         dz = dy = dx = grid_spacing
#     else:
#         dz, dy, dx = grid_spacing

#     cz, cy, cx = center
#     unit_dirs = _fibonacci_sphere(n_shell_points)

#     sphere_radius_bins = np.arange(bin_step, max_sphere_radius + 1, bin_step)
#     last_mean_w = 0.0

#     for radius_bin_ix, curr_r in enumerate(sphere_radius_bins):
#         # Physical offsets from center for each sample point on this shell
#         shell_coords = unit_dirs * curr_r
#         # Convert to fractional grid indices
#         sample_ixs = np.column_stack([
#             cz + shell_coords[:, 0] / dz,
#             cy + shell_coords[:, 1] / dy,
#             cx + shell_coords[:, 2] / dx,
#         ]).T  # shape (3, n_shell_points)

#         # Exclude points outside the grid
#         nz, ny, nx = data.shape
#         in_bounds = (
#             (sample_ixs[0] >= 0) & (sample_ixs[0] <= nz - 1)
#             & (sample_ixs[1] >= 0) & (sample_ixs[1] <= ny - 1)
#             & (sample_ixs[2] >= 0) & (sample_ixs[2] <= nx - 1)
#         )
#         if not np.any(in_bounds):
#             break

#         values = map_coordinates(data, sample_ixs[:, in_bounds], order=1)
#         mean_w = values.mean()

#         if radius_bin_ix > 0 and np.sign(last_mean_w) != np.sign(mean_w):
#             prev_r = sphere_radius_bins[radius_bin_ix - 1]
#             frac = -(last_mean_w / (mean_w - last_mean_w))
#             return prev_r + frac * (curr_r - prev_r)

#         last_mean_w = mean_w

#     return None


def _interp_index(coord_array, value):
    """Find the (fractional) index of `value` within a 1-D coordinate array."""
    return np.interp(value, coord_array, np.arange(len(coord_array)))


def compute_thermal_shapes(storm_ds, thermal_ds):
    sphere_radius = np.full(len(thermal_ds["time"]), np.nan)

    for t_ix, t in enumerate(thermal_ds["time"].values):
        this_time_thermal_ds = thermal_ds.sel(time=t)
        this_time_storm_ds = storm_ds.sel(time=t)

        center = tuple(
            float(this_time_thermal_ds[dim].values)
            # float(this_time_thermal_ds[f"{dim}_smoothed"].values)
            for dim in ["z", "y", "x"]
        )
        w_field = (
            this_time_storm_ds["WC"].transpose("z", "y", "x")
            - this_time_thermal_ds["z_propagation_speed"].values
        )
        zero_r = _find_zero_radius(
            data=w_field.values,
            center=center,
            zyx_coords=(
                this_time_storm_ds["z"].values,
                this_time_storm_ds["y"].values,
                this_time_storm_ds["x"].values,
            ),
        )
        if zero_r is not None:
            sphere_radius[t_ix] = zero_r

    thermal_ds["sphere_radius"] = ("time", sphere_radius)
    return thermal_ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def track_thermals(
    storm_ds,
    min_cluster_length=18,
    propagation_poly_degree=3,
    min_distance=500,  # m
    threshold_abs=0.8,  # m/s
):
    """
    Track thermals in simulation output.

    Parameters
    ----------
    storm_ds : xr.Dataset
        Gridded simulation output with dimensions (time, y, x, z).
        Must contain a ``WC`` variable (vertical velocity) and
        coordinate variables ``x``, ``y``, ``z``, ``time``, ``t_minutes``.
    min_cluster_length : int, optional
        Minimum number of time steps for a cluster to be kept (default 18).
    propagation_poly_degree : int, optional
        Degree of polynomial fit for smoothed trajectories (default 3).
    min_distance : float, optional
        Minimum physical distance between updraft maxima (default 500).
    threshold_abs : float, optional
        Minimum W threshold for peak detection (default 0.8).

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``(thermal_ix, time)`` containing centroid
        positions, propagation speeds, and thermal shape variables.
    """
    # Reset the internal counter so IDs start at 0 each run
    _WMaximum._counter = count()

    # Step 1+2: find maxima and chain them across time
    wmax_clusters = chain_w_maxima(
        storm_ds,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
    )

    # Filter short-lived clusters
    wmax_clusters = [
        wc for wc in wmax_clusters if len(wc.w_maxima) >= min_cluster_length
    ]

    # Step 3: convert each cluster to a trajectory dataset
    thermal_datasets = []
    for cluster in wmax_clusters:
        ds = wmax_cluster_to_dataset(
            cluster, propagation_poly_degree=propagation_poly_degree
        )
        if len(ds["time"]) > 0:
            thermal_datasets.append(ds)

    if not thermal_datasets:
        return xr.Dataset()

    # Step 4: compute shapes for each thermal
    z_grid_spacing = spacing(storm_ds["z"]).values.item()
    xy_grid_spacing = spacing(storm_ds["x"]).values.item()
    grid_spacing_zyx = np.array([z_grid_spacing, xy_grid_spacing, xy_grid_spacing])
    for i, tds in enumerate(tqdm(thermal_datasets, desc="Computing thermal shapes")):
        thermal_datasets[i] = compute_thermal_shapes(
            storm_ds, tds, grid_spacing_zyx=grid_spacing_zyx
        )

    # Step 5: flag thermals that partly extend outside the RAMS domain
    domain_bounds = {
        d: (float(storm_ds[d].min()), float(storm_ds[d].max())) for d in ("x", "y", "z")
    }
    for i, tds in enumerate(thermal_datasets):
        sr = tds["sphere_radius"]
        out_of_domain = xr.zeros_like(sr, dtype=bool)
        for d in ("x", "y", "z"):
            lo, hi = domain_bounds[d]
            out_of_domain = out_of_domain | (tds[d] - sr < lo) | (tds[d] + sr > hi)
        thermal_datasets[i]["out_of_domain"] = out_of_domain

    # Combine into a single dataset with thermal_ix dimension
    # Align all thermals to the union of all time steps (fills with NaN)
    combined = xr.concat(thermal_datasets, dim="thermal_ix")
    combined["thermal_ix"] = np.arange(len(thermal_datasets))

    return combined


def mask_storm_by_thermal(storm_ds, thermal_ds):
    """
    Mask gridded simulation output to points inside a thermal's sphere,
    across all time steps where the thermal exists.

    Parameters
    ----------
    storm_ds : xr.Dataset
        Gridded simulation output with dimensions (time, y, x, z).
    thermal_ds : xr.Dataset
        Single-thermal tracking dataset with dimension (time).
        Times where the thermal doesn't exist should already be dropped.
        Must contain x, y, z (centroid) and sphere_radius.

    Returns
    -------
    xr.Dataset
        storm_ds subset to thermal_ds times, with points outside the
        thermal's ellipsoid masked to NaN at each time step.
    """
    # Subset storm_ds to the times present in thermal_ds
    storm_subset = storm_ds.sel(time=thermal_ds.time)

    sr = thermal_ds["sphere_radius"]

    # Subset to the overall region the thermal moves through
    # Crop to the max bounding box across all times, then mask
    pad = 1.2  # small safety margin
    x_slice = slice(
        float((thermal_ds["x"] - sr * pad).min()),
        float((thermal_ds["x"] + sr * pad).max()),
    )
    y_slice = slice(
        float((thermal_ds["y"] - sr * pad).min()),
        float((thermal_ds["y"] + sr * pad).max()),
    )
    z_slice = slice(
        float((thermal_ds["z"] - sr * pad).min()),
        float((thermal_ds["z"] + sr * pad).max()),
    )
    storm_subset = storm_subset.sel({"x": x_slice, "y": y_slice, "z": z_slice})

    # Spherical distance
    dist_sq = (
        ((storm_subset.x - thermal_ds["x"]) / sr) ** 2
        + ((storm_subset.y - thermal_ds["y"]) / sr) ** 2
        + ((storm_subset.z - thermal_ds["z"]) / sr) ** 2
    )

    mask = dist_sq <= 1.0
    xyz_var_suffixes = ["", "_ix", "_smoothed", "_propagation_speed"]
    for var in ["x", "y", "z"]:
        thermal_ds = thermal_ds.rename(
            {var + suffix: "thermal_" + var + suffix for suffix in xyz_var_suffixes}
        )
    return thermal_ds.merge(storm_subset.where(mask))


def interpolate_to_normalized_coords(this_time_rams_ds, xyz_grids, interpolated_vars):
    from scipy.interpolate import RegularGridInterpolator

    # Use dim order from first variable; transpose others to match
    dim_order = this_time_rams_ds[interpolated_vars[0]].dims
    coords = [this_time_rams_ds[d].values for d in dim_order]
    query_points = np.stack([xyz_grids[d].ravel() for d in dim_order], axis=-1)

    # Stack all variables along a trailing axis for a single interpolation call
    stacked = np.stack(
        [this_time_rams_ds[v].transpose(*dim_order).values for v in interpolated_vars],
        axis=-1,
    )
    interp = RegularGridInterpolator(
        coords, stacked, bounds_error=False, fill_value=np.nan
    )
    result = interp(query_points)  # (n_points, n_vars)
    return {v: result[:, i] for i, v in enumerate(interpolated_vars)}


def normalize_thermals(
    thermal_ds,
    rams_ds,
    composited_vars=("WC",),
    n_r=20,
    n_theta=36,
    n_phi=72,
    r_norm_max=3.0,
):
    """
    Build a composite dataset in thermal-normalized spherical coordinates.

    For each thermal at each time step, interpolates storm data from the
    Cartesian subcube onto a (r_norm, theta, phi) grid centered on the
    thermal, where r_norm is the radial distance normalized so that
    1 = thermal edge.

    Parameters
    ----------
    thermal_dss : list[xr.Dataset]
        Single-thermal merged datasets from mask_storm_by_thermal.
    storm_dss : list[xr.Dataset] or xr.Dataset
        Corresponding storm datasets.
    storm_vars : tuple of str
        Variables to extract.
    n_r : int
        Number of radial grid points.
    n_theta : int
        Number of polar angle grid points (0 = up, pi = down).
    n_phi : int
        Number of azimuthal angle grid points (0 to 2pi).
    r_norm_max : float
        Max normalized radius (1 = thermal edge).

    Returns
    -------
    xr.Dataset
        Composite with dims (thermal_ix, dt_minutes, r_norm, theta, phi).
        r_norm=1 is the thermal edge in all directions.
    """

    # Define common output grid
    r_grid_vals = np.linspace(0, r_norm_max, n_r)
    theta_grid_vals = np.linspace(0, np.pi, n_theta)
    phi_grid_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Meshgrid for output points (r, theta, phi)
    R_grid, theta_grid, phi_grid = np.meshgrid(
        r_grid_vals, theta_grid_vals, phi_grid_vals, indexing="ij"
    )

    # Precompute trig (invariant across thermals and times)
    sin_theta_cos_phi = np.sin(theta_grid) * np.cos(phi_grid)
    sin_theta_sin_phi = np.sin(theta_grid) * np.sin(phi_grid)
    cos_theta = np.cos(theta_grid)

    all_normalized_dss = []

    # Iterate over thermals
    for this_thermal_ix, this_thermal_ds in tqdm(
        thermal_ds.groupby("thermal_ix"),
        total=len(thermal_ds["thermal_ix"]),
        desc="Normalizing thermals",
    ):

        # Drop times where this thermal didn't exist or has no valid shape
        this_thermal_ds = this_thermal_ds.dropna(dim="time", subset=["sphere_radius"])

        single_time_dss = []
        for t in this_thermal_ds.time.values:
            # Get data for this thermal at this time
            this_time_thermal_ds = this_thermal_ds.sel({"time": t}, drop=False)

            # For each (r_norm, theta, phi), compute the physical (dx, dy, dz)
            # r_norm = distance / sphere_radius, so rho = r_norm * sphere_radius
            sphere_radius = float(this_time_thermal_ds["sphere_radius"].values[0])
            rho = R_grid * sphere_radius

            # Get actual x, y, z coordinates of the points on the spherical
            # coordinates grid
            center_x = float(this_time_thermal_ds["x"].values[0])
            center_y = float(this_time_thermal_ds["y"].values[0])
            center_z = float(this_time_thermal_ds["z"].values[0])
            xyz_grids = {
                "x": rho * sin_theta_cos_phi + center_x,
                "y": rho * sin_theta_sin_phi + center_y,
                "z": rho * cos_theta + center_z,
            }

            # Spatially subset the rams dataser
            pad = 1.2
            subsetted_rams_ds = rams_ds.sel({
                "x": slice(
                    float((center_x - sphere_radius * pad)),
                    float((center_x + sphere_radius * pad)),
                ),
                "y": slice(
                    float((center_y - sphere_radius * pad)),
                    float((center_y + sphere_radius * pad)),
                ),
                "z": slice(
                    float((center_z - sphere_radius * pad)),
                    float((center_z + sphere_radius * pad)),
                ),
            })

            # Interpolate to the thermal-centered grids
            interpolated_vals = interpolate_to_normalized_coords(
                this_time_rams_ds=subsetted_rams_ds.sel({"time": t}),
                xyz_grids=xyz_grids,
                interpolated_vars=composited_vars,
            )
            # Convert these to normalized spherical coordinates
            this_time_spherical_das = {
                k: (
                    ["r_norm", "theta", "phi"],
                    v.reshape(n_r, n_theta, n_phi),
                )
                for k, v in interpolated_vals.items()
            }
            # Concatenate across variables into a dataset for this thermal at this time
            single_time_dss.append(
                xr.Dataset(
                    this_time_spherical_das,
                    coords={
                        "dt_thermal_seconds": (
                            this_time_thermal_ds["dt_thermal_seconds"].values
                        ),
                        "thermal_ix": this_thermal_ix,
                        "r_norm": r_grid_vals,
                        "theta": theta_grid_vals,
                        "phi": phi_grid_vals,
                    },
                ).expand_dims(dim=["thermal_ix"])
                # Now add time as a coordinate on dt_thermal and thermal_ix
                .assign_coords(time=("dt_thermal_seconds", [t]))
            )

        this_thermal_normalized_ds = xr.concat(
            single_time_dss, dim="dt_thermal_seconds", data_vars="all"
        )
        all_normalized_dss.append(this_thermal_normalized_ds)

    concated_ds = xr.concat(
        all_normalized_dss,
        dim="thermal_ix",
        join="outer",
    )

    # Add axisymmetric coordinates

    # Compute Cartesian coordinates from (r_norm, theta)
    # theta=0 is up (positive z), theta=pi is down
    r, th = np.meshgrid(
        concated_ds.r_norm.values, concated_ds.theta.values, indexing="ij"
    )
    x_norm = r * np.sin(th)
    z_norm = r * np.cos(th)

    concated_ds = concated_ds.assign_coords(
        x_norm=(("r_norm", "theta"), x_norm),
        z_norm=(("r_norm", "theta"), z_norm),
    )
    return concated_ds


def sphere_average(normalized_ds):
    """
    Volume-weighted mean over the spherical coordinate grid.

    Parameters
    ----------
    normalized_ds : xr.Dataset
        Output of normalize_thermals with dims
        (..., r_norm, theta, phi).
    r_norm_max : float, optional
        Maximum normalized radius to include (default 1.0, the thermal edge).

    Returns
    -------
    xr.Dataset
        Weighted mean over (r_norm, theta, phi). Remaining dims are
        preserved (e.g. thermal_ix, dt_thermal_seconds).
    """
    weight = normalized_ds["r_norm"] ** 2 * np.sin(normalized_ds["theta"])
    return normalized_ds.weighted(weight).mean(dim=["r_norm", "theta", "phi"])


def thermal_mean(normalized_ds, var):
    """
    Volume-weighted mean of `var` inside the thermal (r_norm <= 1).

    Parameters
    ----------
    normalized_ds : xr.Dataset
        Output of normalize_thermals, with dims (..., r_norm, theta, phi).
    var : str
        Name of the variable to average.

    Returns
    -------
    xr.DataArray
        Mean over the thermal interior. Remaining dims (e.g. thermal_ix,
        dt_thermal_seconds) are preserved.
    """
    interior = normalized_ds[var].sel(r_norm=slice(0, 1))
    weight = interior["r_norm"] ** 2 * np.sin(interior["theta"])
    return interior.weighted(weight).mean(dim=["r_norm", "theta", "phi"])


def thermal_total(normalized_ds, var):
    """
    Volume integral of `var` inside the thermal (r_norm <= 1).

    Uses dV = R^3 * r_norm^2 * sin(theta) d(r_norm) d(theta) d(phi),
    where R is the physical thermal radius read from `normalized_ds["radius"]`.
    For an intensive quantity (e.g. mixing ratio in kg/kg), multiply the
    result by density to get mass.

    Parameters
    ----------
    normalized_ds : xr.Dataset
        Output of normalize_thermals, must contain a `radius` data variable
        with dims (thermal_ix, dt_thermal_seconds) giving the physical
        thermal radius in metres.
    var : str
        Name of the variable to integrate.

    Returns
    -------
    xr.DataArray
        Integral over the thermal interior, with the spatial dims removed.
        Units are (units of var) * m^3.
    """
    interior = normalized_ds[var].sel(r_norm=slice(0, 1))
    integrand = (
        interior
        * interior["r_norm"] ** 2
        * np.sin(interior["theta"])
    )
    integral = integrand.integrate("r_norm").integrate("theta").integrate("phi")
    return integral * normalized_ds["radius"] ** 3
