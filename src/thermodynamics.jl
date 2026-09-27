# ──────────────────────────────────────────────────────────────────
# Thermodynamic derived composites
#
# These live in the package, rather than in the tracking notebook, because
# normalize_thermals runs derived specs on Distributed worker processes. Every worker loads
# ThermalTracking, so a package function is callable there. A function defined in the
# notebook's Main is not: Distributed sends a named function by name, and the workers never
# defined it.
# ──────────────────────────────────────────────────────────────────

# RAMS thermodynamic constants (lib/rconstants.f90)
const RAMS_DRY_AIR_HEAT_CAPACITY = 1004.0
const RAMS_DRY_AIR_GAS_CONSTANT = 287.0
const RAMS_REFERENCE_PRESSURE = 1.0e5

# Coefficients of the RAMS saturation-vapor-pressure polynomial over liquid (eslf in
# lib/therm_lib.f90), lowest order first, in Celsius temperature
const RAMS_ESLF_COEFFICIENTS = (
	0.6105851e3,
	0.4440316e2,
	0.1430341e1,
	0.2641412e-1,
	0.2995057e-3,
	0.2031998e-5,
	0.6936113e-8,
	0.2564861e-11,
	-0.3704404e-13,
)

"""
	rams_saturation_mixing_ratio_liquid(pressure, temperature)

Saturation mixing ratio over liquid water [kg/kg] at `pressure` [Pa] and `temperature` [K],
computed exactly as RAMS's `rslf` (lib/therm_lib.f90) computes it. The saturation vapor
pressure is an 8th-order polynomial in Celsius temperature, with the temperature floored at
−80 °C as in RAMS, evaluated in Horner form. The mixing ratio is then 0.622 e_s / (p − e_s).
Matching RAMS, rather than using a textbook fit, keeps the derived supersaturation identical
to the one that gates cloud nucleation in the model. Works elementwise on arrays.
"""
function rams_saturation_mixing_ratio_liquid(pressure, temperature)
	celsius = max.(-80.0, temperature .- 273.15)
	# Horner evaluation, starting from the highest-order coefficient
	saturation_vapor_pressure = zero(celsius) .+ RAMS_ESLF_COEFFICIENTS[end]
	for coefficient in reverse(RAMS_ESLF_COEFFICIENTS[1:(end - 1)])
		saturation_vapor_pressure = saturation_vapor_pressure .* celsius .+ coefficient
	end
	return 0.622 .* saturation_vapor_pressure ./ (pressure .- saturation_vapor_pressure)
end

"""
	supersaturation_liquid(native)

Supersaturation over liquid water [%] on the native RAMS grid, from the `THETA`, `PI`, and
`RV` arrays in the `native` Dict that `normalize_thermals` passes to a derived spec.

Temperature is T = THETA · PI / cp, and pressure is p = p00 (PI / cp)^(cp / Rd), with the RAMS
constants. Supersaturation is then 100 (RV / rslf(p, T) − 1), the same quantity RAMS uses to
decide whether cloud nucleation can occur (`excessrv` in mic_nuc.f90). Negative values are
subsaturated.
"""
function supersaturation_liquid(native)
	exner = native["PI"]
	# Temperature [K] and pressure [Pa] from the Exner function
	temperature = native["THETA"] .* exner ./ RAMS_DRY_AIR_HEAT_CAPACITY
	pressure =
		RAMS_REFERENCE_PRESSURE .*
		(exner ./ RAMS_DRY_AIR_HEAT_CAPACITY) .^
		(RAMS_DRY_AIR_HEAT_CAPACITY / RAMS_DRY_AIR_GAS_CONSTANT)
	saturation_mixing_ratio = rams_saturation_mixing_ratio_liquid(pressure, temperature)
	return 100.0 .* (native["RV"] ./ saturation_mixing_ratio .- 1.0)
end

"""
	supersaturated_liquid_flag(native)

1.0 where the air is supersaturated over liquid water (supersaturation ≥ 0) and 0.0
elsewhere, on the native grid. The flag is Float64 rather than Bool so that interpolation and
the azimuthal mean treat it as a number. Its azimuthal mean is then the supersaturated
fraction of each ring. Linear interpolation blends the flag across the edge between a
supersaturated and a subsaturated grid cell, so that fraction is approximate at such edges.
"""
function supersaturated_liquid_flag(native)
	return Float64.(supersaturation_liquid(native) .>= 0.0)
end

"""
	derived_supersaturation(name = "SUPERSAT_LIQ")

Derived composite of supersaturation over liquid water [%] (see `supersaturation_liquid`).
With azimuthal averaging it also gets a `<name>_phi_std` field: the azimuthal spread of
supersaturation, which the azimuthal mean hides.
"""
function derived_supersaturation(name::AbstractString="SUPERSAT_LIQ")::CompositeSpec
	return derived(name, ["THETA", "PI", "RV"], supersaturation_liquid)
end

"""
	derived_supersaturated_flag(name = "IS_SUPERSAT")

Derived composite of the supersaturated flag (see `supersaturated_liquid_flag`). With
azimuthal averaging, the composited field is the supersaturated fraction of each ring.
"""
function derived_supersaturated_flag(name::AbstractString="IS_SUPERSAT")::CompositeSpec
	return derived(name, ["THETA", "PI", "RV"], supersaturated_liquid_flag)
end
