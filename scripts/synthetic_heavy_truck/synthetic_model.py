"""Physical trip model and telemetry generation for the synthetic dataset.

This module turns a truck + route + duration into a single trip's telemetry
array. The physics is deliberately simple but internally consistent:

* Motion: a commanded speed profile is low-pass filtered with acceleration
  limits to give a realistic speed trace; acceleration is the time derivative.
* Forces: rolling resistance, aerodynamic drag, grade (sin(atan(%/100)))
  and inertial force are summed; tractive power follows from force x speed.
* Motion is power-limited: commanded acceleration is capped by the tractive
  force the engine can actually deliver at the current speed, so acceleration
  is consistent with available power and never demands more than rated power.
* Engine: internal parasitic power (friction+pumping+auxiliaries, scaled by
  displacement and RPM) plus tractive power set the total engine output; load
  is that total relative to rated power; RPM tracks speed + demand and starts
  at the configured idle RPM; torque is total power / RPM; throttle follows the
  speed-error control signal.
* Fuel: the fuel rate is computed from total engine output via a thermal
  efficiency map (no constant additive idle term - idle fuel emerges naturally
  from parasitic power at low load), then scaled by the truck-specific
  efficiency factor. Per-second consumed increments are fuel_rate / 3600 and
  cumulative fuel is the within-trip cumulative sum, so increments agree with
  the rate and sampling interval and cumulative fuel resets at each trip
  boundary.
* Engine-off records have zero speed/engine/fuel values while temperatures
  evolve smoothly (coolant warms while running and decays toward ambient after
  shutdown).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_config import (
    AIR_DENSITY_KG_M3,
    CLASS_SPECS,
    DIESEL_ENERGY_J_PER_L,
    DRIVETRAIN_EFFICIENCY,
    GRAVITY_MPS2,
    IDLE_PARASITIC_KW_PER_L,
    ROUTE_SPECS,
    SAMPLING_INTERVAL_S,
)

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

def _smooth(x: np.ndarray, window: int = 40) -> np.ndarray:
    if len(x) < 2:
        return x.copy()
    if window > len(x):
        window = max(1, len(x) // 2)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ---------------------------------------------------------------------------
# Motion profile
# ---------------------------------------------------------------------------

def build_speed_profile(
    active_d: int,
    route_spec: dict,
    rng: np.random.Generator,
    truck: dict,
    grade: np.ndarray,
    ramp_s: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (target_speed_mps, actual_speed_mps) over an active driving block.

    The target profile is a cruise baseline with stochastic stops and a final
    deceleration; the actual speed follows the target under acceleration limits
    AND under the tractive-power limit of the truck's engine. This makes the
    commanded acceleration consistent with the available engine power,
    drivetrain efficiency, vehicle mass and road resistance: whenever the
    power demand would exceed rated power, the achieved acceleration is capped
    at what the engine can actually deliver at the current speed.
    """
    cruise = route_spec["cruise_kph"] / 3.6
    vmax = route_spec["vmax_kph"] / 3.6
    sigma = route_spec["sigma_kph"] / 3.6

    # commanded profile
    target = np.full(active_d, cruise)
    # speed noise
    target = target + rng.normal(0.0, sigma, active_d)
    target = _clip(target, 0.0, vmax)

    # initial pull-out ramp: accelerate from a stop to cruise
    if ramp_s is None:
        ramp_s = int(round(rng.uniform(20, 60)))
    ramp_s = min(ramp_s, max(1, active_d // 2))
    ramp = np.linspace(0.0, cruise * 0.92, ramp_s)
    target[:ramp_s] = ramp

    # full stops distributed over the route
    n_stops = int(round(route_spec["stops_per_hour"] * active_d / 3600.0))
    stop_min, stop_max = route_spec["stop_dur"]
    if n_stops > 0 and active_d > 120:
        stop_start = rng.integers(ramp_s + 10, max(ramp_s + 11, active_d - 90), size=n_stops)
        for s in stop_start:
            dur = int(rng.integers(stop_min, stop_max + 1))
            e = min(s + dur, active_d - 5)
            if e > s:
                target[s:e] = 0.0

    # final deceleration to stop
    fin = int(rng.integers(25, 60))
    fin = min(fin, max(1, active_d // 3))
    if fin > 1 and active_d > fin:
        target[-fin:] = np.linspace(target[-fin], 0.0, fin)

    # ---- truck physics for power-limited acceleration ----
    m = truck["vehicle_mass_kg"]
    crr = truck["crr"]
    cda = truck["cda_m2"]
    rated_kw = truck["rated_power_kw"]
    f_roll = crr * m * GRAVITY_MPS2
    grade_arr = np.asarray(grade, dtype=float)

    def _a_power_cap(v: float, g: float) -> float:
        """Achievable acceleration (m/s^2) at speed v (m/s) on grade g (%).

        Solved implicitly at the END-of-step speed v + a*dt, so the tractive
        power demand evaluated at the next telemetry sample is exactly the one
        used to cap acceleration (no hidden current-speed vs end-of-step
        mismatch). May be negative when the grade is too steep for the
        available power, producing a natural, physically consistent slow-down
        (the truck settles to the speed where demand == rated power). Could        """
        if v <= 1e-3:
            return 0.85  # launching from a stop; power limit is not binding
        f_g = m * GRAVITY_MPS2 * math.sin(math.atan(min(max(g, -89.0), 89.0) / 100.0))
        a = 0.0
        for _ in range(20):
            vn = v + a * SAMPLING_INTERVAL_S
            f_avail = rated_kw * 1000.0 * DRIVETRAIN_EFFICIENCY / vn
            f_aero = 0.5 * AIR_DENSITY_KG_M3 * cda * vn * vn
            a_new = (f_avail - (f_roll + f_aero + f_g)) / m
            a = 0.5 * a + 0.5 * max(-1.60, min(0.85, a_new))
        return a

    a_accel = 0.85   # m/s^2 comfortable acceleration
    a_decel = 1.60   # m/s^2 braking (not power-limited)
    k = 0.8
    speed = np.zeros(active_d)
    vprev = 0.0
    for i in range(active_d):
        err = target[i] - vprev
        a_des = max(-a_decel, min(a_accel, k * err))
        a_power = _a_power_cap(vprev, grade_arr[i])
        # Unified power-feasibility rule: the achieved acceleration never exceeds
        # what the engine can deliver (a_power), for both signs. On a grade too
        # steep for the available power this becomes negative (a natural slow-
        # down), including when the command is to coast/hold - otherwise the
        # truck would hold too high a speed and demand more than rated power.
        # Voluntary braking (target clearly below current speed) still applies
        # when it is less aggressive than the power-limited value.
        a_eff = min(a_des, a_power)
        a_eff = max(-a_decel, min(a_accel, a_eff))
        vprev = max(0.0, vprev + a_eff * SAMPLING_INTERVAL_S)
        speed[i] = vprev

    return target, speed


def build_grade_profile(active_d: int, route_spec: dict,
                        rng: np.random.Generator) -> np.ndarray:
    """Return a smooth road-grade (%) profile over the active block."""
    gmax = route_spec["gmax"]
    mode = route_spec.get("grade_mode", "random_walk")

    if mode == "segments":
        # Deterministic, smooth uphill/downhill segments (used for highway).
        # Alternating climb/descent spans with realistic magnitudes, stitched
        # with smooth (interpolated + rounded) transitions - never independent
        # per-record random grades, so grade is continuous and route-consistent.
        n_seg = int(rng.integers(4, 9))
        edges = np.linspace(0.0, active_d, n_seg + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        mag = rng.uniform(0.35, 1.0, size=n_seg) * gmax
        base_sign = float(rng.choice([1.0, -1.0]))
        signs = base_sign * (-1.0) ** np.arange(n_seg)
        targets = np.clip(signs * mag, -gmax, gmax)
        xs = np.concatenate(([0.0], mid, [float(max(0, active_d - 1))]))
        ys = np.concatenate(([0.0], targets, [0.0]))
        grade = np.interp(np.arange(active_d), xs, ys)
        grade = _smooth(grade, window=min(25, max(2, active_d // 24)))
        grade = _clip(grade, -gmax, gmax)
        grade[0] = 0.0
        return grade

    # default: smooth random-walk grade (city / urban / mountainous)
    gsigma = route_spec["gsigma"]
    raw = rng.normal(0.0, gsigma, active_d)
    smoothed = _smooth(raw, window=min(45, max(2, active_d // 8)))
    grade = _clip(smoothed, -gmax, gmax)
    grade[0] = 0.0
    return grade


# ---------------------------------------------------------------------------
# Engine / fuel model
# ---------------------------------------------------------------------------

def engine_telemetry(
    speed_mps: np.ndarray,
    accel_mps2: np.ndarray,
    grade_pct: np.ndarray,
    truck: dict,
    route_spec: dict,
) -> dict:
    """Compute engine load, RPM, torque, throttle and fuel rate (L/h).

    Fuel model (all SI/dimensionally consistent):
      * Tractive force (N) = roll + aero + grade + inertia.
      * Grade force uses sin(atan(percent/100)) - the correct conversion of a
        percentage slope to the sine of its angle.
      * Wheel power (W) = force x speed; engine tractive power (kW) =
        wheel power / drivetrain efficiency.
      * Total engine output = tractive power + internal parasitic power.
      * Engine load (%) = 8 at zero output .. 100 at rated total output.
      * Fuel (L/h) = total output (kW) * 1000 / (thermal_efficiency *
        diesel_energy_density (J/L)) * 3600. Thermal efficiency falls from
        0.40 at low load to 0.25 at full load. Stationary idle fuel emerges
        naturally from parasitic power (about 1-3 L/h); there is no constant
        additive idle term applied to driving records.
    """
    m = truck["vehicle_mass_kg"]
    crr = truck["crr"]
    cda = truck["cda_m2"]
    rated_kw = truck["rated_power_kw"]
    rated_rpm = truck["rated_rpm"]
    idle_rpm = truck["idle_rpm"]
    disp_l = float(truck.get("displacement_l", 11.0))
    rated_torque = rated_kw * 1000.0 * 9.549 / max(1.0, rated_rpm)  # Nm

    # forces (N) - grade via sin(atan(percent/100)), not sin(percent-as-degrees)
    r = 0.5 * AIR_DENSITY_KG_M3 * cda * speed_mps ** 2
    f_roll = crr * m * GRAVITY_MPS2 * np.ones_like(speed_mps)
    f_grade = m * GRAVITY_MPS2 * np.sin(np.arctan(np.clip(grade_pct, -89.0, 89.0) / 100.0))
    # inertia: signed - positive acceleration demands tractive force, while on
    # deceleration the vehicle's momentum assists, reducing the required
    # tractive power (realistic overrun / engine-braking behaviour). This must
    # match the power-limited motion solver, which also treats negative
    # acceleration as reducing the force demand; otherwise grade-limited
    # deceleration would appear to demand more than rated power.
    f_inertia = m * accel_mps2
    f_need = np.maximum(f_roll + r + f_grade + f_inertia, 0.0)

    # engine tractive power demand (kW)
    p_wheels_w = f_need * speed_mps
    p_eng_kw = p_wheels_w / 1000.0 / DRIVETRAIN_EFFICIENCY
    # Motion is power-limited in build_speed_profile, so demand is <= rated by
    # construction (this is verified by the power-feasibility validation check).
    # Nothing is clipped here to hide an upstream demand error; load below is a
    # definitional 0..100 fraction of rated power, and any real over-demand would
    # show up in the validation check rather than being masked.

    # RPM: starts at the configured idle rpm and rises only under demand (no
    # artificial 0.30 fraction floor)
    cruise_ref = max(1e-3, route_spec["cruise_kph"] / 3.6 * 1.1)
    frac = _clip(speed_mps / cruise_ref, 0.0, 1.0)
    rpm = idle_rpm + (rated_rpm - idle_rpm) * frac ** 1.3
    rpm = rpm + np.maximum(accel_mps2, 0.0) * 140.0
    rpm = _clip(rpm, idle_rpm, rated_rpm + 300.0)
    rpm_norm = _clip((rpm - idle_rpm) / max(1.0, rated_rpm - idle_rpm), 0.0, 1.0)

    # internal parasitic power (friction+pumping+auxiliaries), scales with RPM
    p_para_idle = IDLE_PARASITIC_KW_PER_L * disp_l
    p_para_full = p_para_idle * 2.2
    p_parasitic_kw = p_para_idle + (p_para_full - p_para_idle) * rpm_norm ** 1.5

    # engine load (%): 8 at zero output, 100 at rated total output
    p_total_kw = p_eng_kw + p_parasitic_kw
    load = 8.0 + 92.0 * _clip(p_total_kw / rated_kw, 0.0, 1.0)

    # torque (Nm) from total engine output and RPM
    torque = np.where(rpm > 1.0, p_total_kw * 1000.0 * 9.549 / rpm, 0.0)
    torque = _clip(torque, 0.0, rated_torque)

    # fuel rate (L/h) from total engine output and thermal efficiency
    eta_therm = np.maximum(0.40 - 0.06 * (load / 100.0), 0.25)
    fuel_rate = p_total_kw * 1000.0 / (eta_therm * DIESEL_ENERGY_J_PER_L) * 3600.0
    fuel_rate = fuel_rate / truck["synthetic_efficiency_factor"]
    fuel_rate = _clip(fuel_rate, 0.0, 220.0)

    return dict(
        engine_rpm=rpm,
        engine_load_pct=load,
        torque_nm=torque,
        fuel_rate_lph=fuel_rate,
    )


def compute_throttle(target_speed: np.ndarray, speed: np.ndarray,
                     accel: np.ndarray, grade: np.ndarray) -> np.ndarray:
    """Throttle position (%) from the speed-error control signal."""
    err = target_speed - speed
    base = _clip(18.0 + grade * 2.5, 5.0, 45.0)
    throttle = base + 70.0 * np.tanh(err / 2.5) + np.maximum(accel, 0.0) * 25.0
    throttle = _clip(throttle, 0.0, 100.0)
    throttle = np.where(accel < -0.4, 0.0, throttle)
    return throttle


# ---------------------------------------------------------------------------
# Temperatures
# ---------------------------------------------------------------------------

def build_temperatures(
    total_d: int,
    ignition: np.ndarray,
    rpm_full: np.ndarray,
    ambient_base: float,
    rng: np.random.Generator,
) -> dict:
    """Coolant / intake / ambient temperatures (all smooth, no jumps)."""
    # ambient: slow sinusoidal drift + tiny noise around trip baseline
    t = np.arange(total_d)
    ambient = ambient_base + 1.5 * np.sin(2.0 * np.pi * t / max(1, total_d * 2) + rng.uniform(0, 6.28))
    ambient = ambient + rng.normal(0.0, 0.35, total_d)
    # smooth ambient a little
    ambient = _smooth(ambient, window=min(30, max(2, total_d // 20)))

    # intake air: ambient + engine-bay heating while running, scales with RPM
    heat = np.where(ignition == 1, 14.0 + 8.0 * _clip(rpm_full / 1900.0, 0.0, 1.0), 3.0)
    intake = ambient + heat

    # coolant: warms toward operating temp while running, decays toward ambient off
    op_temp = 88.0
    coolant = np.empty(total_d)
    val = ambient[0]
    tau_on = 150.0
    tau_off = 900.0
    warm = 1.0 - math.exp(-SAMPLING_INTERVAL_S / tau_on)
    cool = 1.0 - math.exp(-SAMPLING_INTERVAL_S / tau_off)
    for i in range(total_d):
        if ignition[i] == 1:
            # small load-dependent departure from the operating point
            target_t = op_temp + 6.0 * _clip(rpm_full[i] / 1900.0, 0.0, 1.0)
            val = val + (target_t - val) * warm
        else:
            val = val + (ambient[i] - val) * cool
        coolant[i] = val
    return dict(
        ambient_temp_c=ambient,
        intake_air_temp_c=intake,
        coolant_temp_c=coolant,
    )


# ---------------------------------------------------------------------------
# Trip assembly
# ---------------------------------------------------------------------------

class TripGenerator:
    """Creates one trip's telemetry DataFrame."""

    def __init__(self, rng: np.random.Generator, rid_cursor: List[int]):
        self.rng = rng
        self.rid = rid_cursor  # mutable single-element list

    def generate(
        self,
        truck: dict,
        trip_id: str,
        start_ts: pd.Timestamp,
        route_name: str,
        duration_s: int,
    ) -> Tuple[pd.DataFrame, dict]:
        rng = self.rng
        D = int(duration_s)
        route_spec = ROUTE_SPECS[route_name]

        # split into pre-ignition, active, post-shutdown blocks
        pre_off = int(rng.integers(0, 12))
        post_off = int(rng.integers(0, 16))
        active_d = D - pre_off - post_off
        if active_d < 30:
            # shrink off-blocks so the active block is meaningful
            pre_off = 0
            post_off = max(0, D - 60)
            active_d = D - pre_off - post_off
        if active_d < 20:
            active_d = max(1, D)
            pre_off = 0
            post_off = 0

        # ---- motion ----
        # grade is built first so the power-limited speed profile can account for
        # the climb resistance when capping achievable acceleration
        grade = build_grade_profile(active_d, route_spec, rng)
        target_speed, speed = build_speed_profile(active_d, route_spec, rng, truck, grade)
        accel = np.diff(speed, prepend=0.0)

        # ---- engine / fuel over the active block ----
        eng = engine_telemetry(speed, accel, grade, truck, route_spec)
        throttle = compute_throttle(target_speed, speed, accel, grade)

        # ---- assemble full-length (D) arrays with off-blocks zeroed ----
        ignition = np.concatenate([np.zeros(pre_off), np.ones(active_d), np.zeros(post_off)])
        speed_full = np.concatenate([np.zeros(pre_off), speed, np.zeros(post_off)])
        accel_full = np.concatenate([np.zeros(pre_off), accel, np.zeros(post_off)])
        grade_full = np.concatenate([np.zeros(pre_off), grade, np.zeros(post_off)])
        rpm_full = np.concatenate([np.zeros(pre_off), eng["engine_rpm"], np.zeros(post_off)])
        load_full = np.concatenate([np.zeros(pre_off), eng["engine_load_pct"], np.zeros(post_off)])
        torque_full = np.concatenate([np.zeros(pre_off), eng["torque_nm"], np.zeros(post_off)])
        thr_full = np.concatenate([np.zeros(pre_off), throttle, np.zeros(post_off)])
        fuel_rate_full = np.concatenate([np.zeros(pre_off), eng["fuel_rate_lph"], np.zeros(post_off)])

        # ---- fuel increments consistent with rate and interval ----
        increment = fuel_rate_full / 3600.0 * SAMPLING_INTERVAL_S
        # Cumulative fuel is the within-trip running sum of increments. Engine-off
        # increments are zero, so cumulative is flat during engine-off blocks and
        # it resets to ~0 at the start of each trip.
        cumulative = np.cumsum(increment)

        # ---- temperatures ----
        ambient_base = float(rng.normal(20.0, 9.0))
        temps = build_temperatures(
            D, ignition, rpm_full, ambient_base, rng,
        )

        # ---- coordinates: move along a fixed heading ----
        lat0 = float(rng.uniform(23.0, 31.0))
        lon0 = float(rng.uniform(72.0, 80.0))
        heading = float(rng.uniform(0.0, 2.0 * math.pi))
        # step = speed [m/s] * 1s, converted to ~degrees (1 deg lat ~ 111 km)
        dx = speed_full * math.cos(heading) / 111000.0
        dy = speed_full * math.sin(heading) / 111000.0
        lat = lat0 + np.cumsum(dy)
        lon = lon0 + np.cumsum(dx)

        # ---- distance / aggregates ----
        distance_km = float(np.sum(speed_full * SAMPLING_INTERVAL_S) / 1000.0)
        duration_seconds = D

        # ---- timestamps ----
        times = [start_ts + pd.Timedelta(seconds=i) for i in range(D)]

        # ---- record ids ----
        ids = np.arange(self.rid[0], self.rid[0] + D, dtype=np.int64)
        self.rid[0] += D

        df = pd.DataFrame({
            "record_id": ids,
            "timestamp_utc": [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in times],
            "truck_id": truck["truck_id"],
            "trip_id": trip_id,
            "vehicle_class": truck["vehicle_class"],
            "gross_vehicle_weight_kg": truck["gross_vehicle_weight_kg"],
            "payload_kg": truck["payload_kg"],
            "vehicle_mass_kg": truck["vehicle_mass_kg"],
            "speed_kph": speed_full * 3.6,
            "acceleration_mps2": accel_full,
            "road_grade_pct": grade_full,
            "latitude": lat,
            "longitude": lon,
            "ignition_status": ignition.astype(np.int8),
            "engine_rpm": rpm_full,
            "engine_load_pct": load_full,
            "throttle_position_pct": thr_full,
            "torque_nm": torque_full,
            "coolant_temp_c": temps["coolant_temp_c"],
            "intake_air_temp_c": temps["intake_air_temp_c"],
            "ambient_temp_c": temps["ambient_temp_c"],
            "fuel_rate_lph": fuel_rate_full,
            "fuel_consumed_l_increment": increment,
            "cumulative_fuel_l_trip": cumulative,
        })

        agg = {
            "trip_id": trip_id,
            "truck_id": truck["truck_id"],
            "vehicle_class": truck["vehicle_class"],
            "start_timestamp_utc": start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_timestamp_utc": times[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "record_count": D,
            "duration_seconds": duration_seconds,
            "distance_km": round(distance_km, 4),
            "payload_kg": truck["payload_kg"],
            "route_type": route_name,
        }
        return df, agg
