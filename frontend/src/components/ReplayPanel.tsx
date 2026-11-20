import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type {
  ReplayCatalog,
  ReplayPoint,
  ReplayResponse,
} from "../api/types";

export interface ReplayLoadAction {
  engine_rpm: number;
  speed_kph: number;
  engine_load_pct: number;
  throttle_position_pct: number;
  road_grade_pct: number;
}

interface ReplayPanelProps {
  onLoadReplay: (r: ReplayResponse) => void;
  onPoint: (p: ReplayPoint | null) => void;
  onProgress: (index: number) => void;
  onLoadSampleToInputs: (a: ReplayLoadAction) => void;
}

const SPEEDS = [0.5, 1, 2, 4];
const SAMPLE = 360;

function formatTime(ts?: string): string {
  if (!ts) return "—";
  return ts.replace("T", " ").replace("Z", "").replace(/\.\d+$/, "");
}

export default function ReplayPanel({
  onLoadReplay,
  onPoint,
  onProgress,
  onLoadSampleToInputs,
}: ReplayPanelProps) {
  const [catalog, setCatalog] = useState<ReplayCatalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [truckId, setTruckId] = useState<string>("");
  const [tripId, setTripId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [replay, setReplay] = useState<ReplayResponse | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [index, setIndex] = useState(0);

  const trucks = catalog?.trucks ?? [];
  const truck = trucks.find((t) => t.truck_id === truckId);
  const trips = truck?.trips ?? [];
  const trip = trips.find((t) => t.trip_id === tripId);
  const points = replay?.points ?? [];
  const currentPoint = points[index] ?? null;

  /* fetch the truck/trip catalog on mount */
  useEffect(() => {
    api
      .replayCatalog()
      .then((c) => {
        setCatalog(c);
        const first = c.trucks[0];
        if (first) {
          setTruckId(first.truck_id);
          setTripId(first.trips[0]?.trip_id ?? "");
        }
      })
      .catch((e) => {
        setCatalogError(
          e instanceof ApiError
            ? `Replay catalog error (${e.status}): ${e.message}`
            : "Could not load replay catalog.",
        );
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* when truck changes, pick its first trip */
  const handleTruckChange = useCallback(
    (id: string) => {
      setTruckId(id);
      const t = trucks.find((x) => x.truck_id === id);
      setTripId(t?.trips[0]?.trip_id ?? "");
      setPlaying(false);
    },
    [trucks],
  );

  /* fetch replay when a trip is selected */
  useEffect(() => {
    if (!tripId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPlaying(false);
    api
      .replay(tripId, SAMPLE)
      .then((r) => {
        if (cancelled) return;
        setReplay(r);
        setIndex(0);
        onLoadReplay(r);
        onPoint(r.points[0] ?? null);
      })
      .catch((e) => {
        if (cancelled) return;
        setReplay(null);
        setError(
          e instanceof ApiError
            ? `Replay error (${e.status}): ${e.message}`
            : "Could not load replay data.",
        );
        onPoint(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      onPoint(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tripId]);

  /* playback timer (cleanup on unmount / speed / trip change) */
  useEffect(() => {
    if (!playing || points.length < 2) return;
    const ms = Math.max(80, 1000 / speed);
    const id = window.setInterval(() => {
      setIndex((i) => {
        if (i >= points.length - 1) {
          setPlaying(false); // end of trip
          return i;
        }
        return i + 1;
      });
    }, ms);
    return () => window.clearInterval(id);
  }, [playing, speed, points.length]);

  /* report current point + progress outward */
  useEffect(() => {
    onPoint(currentPoint);
    onProgress(index);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index]);

  const handleReset = useCallback(() => {
    setPlaying(false);
    setIndex(0);
  }, []);

  const handleSeek = useCallback((i: number) => {
    const clamped = Math.max(0, Math.min(points.length - 1, i));
    setIndex(clamped);
  }, [points.length]);

  const progress = points.length > 1 ? index / (points.length - 1) : 0;

  return (
    <section className="card area-replay">
      <div className="card-head">
        <div className="card-title">
          <span className="icon">🔄</span> Synthetic telemetry replay
        </div>
        <span className="tag">historical · synthetic</span>
      </div>

      <div className="card-pad">
        <div className="replay-source-note">
          <span>⚠️</span>
          <span>
            Replay shows pre-recorded <strong>synthetic</strong> telemetry from
            the research dataset — not a live truck. Its fuel-rate values are
            formula-generated targets, not measured consumption. Production
            prediction only from a successful backend model run.
          </span>
        </div>

        {catalogError && (
          <div className="replay-error" role="alert">
            {catalogError}
          </div>
        )}

        {!catalog && !catalogError && (
          <div className="skeleton pulse" style={{ height: 120 }} />
        )}

        <div className="replay-selects">
          <label className="replay-field">
            <span className="replay-label">Truck</span>
            <select
              value={truckId}
              onChange={(e) => handleTruckChange(e.target.value)}
              disabled={!trucks.length}
            >
              {!truckId && <option value="">Select truck…</option>}
              {trucks.map((t) => (
                <option key={t.truck_id} value={t.truck_id}>
                  {t.truck_id} · {t.vehicle_class}
                </option>
              ))}
            </select>
          </label>

          <label className="replay-field">
            <span className="replay-label">Trip</span>
            <select
              value={tripId}
              onChange={(e) => setTripId(e.target.value)}
              disabled={!trips.length}
            >
              {!tripId && <option value="">Select trip…</option>}
              {trips.map((t) => (
                <option key={t.trip_id} value={t.trip_id}>
                  {t.trip_id} · {t.route_type} · {t.record_count} rows
                </option>
              ))}
            </select>
          </label>
        </div>

        {error && (
          <div className="replay-error" role="alert">
            {error}
          </div>
        )}

        {loading && <div className="skeleton pulse" style={{ height: 120 }} />}

        {!loading && replay && (
          <>
            <div className="replay-meta">
              {trip ? (
                <span>
                  {trip.route_type} · {trip.record_count.toLocaleString()} records ·{" "}
                  {trip.distance_km.toFixed(1)} km · {formatTime(trip.start_timestamp_utc)} →{" "}
                  {formatTime(trip.end_timestamp_utc)}
                </span>
              ) : (
                <span>{replay.trip_id}</span>
              )}
              <span className="muted-2">
                showing {replay.record_count_sampled} of{" "}
                {replay.record_count_total.toLocaleString()} sampled rows
              </span>
            </div>

            {/* transport controls */}
            <div className="replay-transport">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setPlaying((p) => !p)}
                disabled={points.length < 2}
              >
                {playing ? "⏸ Pause" : "▶ Play"}
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={handleReset}
                disabled={index === 0 && !playing}
              >
                ⏮ Reset
              </button>
              <label className="replay-speed">
                <span>Speed</span>
                <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
                  {SPEEDS.map((s) => (
                    <option key={s} value={s}>
                      {s}×
                    </option>
                  ))}
                </select>
              </label>
              <span className="replay-status">
                {playing ? (
                  <span className="status-line status-ok">
                    <span className="dot dot-ok" /> replaying
                  </span>
                ) : index >= points.length - 1 && points.length ? (
                  <span className="status-line status-warn">
                    <span className="dot dot-warn" /> trip ended
                  </span>
                ) : (
                  <span className="status-line status-idle">
                    <span className="dot dot-idle" /> paused
                  </span>
                )}
              </span>
            </div>

            {/* progress / seek */}
            <div className="replay-progress">
              <input
                type="range"
                min={0}
                max={Math.max(0, points.length - 1)}
                value={index}
                onChange={(e) => handleSeek(Number(e.target.value))}
                aria-label="Replay timeline seek"
              />
              <div className="replay-progress-meta">
                <span>{formatTime(currentPoint?.timestamp)}</span>
                <span>
                  {index + 1} / {points.length} · {Math.round(progress * 100)}%
                </span>
              </div>
            </div>

            {/* live readout */}
            {currentPoint && (
              <div className="replay-readout">
                <div className="replay-reading">
                  <span className="rk">Speed</span>
                  <span className="rv">{currentPoint.speed_kph.toFixed(1)} km/h</span>
                </div>
                <div className="replay-reading">
                  <span className="rk">Engine RPM</span>
                  <span className="rv">{Math.round(currentPoint.engine_rpm)} rpm</span>
                </div>
                <div className="replay-reading">
                  <span className="rk">Engine load</span>
                  <span className="rv">{currentPoint.engine_load_pct.toFixed(1)}%</span>
                </div>
                <div className="replay-reading">
                  <span className="rk">Throttle</span>
                  <span className="rv">
                    {(currentPoint.throttle_position_pct ?? 0).toFixed(1)}%
                  </span>
                </div>
                <div className="replay-reading">
                  <span className="rk">Grade</span>
                  <span className="rv">{currentPoint.road_grade_pct?.toFixed(2)}%</span>
                </div>
                <div className="replay-reading replay-fuel">
                  <span className="rk">Fuel rate (target)</span>
                  <span className="rv">{currentPoint.fuel_rate_lph.toFixed(2)} L/h</span>
                </div>
              </div>
            )}

            {/* explicit load into prediction inputs */}
            {currentPoint && (
              <button
                type="button"
                className="btn btn-ghost replay-load"
                onClick={() =>
                  onLoadSampleToInputs({
                    engine_rpm: currentPoint.engine_rpm,
                    speed_kph: currentPoint.speed_kph,
                    engine_load_pct: currentPoint.engine_load_pct,
                    throttle_position_pct: currentPoint.throttle_position_pct ?? 0,
                    road_grade_pct: currentPoint.road_grade_pct ?? 0,
                  })
                }
              >
                ⬇ Load this replay sample into prediction inputs
              </button>
            )}
          </>
        )}
      </div>
    </section>
  );
}
