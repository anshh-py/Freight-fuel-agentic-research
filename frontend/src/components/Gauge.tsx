import React, { useMemo } from "react";

export interface Segment {
  from: number;
  to: number;
  color: string;
}

interface GaugeProps {
  label: string;
  unit: string;
  min: number;
  max: number;
  value: number;
  segments: Segment[];
  ticks?: number[];
  format?: (v: number) => string;
}

const CX = 100;
const CY = 100;
const R_OUTER = 78;
const R_INNER = 66;
const R_TICK = 62;
const START_ANGLE = -135;
const SPAN = 270;

function polar(r: number, angleDeg: number): [number, number] {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(rad), CY + r * Math.sin(rad)];
}

function arcPath(r: number, a0: number, a1: number): string {
  const [x0, y0] = polar(r, a0);
  const [x1, y1] = polar(r, a1);
  const large = a1 - a0 > 180 ? 1 : 0;
  return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
}

function angleFor(value: number, min: number, max: number): number {
  const t = (value - min) / (max - min);
  return START_ANGLE + Math.max(0, Math.min(1, t)) * SPAN;
}

export default function Gauge({
  label,
  unit,
  min,
  max,
  value,
  segments,
  ticks = [],
  format = (v) => String(v),
}: GaugeProps) {
  const needleAngle = angleFor(value, min, max);

  const segmentPaths = useMemo(
    () =>
      segments.map((s) => ({
        ...s,
        d: arcPath(
          R_OUTER,
          angleFor(s.from, min, max),
          angleFor(s.to, min, max),
        ),
      })),
    [segments, min, max],
  );

  const tickMarks = useMemo(
    () =>
      ticks.map((t) => {
        const a = angleFor(t, min, max);
        const [x1, y1] = polar(R_TICK, a);
        const [x2, y2] = polar(R_TICK + 6, a);
        const [lx, ly] = polar(R_TICK + 20, a);
        return { value: t, x1, y1, x2, y2, lx, ly };
      }),
    [ticks, min, max],
  );

  return (
    <div className="gauge-card card">
      <div className="gauge-title">{label}</div>
      <svg
        viewBox="0 0 200 178"
        width="100%"
        role="img"
        aria-label={`${label}: ${format(value)} ${unit}`}
      >
        {/* face */}
        <circle cx={CX} cy={CY} r={R_OUTER + 6} fill="#0a1220" />
        <circle cx={CX} cy={CY} r={R_INNER} fill="#0d1524" stroke="#1d2d47" />

        {/* colored operating regions */}
        {segmentPaths.map((s, i) => (
          <path
            key={`${s.from}-${s.to}-${i}`}
            d={s.d}
            fill="none"
            stroke={s.color}
            strokeWidth="9"
            strokeLinecap="butt"
          />
        ))}

        {/* minor face line */}
        <circle cx={CX} cy={CY} r={R_INNER - 8} fill="none" stroke="#1d2d47" strokeWidth="1.5" />

        {/* tick marks + labels */}
        {tickMarks.map((t) => (
          <g key={t.value}>
            <line
              x1={t.x1}
              y1={t.y1}
              x2={t.x2}
              y2={t.y2}
              stroke="#5c6f8c"
              strokeWidth="2"
            />
            <text
              x={t.lx}
              y={t.ly}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="9.5"
              fill="#8da2bd"
              fontWeight="600"
            >
              {format(t.value)}
            </text>
          </g>
        ))}

        {/* needle (animated) */}
        <g
          style={{
            transform: `rotate(${needleAngle}deg)`,
            transformOrigin: `${CX}px ${CY}px`,
            transition: "transform 480ms cubic-bezier(0.25, 0.8, 0.3, 1)",
          }}
        >
          <line
            x1={CX}
            y1={CY}
            x2={CX}
            y2={CY - R_INNER + 8}
            stroke="#e6edf7"
            strokeWidth="3.5"
            strokeLinecap="round"
          />
          <line
            x1={CX}
            y1={CY}
            x2={CX}
            y2={CY - 14}
            stroke="#35c4ff"
            strokeWidth="5"
            strokeLinecap="round"
          />
        </g>

        {/* hub */}
        <circle cx={CX} cy={CY} r="6.5" fill="#e6edf7" />
        <circle cx={CX} cy={CY} r="3" fill="#0d1524" />
      </svg>

      <div className="gauge-value">
        {format(value)}
        <span className="gauge-unit"> {unit}</span>
      </div>
    </div>
  );
}
