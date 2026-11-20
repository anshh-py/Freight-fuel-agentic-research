import React from "react";

interface ControlSliderProps {
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}

export default function ControlSlider({
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
  format = (v) => String(v),
}: ControlSliderProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="control-row">
      <div className="control-label">
        <span className="name">{label}</span>
        <span className="unit">{unit}</span>
      </div>

      <input
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ "--fill": `${pct}%` } as React.CSSProperties}
        aria-label={`${label} (${unit})`}
      />

      <div className="stepper" aria-label={`${label} stepper`}>
        <button
          type="button"
          onClick={() => onChange(Math.max(min, Number((value - step).toFixed(2))))}
          disabled={value <= min}
          aria-label={`Decrease ${label}`}
        >
          −
        </button>
        <span className="sep" />
        <span className="control-value">{format(value)}</span>
        <span className="sep" />
        <button
          type="button"
          onClick={() => onChange(Math.min(max, Number((value + step).toFixed(2))))}
          disabled={value >= max}
          aria-label={`Increase ${label}`}
        >
          +
        </button>
      </div>
    </div>
  );
}
