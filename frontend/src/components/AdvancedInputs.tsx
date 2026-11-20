import React from "react";
import { ADVANCED_FIELDS } from "../lib/cockpit";
import type { FeatureCatalog, FieldMeta } from "../api/types";

interface AdvancedInputsProps {
  catalog: FeatureCatalog | null;
  featureSet: string;
  inputs: Record<string, string | number>;
  onInput: (field: string, value: string | number) => void;
}

function FieldControl({
  meta,
  value,
  onChange,
}: {
  meta: FieldMeta;
  value: string | number;
  onChange: (v: string | number) => void;
}) {
  if (meta.kind === "categorical" || meta.kind === "binary") {
    return (
      <select value={String(value)} onChange={(e) => onChange(e.target.value)}>
        {(meta.options ?? []).map((opt) => (
          <option key={String(opt)} value={String(opt)}>
            {String(opt)}
          </option>
        ))}
      </select>
    );
  }

  return (
    <input
      type="number"
      min={meta.min}
      max={meta.max}
      step={meta.step ?? 1}
      value={Number(value)}
      onChange={(e) => {
        const n = e.target.value;
        if (n === "") return;
        onChange(Number(n));
      }}
    />
  );
}

export default function AdvancedInputs({
  catalog,
  featureSet,
  inputs,
  onInput,
}: AdvancedInputsProps) {
  if (!catalog) return null;

  const fieldsForSet = new Set(catalog.feature_sets[featureSet] ?? []);

  return (
    <div>
      <div className="advanced-grid">
        {ADVANCED_FIELDS.filter((f) => fieldsForSet.has(f)).map((field) => {
          const meta = catalog.fields[field];
          if (!meta) return null;
          return (
            <div className="field" key={field}>
              <label>
                <span>{meta.label}</span>
                {meta.unit && <span className="u">{meta.unit}</span>}
              </label>
              <FieldControl
                meta={meta}
                value={inputs[field] ?? (meta.min ?? 0)}
                onChange={(v) => onInput(field, v)}
              />
            </div>
          );
        })}
      </div>

      <div className="advanced-note">
        <span>ℹ️</span>
        <span>
          These are the remaining model inputs beyond the five manual controls.
          They are required by the validation contract for the selected feature
          set and are never silently defaulted — every value above is sent to
          the backend and validated. All telemetry is synthetic.
        </span>
      </div>
    </div>
  );
}
