import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Pill, Chrome } from "../components/primitives";
import { WF } from "../tokens";
import type { DatasetData } from "../App";
import type { Filters, Sort } from "../types";
import {
  demandCounts,
  distinctSpecs,
  filterStimuli,
  sortStimuli,
  varyingParamKeys,
} from "../lib/derive";
import { ManifestTab } from "./ManifestTab";

const EMPTY_FILTERS: Filters = {
  search: "",
  demands: { mode: "any", values: [] },
  paramQuery: "",
  idFrom: "",
  idTo: "",
};

export function DatasetPage({ data }: { data: DatasetData }) {
  const { manifest, stimuli } = data;
  const navigate = useNavigate();
  const [tab, setTab] = useState<"stimuli" | "manifest">("stimuli");
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [sort, setSort] = useState<Sort>({ key: "sample_id", dir: "asc" });

  const paramCols = useMemo(() => varyingParamKeys(manifest.specs), [manifest.specs]);
  const counts = useMemo(() => demandCounts(stimuli), [stimuli]);
  const specCount = useMemo(
    () => distinctSpecs(manifest, stimuli).filter((s) => s.count > 0).length,
    [manifest, stimuli],
  );

  const rows = useMemo(
    () => sortStimuli(filterStimuli(stimuli, filters), sort),
    [stimuli, filters, sort],
  );

  const cols = [
    { k: "sample_id", label: "sample_id", w: "80px" },
    { k: "spec", label: "spec", w: "50px" },
    { k: "demands", label: "demands", w: "240px" },
    ...paramCols.map((k) => ({ k, label: k, w: "110px" })),
    { k: "modality", label: "modality", w: "80px" },
    { k: "target", label: "target", w: "80px" },
    { k: "validators", label: "validators_ran", w: "120px" },
  ];
  const gridCols = cols.map((c) => c.w).join(" ");

  function toggleSort(key: string) {
    setSort((s) =>
      s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" },
    );
  }

  return (
    <div style={pageStyle}>
      <Chrome>
        <span style={{ color: WF.text, fontWeight: 600 }}>Dataset</span>
        <span style={{ color: WF.textDim }}>·</span>
        <span>{manifest.name}/</span>
        <span style={{ flex: 1 }} />
      </Chrome>

      <TabBar
        tabs={[
          { k: "stimuli", label: "Stimuli", hint: String(stimuli.length) },
          { k: "manifest", label: "Manifest", hint: null },
        ]}
        active={tab}
        onSelect={(k) => setTab(k as "stimuli" | "manifest")}
      />

      {tab === "manifest" ? (
        <ManifestTab data={data} />
      ) : (
        <>
          <SummaryStrip data={data} onViewManifest={() => setTab("manifest")} />
          <FilterBar
            filters={filters}
            setFilters={setFilters}
            demandOptions={counts}
            shown={rows.length}
            total={stimuli.length}
          />

          <div
            style={{
              flex: 1,
              minHeight: 0,
              overflow: "hidden",
              background: WF.panel,
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div style={{ ...headerRow, gridTemplateColumns: gridCols }}>
              {cols.map((c) => (
                <button
                  key={c.k}
                  onClick={() => toggleSort(c.k)}
                  style={sortHeaderBtn}
                  title="sort"
                >
                  {c.label}
                  {sort.key === c.k ? (sort.dir === "asc" ? " ↑" : " ↓") : ""}
                </button>
              ))}
            </div>

            <div style={{ overflow: "auto", flex: 1 }}>
              {rows.map((s) => (
                <div
                  key={s.sample_id}
                  onClick={() => navigate(`/stimuli/${encodeURIComponent(s.sample_id)}`)}
                  style={{ ...bodyRow, gridTemplateColumns: gridCols }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = WF.panelAlt)}
                  onMouseLeave={(e) => (e.currentTarget.style.background = WF.panel)}
                >
                  <div style={{ color: WF.accent }}>{s.sample_id}</div>
                  <div>{s.spec_index ?? "?"}</div>
                  <div style={{ overflow: "hidden" }}>
                    {Object.keys(s.spec.demands).length === 0 ? (
                      <Pill tone="ghost">control</Pill>
                    ) : (
                      <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
                        {Object.entries(s.spec.demands).map(([name, level]) => (
                          <Pill key={name}>{`${name}: ${level}`}</Pill>
                        ))}
                      </div>
                    )}
                  </div>
                  {paramCols.map((k) => (
                    <div key={k} style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                      {formatParam(s.spec.params[k])}
                    </div>
                  ))}
                  <div>
                    <Pill tone="ghost">{s.modality}</Pill>
                  </div>
                  <div style={{ fontWeight: 600 }}>{String(s.target)}</div>
                  <div style={{ color: WF.textMute }}>{s.validators_ran.length} ✓</div>
                </div>
              ))}
            </div>

            <div style={footerBar}>
              <span>{rows.length} rows</span>
              <span>· {specCount} specs</span>
              <span style={{ flex: 1 }} />
              <span>
                sort: {sort.key} {sort.dir === "asc" ? "↑" : "↓"}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function formatParam(v: unknown): string {
  if (v === undefined) return "";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

function SummaryStrip({
  data,
  onViewManifest,
}: {
  data: DatasetData;
  onViewManifest: () => void;
}) {
  const m = data.manifest;
  const items: [string, string | number][] = [
    ["name", m.name],
    ["n_stimuli", m.n_stimuli ?? data.stimuli.length],
    ["n_reps", m.n_reps ?? "?"],
    ["seed", m.global_seed ?? "?"],
    ["lib", "v" + (m.library_version ?? "?")],
    ["written", (m.timestamp ?? "").slice(0, 10)],
  ];
  return (
    <div style={summaryStrip}>
      {items.map(([k, v]) => (
        <div key={k} style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          <span
            style={{
              color: WF.textDim,
              fontSize: 9,
              textTransform: "uppercase",
              letterSpacing: 0.6,
            }}
          >
            {k}
          </span>
          <span style={{ color: WF.text }}>{v}</span>
        </div>
      ))}
      <span style={{ flex: 1 }} />
      <button onClick={onViewManifest} style={linkBtn}>
        view full manifest →
      </button>
    </div>
  );
}

function FilterBar({
  filters,
  setFilters,
  demandOptions,
  shown,
  total,
}: {
  filters: Filters;
  setFilters: (f: Filters) => void;
  demandOptions: Map<string, number>;
  shown: number;
  total: number;
}) {
  return (
    <div style={filterBar}>
      <input
        style={{ ...inputStyle, width: 260 }}
        placeholder="search messages, target, sample_id…"
        value={filters.search}
        onChange={(e) => setFilters({ ...filters, search: e.target.value })}
      />
      <DemandPicker
        value={filters.demands}
        options={demandOptions}
        onChange={(demands) => setFilters({ ...filters, demands })}
      />
      <input
        style={{ ...inputStyle, width: 150 }}
        placeholder="params  key=value"
        value={filters.paramQuery}
        onChange={(e) => setFilters({ ...filters, paramQuery: e.target.value })}
      />
      <input
        style={{ ...inputStyle, width: 60 }}
        placeholder="id ≥"
        value={filters.idFrom}
        onChange={(e) => setFilters({ ...filters, idFrom: e.target.value })}
      />
      <input
        style={{ ...inputStyle, width: 60 }}
        placeholder="id ≤"
        value={filters.idTo}
        onChange={(e) => setFilters({ ...filters, idTo: e.target.value })}
      />
      <span style={{ flex: 1 }} />
      <span style={{ fontFamily: WF.mono, fontSize: 10, color: WF.textMute }}>
        showing {shown} / {total}
      </span>
    </div>
  );
}

function DemandPicker({
  value,
  options,
  onChange,
}: {
  value: Filters["demands"];
  options: Map<string, number>;
  onChange: (v: Filters["demands"]) => void;
}) {
  const [open, setOpen] = useState(false);
  const summary =
    value.values.length === 0 ? "any" : `${value.mode} (${value.values.length})`;
  const entries = [...options.entries()].sort((a, b) => a[0].localeCompare(b[0]));

  function toggleValue(d: string) {
    const set = new Set(value.values);
    if (set.has(d)) set.delete(d);
    else set.add(d);
    onChange({ ...value, values: [...set] });
  }

  return (
    <div style={{ position: "relative" }}>
      <button style={{ ...inputStyle, width: 150, cursor: "pointer" }} onClick={() => setOpen((o) => !o)}>
        demands: {summary} ▾
      </button>
      {open && (
        <div style={popover}>
          <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
            {(["any", "all", "none"] as const).map((m) => (
              <button
                key={m}
                onClick={() => onChange({ ...value, mode: m })}
                style={{
                  ...modeBtn,
                  background: value.mode === m ? WF.accentBg : "transparent",
                  color: value.mode === m ? WF.accent : WF.textMute,
                  borderColor: value.mode === m ? WF.accentBorder : WF.border,
                }}
              >
                {m}
              </button>
            ))}
          </div>
          <div style={{ maxHeight: 220, overflow: "auto" }}>
            {entries.length === 0 && (
              <div style={{ color: WF.textDim, fontSize: 11, fontFamily: WF.mono }}>
                no demands in dataset
              </div>
            )}
            {entries.map(([d, c]) => (
              <label key={d} style={checkRow}>
                <input
                  type="checkbox"
                  checked={value.values.includes(d)}
                  onChange={() => toggleValue(d)}
                />
                <span style={{ flex: 1 }}>{d}</span>
                <span style={{ color: WF.textDim }}>{c}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TabBar({
  tabs,
  active,
  onSelect,
}: {
  tabs: { k: string; label: string; hint: string | null }[];
  active: string;
  onSelect: (k: string) => void;
}) {
  return (
    <div style={tabBar}>
      {tabs.map((t) => {
        const on = active === t.k;
        return (
          <button
            key={t.k}
            onClick={() => onSelect(t.k)}
            style={{
              padding: "10px 14px",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontFamily: WF.sans,
              fontSize: 12,
              fontWeight: on ? 600 : 500,
              color: on ? WF.text : WF.textMute,
              borderBottom: on ? `2px solid ${WF.accent}` : "2px solid transparent",
              marginBottom: -1,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            {t.label}
            {t.hint && <Pill tone={on ? "accent" : "ghost"}>{t.hint}</Pill>}
          </button>
        );
      })}
      <span style={{ flex: 1 }} />
    </div>
  );
}

const pageStyle: React.CSSProperties = {
  width: "100%",
  height: "100%",
  display: "flex",
  flexDirection: "column",
  background: WF.bg,
  color: WF.text,
  fontFamily: WF.sans,
  fontSize: 12,
};

const tabBar: React.CSSProperties = {
  display: "flex",
  gap: 0,
  padding: "0 14px",
  borderBottom: `1px solid ${WF.border}`,
  background: WF.panel,
  alignItems: "stretch",
};

const summaryStrip: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 22,
  padding: "10px 16px",
  background: WF.panel,
  borderBottom: `1px solid ${WF.divider}`,
  fontFamily: WF.mono,
  fontSize: 11,
};

const filterBar: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  flexWrap: "wrap",
  padding: "8px 16px",
  background: WF.bg,
  borderBottom: `1px solid ${WF.divider}`,
};

const inputStyle: React.CSSProperties = {
  height: 26,
  padding: "0 8px",
  background: WF.panel,
  border: `1px solid ${WF.border}`,
  borderRadius: 3,
  color: WF.text,
  fontFamily: WF.mono,
  fontSize: 11,
  textAlign: "left",
};

const headerRow: React.CSSProperties = {
  display: "grid",
  fontFamily: WF.mono,
  fontSize: 10,
  color: WF.textMute,
  background: WF.panelAlt,
  borderBottom: `1px solid ${WF.border}`,
  textTransform: "uppercase",
  letterSpacing: 0.5,
  padding: "7px 14px",
  gap: 14,
  flexShrink: 0,
};

const sortHeaderBtn: React.CSSProperties = {
  background: "transparent",
  border: "none",
  padding: 0,
  textAlign: "left",
  cursor: "pointer",
  font: "inherit",
  color: WF.textMute,
  textTransform: "uppercase",
  letterSpacing: 0.5,
};

const bodyRow: React.CSSProperties = {
  display: "grid",
  gap: 14,
  padding: "7px 14px",
  fontFamily: WF.mono,
  fontSize: 11,
  borderBottom: `1px solid ${WF.divider}`,
  background: WF.panel,
  cursor: "pointer",
};

const footerBar: React.CSSProperties = {
  padding: "6px 14px",
  borderTop: `1px solid ${WF.border}`,
  background: WF.panelAlt,
  fontFamily: WF.mono,
  fontSize: 10,
  color: WF.textMute,
  display: "flex",
  gap: 14,
  flexShrink: 0,
};

const linkBtn: React.CSSProperties = {
  background: "transparent",
  border: "none",
  cursor: "pointer",
  color: WF.accent,
  fontFamily: WF.sans,
  fontSize: 11,
  padding: 0,
  alignSelf: "flex-end",
};

const popover: React.CSSProperties = {
  position: "absolute",
  top: 30,
  left: 0,
  zIndex: 10,
  width: 240,
  background: WF.panel,
  border: `1px solid ${WF.border}`,
  borderRadius: 3,
  padding: 8,
  fontFamily: WF.mono,
  fontSize: 11,
  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
};

const modeBtn: React.CSSProperties = {
  flex: 1,
  padding: "3px 0",
  borderRadius: 3,
  border: `1px solid ${WF.border}`,
  cursor: "pointer",
  fontFamily: WF.mono,
  fontSize: 10,
};

const checkRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 6,
  padding: "3px 2px",
  cursor: "pointer",
};
