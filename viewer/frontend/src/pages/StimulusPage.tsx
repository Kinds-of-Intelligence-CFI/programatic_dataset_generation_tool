import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Chrome, GhostButton, Pill, PanelHead } from "../components/primitives";
import { WF, assetUrl } from "../tokens";
import type { DatasetData } from "../App";
import type { ContentItem, Message, StimulusRecord } from "../types";
import { varyingParamKeys } from "../lib/derive";

type StimTab = "rendered" | "spec" | "validators" | "raw";

function demandsText(demands: Record<string, number>): string {
  return Object.entries(demands)
    .map(([name, level]) => `${name}: ${level}`)
    .join(", ");
}

function firstDemandLabel(demands: Record<string, number>): string {
  const entries = Object.entries(demands);
  if (entries.length === 0) return "";
  const [name, level] = entries[0];
  return `${name}: ${level}`;
}

export function StimulusPage({ data }: { data: DatasetData }) {
  const { manifest, stimuli } = data;
  const { sampleId } = useParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState<StimTab>("rendered");
  const [sidebarFilter, setSidebarFilter] = useState("");

  const idx = stimuli.findIndex((s) => s.sample_id === sampleId);
  const stim = idx >= 0 ? stimuli[idx] : undefined;
  const paramKeys = useMemo(() => varyingParamKeys(manifest.specs), [manifest.specs]);

  const open = (id: string | undefined) => {
    if (id !== undefined) navigate(`/stimuli/${encodeURIComponent(id)}`);
  };

  const sidebarRows = stimuli.filter((s) => {
    const q = sidebarFilter.trim().toLowerCase();
    if (!q) return true;
    return (
      s.sample_id.toLowerCase().includes(q) ||
      demandsText(s.spec.demands).toLowerCase().includes(q) ||
      JSON.stringify(s.spec.params).toLowerCase().includes(q)
    );
  });

  return (
    <div style={pageStyle}>
      <Chrome>
        <button onClick={() => navigate("/")} style={backBtn}>
          ← Dataset
        </button>
        <span style={{ color: WF.textDim }}>·</span>
        <span>{manifest.name}/</span>
        <span style={{ color: WF.textDim }}>·</span>
        <span style={{ color: WF.text }}>#{sampleId}</span>
        <span style={{ flex: 1 }} />
      </Chrome>

      <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
        <Sidebar
          rows={sidebarRows}
          total={stimuli.length}
          activeId={stim?.sample_id}
          paramKeys={paramKeys}
          filter={sidebarFilter}
          setFilter={setSidebarFilter}
          onOpen={open}
        />

        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          {!stim ? (
            <div style={{ padding: 24, fontFamily: WF.mono, color: WF.textMute }}>
              No stimulus with sample_id {JSON.stringify(sampleId)}.
            </div>
          ) : (
            <>
              <Header
                stim={stim}
                onPrev={() => open(stimuli[Math.max(0, idx - 1)]?.sample_id)}
                onNext={() => open(stimuli[Math.min(stimuli.length - 1, idx + 1)]?.sample_id)}
                atStart={idx <= 0}
                atEnd={idx >= stimuli.length - 1}
              />
              <TabRow
                tab={tab}
                setTab={setTab}
                validatorCount={stim.validators_ran.length}
              />
              <div style={{ flex: 1, minHeight: 0, overflow: "auto", background: WF.bg, padding: 16 }}>
                {tab === "rendered" && <Rendered stim={stim} />}
                {tab === "spec" && <SpecMeta stim={stim} />}
                {tab === "validators" && <Validators stim={stim} />}
                {tab === "raw" && <Raw stim={stim} />}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Sidebar({
  rows,
  total,
  activeId,
  paramKeys,
  filter,
  setFilter,
  onOpen,
}: {
  rows: StimulusRecord[];
  total: number;
  activeId: string | undefined;
  paramKeys: string[];
  filter: string;
  setFilter: (v: string) => void;
  onOpen: (id: string) => void;
}) {
  return (
    <div
      style={{
        width: 230,
        flexShrink: 0,
        borderRight: `1px solid ${WF.border}`,
        background: WF.panel,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{ padding: "8px 10px", borderBottom: `1px solid ${WF.divider}` }}>
        <input
          style={{
            width: "100%",
            boxSizing: "border-box",
            height: 26,
            padding: "0 8px",
            background: WF.panel,
            border: `1px solid ${WF.border}`,
            borderRadius: 3,
            fontFamily: WF.mono,
            fontSize: 11,
          }}
          placeholder="filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <div style={{ overflow: "auto", flex: 1 }}>
        {rows.map((s) => {
          const on = s.sample_id === activeId;
          const sub = paramKeys
            .map((k) => formatParam(s.spec.params[k]))
            .filter(Boolean)
            .join(" · ");
          return (
            <div
              key={s.sample_id}
              onClick={() => onOpen(s.sample_id)}
              style={{
                padding: "7px 10px",
                borderBottom: `1px solid ${WF.divider}`,
                background: on ? WF.accentBg : "transparent",
                borderLeft: on ? `2px solid ${WF.accent}` : "2px solid transparent",
                fontFamily: WF.mono,
                fontSize: 11,
                display: "flex",
                alignItems: "center",
                gap: 8,
                cursor: "pointer",
                color: on ? WF.text : WF.textMute,
              }}
            >
              <span style={{ width: 22, color: on ? WF.accent : WF.textDim }}>
                #{s.sample_id}
              </span>
              <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {Object.keys(s.spec.demands).length === 0 ? (
                    <span style={{ color: WF.textDim }}>(control)</span>
                  ) : (
                    firstDemandLabel(s.spec.demands)
                  )}
                </span>
                {sub && <span style={{ fontSize: 9, color: WF.textDim }}>{sub}</span>}
              </div>
              <span style={{ fontWeight: 600, color: on ? WF.text : WF.textMute }}>
                {String(s.target)}
              </span>
            </div>
          );
        })}
      </div>
      <div
        style={{
          padding: "5px 10px",
          borderTop: `1px solid ${WF.border}`,
          background: WF.panelAlt,
          fontFamily: WF.mono,
          fontSize: 10,
          color: WF.textMute,
        }}
      >
        {rows.length === total ? `${total} stimuli` : `${rows.length} / ${total} stimuli`}
      </div>
    </div>
  );
}

function Header({
  stim,
  onPrev,
  onNext,
  atStart,
  atEnd,
}: {
  stim: StimulusRecord;
  onPrev: () => void;
  onNext: () => void;
  atStart: boolean;
  atEnd: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 16px",
        background: WF.panel,
        borderBottom: `1px solid ${WF.border}`,
        flexWrap: "wrap",
      }}
    >
      <span style={{ fontFamily: WF.mono, fontSize: 13, fontWeight: 700 }}>
        sample_id = "{stim.sample_id}"
      </span>
      <span style={{ color: WF.textDim }}>·</span>
      <span style={{ fontFamily: WF.mono, fontSize: 11, color: WF.textMute }}>
        spec {stim.spec_index ?? "?"}
      </span>
      <span style={{ color: WF.textDim }}>·</span>
      {Object.keys(stim.spec.demands).length === 0 ? (
        <Pill tone="ghost">control</Pill>
      ) : (
        Object.entries(stim.spec.demands).map(([name, level]) => (
          <Pill key={name}>{`${name}: ${level}`}</Pill>
        ))
      )}
      <Pill tone="ghost">{stim.modality}</Pill>
      <span style={{ flex: 1 }} />
      <GhostButton onClick={onPrev} disabled={atStart}>
        ↑ prev
      </GhostButton>
      <GhostButton onClick={onNext} disabled={atEnd}>
        next ↓
      </GhostButton>
    </div>
  );
}

function TabRow({
  tab,
  setTab,
  validatorCount,
}: {
  tab: StimTab;
  setTab: (t: StimTab) => void;
  validatorCount: number;
}) {
  const tabs: { k: StimTab; label: string; hint?: string }[] = [
    { k: "rendered", label: "Rendered" },
    { k: "spec", label: "Spec & metadata" },
    { k: "validators", label: "Validators", hint: `${validatorCount}` },
    { k: "raw", label: "Raw JSON" },
  ];
  return (
    <div
      style={{
        display: "flex",
        gap: 0,
        padding: "0 16px",
        borderBottom: `1px solid ${WF.border}`,
        background: WF.panel,
      }}
    >
      {tabs.map((t) => {
        const on = tab === t.k;
        return (
          <button
            key={t.k}
            onClick={() => setTab(t.k)}
            style={{
              padding: "9px 12px",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontFamily: WF.mono,
              fontSize: 11,
              color: on ? WF.text : WF.textMute,
              borderBottom: on ? `2px solid ${WF.accent}` : "2px solid transparent",
              marginBottom: -1,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            {t.label}
            {t.hint && <Pill tone={on ? "ok" : "ghost"}>{t.hint}</Pill>}
          </button>
        );
      })}
    </div>
  );
}

// ---- Rendered tab ----------------------------------------------------------

function Rendered({ stim }: { stim: StimulusRecord }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {stim.messages.map((m, i) => (
        <MessageBlock key={i} message={m} index={i} total={stim.messages.length} />
      ))}
      <div
        style={{
          padding: "10px 14px",
          background: WF.panelAlt,
          border: `1px solid ${WF.divider}`,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <span
          style={{
            fontFamily: WF.mono,
            fontSize: 10,
            color: WF.textMute,
            textTransform: "uppercase",
            letterSpacing: 0.6,
          }}
        >
          target
        </span>
        <span
          style={{
            fontFamily: WF.mono,
            fontSize: 15,
            fontWeight: 700,
            padding: "3px 12px",
            background: WF.accentBg,
            color: WF.accent,
            borderRadius: 3,
          }}
        >
          {String(stim.target)}
        </span>
      </div>
    </div>
  );
}

function MessageBlock({
  message,
  index,
  total,
}: {
  message: Message;
  index: number;
  total: number;
}) {
  const items: ContentItem[] =
    typeof message.content === "string"
      ? [{ type: "text", text: message.content }]
      : message.content;
  return (
    <div style={{ background: WF.panel, border: `1px solid ${WF.divider}` }}>
      <PanelHead right={<Pill tone="ghost">{`message ${index + 1} / ${total}`}</Pill>}>
        {message.role}
      </PanelHead>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
        {items.map((c, i) => (
          <div key={i}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
              <Pill tone="accent">{message.role}</Pill>
              <span style={{ fontFamily: WF.mono, fontSize: 10, color: WF.textDim }}>
                content[{i}] · {c.type}
              </span>
            </div>
            <ContentBody item={c} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ContentBody({ item }: { item: ContentItem }) {
  if (item.type === "text") {
    return (
      <pre
        style={{
          margin: 0,
          background: WF.panelAlt,
          border: `1px solid ${WF.divider}`,
          padding: 10,
          fontFamily: WF.mono,
          fontSize: 12,
          color: WF.text,
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {item.text}
      </pre>
    );
  }
  if (item.type === "image") {
    return (
      <img
        src={assetUrl(item.image)}
        alt="stimulus"
        style={{ maxWidth: "100%", border: `1px solid ${WF.divider}` }}
      />
    );
  }
  if (item.type === "audio") {
    return <audio controls src={assetUrl(item.audio)} style={{ width: "100%" }} />;
  }
  if (item.type === "video") {
    return <video controls src={assetUrl(item.video)} style={{ maxWidth: "100%" }} />;
  }
  return (
    <a
      href={assetUrl(item.document)}
      target="_blank"
      rel="noreferrer"
      style={{ fontFamily: WF.mono, fontSize: 12, color: WF.accent }}
    >
      {item.filename ?? item.document}
    </a>
  );
}

// ---- Spec & metadata tab ---------------------------------------------------

function SpecMeta({ stim }: { stim: StimulusRecord }) {
  const specRows: [string, string][] = [
    [
      "demands",
      Object.keys(stim.spec.demands).length
        ? demandsText(stim.spec.demands)
        : "(empty — control)",
    ],
    ...Object.entries(stim.spec.params).map(
      ([k, v]) => [`params.${k}`, formatParam(v)] as [string, string],
    ),
  ];
  const metaRows: [string, string][] = Object.entries(stim.metadata).map(
    ([k, v]) => [k, formatParam(v)],
  );
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      <KeyValueCard title="spec" rows={specRows} />
      <KeyValueCard title="metadata" rows={metaRows} />
    </div>
  );
}

function KeyValueCard({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div style={{ background: WF.panel, border: `1px solid ${WF.divider}` }}>
      <PanelHead right={<Pill tone="ghost">{rows.length} keys</Pill>}>{title}</PanelHead>
      {rows.map(([k, v]) => (
        <div
          key={k}
          style={{
            display: "grid",
            gridTemplateColumns: "180px 1fr",
            gap: 12,
            padding: "6px 14px",
            borderBottom: `1px solid ${WF.divider}`,
            fontFamily: WF.mono,
            fontSize: 12,
          }}
        >
          <span style={{ color: WF.textMute }}>{k}</span>
          <span style={{ color: WF.text, wordBreak: "break-all" }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

// ---- Validators tab --------------------------------------------------------

function Validators({ stim }: { stim: StimulusRecord }) {
  const n = stim.validators_ran.length;
  return (
    <div style={{ background: WF.panel, border: `1px solid ${WF.divider}`, maxWidth: 760 }}>
      <PanelHead right={<Pill tone="ok">{`${n} / ${n} passed`}</Pill>}>validators_ran</PanelHead>
      <div style={{ fontFamily: WF.mono, fontSize: 12 }}>
        {n === 0 && (
          <div style={{ padding: "10px 14px", color: WF.warn }}>
            No validators ran on this sample.
          </div>
        )}
        {stim.validators_ran.map((name) => (
          <div
            key={name}
            style={{
              display: "flex",
              gap: 10,
              padding: "8px 14px",
              alignItems: "center",
              borderBottom: `1px solid ${WF.divider}`,
            }}
          >
            <span style={{ color: WF.ok, fontWeight: 700 }}>✓</span>
            <span style={{ color: WF.text }}>{name}</span>
          </div>
        ))}
      </div>
      <div
        style={{
          padding: "10px 14px",
          fontFamily: WF.mono,
          fontSize: 11,
          color: WF.textMute,
          background: WF.panelAlt,
          borderTop: `1px solid ${WF.divider}`,
        }}
      >
        A validator ran on this sample iff its demand is "*" or appears in the spec's
        demands. Failures abort generation, so every validator listed here passed.
      </div>
    </div>
  );
}

// ---- Raw JSON tab ----------------------------------------------------------

function Raw({ stim }: { stim: StimulusRecord }) {
  // Drop the two fields the viewer derives server-side so this matches the
  // line as written to stimuli.jsonl.
  const { spec_index: _si, modality: _m, ...record } = stim;
  void _si;
  void _m;
  return (
    <div style={{ background: WF.panel, border: `1px solid ${WF.divider}` }}>
      <PanelHead>stimuli.jsonl[{stim.sample_id}]</PanelHead>
      <pre
        style={{
          margin: 0,
          padding: 14,
          fontFamily: WF.mono,
          fontSize: 12,
          lineHeight: 1.55,
          color: WF.text,
          background: WF.panel,
          overflow: "auto",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {JSON.stringify(record, null, 2)}
      </pre>
    </div>
  );
}

function formatParam(v: unknown): string {
  if (v === undefined) return "";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
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

const backBtn: React.CSSProperties = {
  background: "transparent",
  border: "none",
  cursor: "pointer",
  padding: 0,
  color: WF.accent,
  fontFamily: WF.mono,
  fontSize: 11,
};
