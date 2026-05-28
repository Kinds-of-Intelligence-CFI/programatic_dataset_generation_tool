import { useMemo } from "react";
import { Pill, PanelHead } from "../components/primitives";
import { WF } from "../tokens";
import type { DatasetData } from "../App";
import { distinctSpecs } from "../lib/derive";

function paramValue(v: unknown): string {
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

export function ManifestTab({ data }: { data: DatasetData }) {
  const { manifest, stimuli } = data;
  const specs = useMemo(() => distinctSpecs(manifest, stimuli), [manifest, stimuli]);

  const folderTree = useMemo(() => buildFolderTree(data), [data]);

  const rows: [string, React.ReactNode][] = [
    ["name", manifest.name],
    [
      "n_stimuli",
      <span key="n">
        <span style={{ color: WF.text }}>{manifest.n_stimuli ?? stimuli.length}</span>
        <span style={{ color: WF.textDim }}>
          {" "}
          &nbsp;= {manifest.specs.length} specs × {manifest.n_reps ?? "?"} reps
        </span>
      </span>,
    ],
    ["global_seed", String(manifest.global_seed ?? "")],
    ["timestamp", manifest.timestamp ?? ""],
    ["library_version", <Pill key="lib">v{manifest.library_version ?? "?"}</Pill>],
    [
      "functional",
      manifest.functional === null ? (
        <span style={{ color: WF.textDim }}>null</span>
      ) : (
        <Pill tone="accent">set</Pill>
      ),
    ],
  ];

  return (
    <div style={{ padding: 20, overflow: "auto", flex: 1, minHeight: 0, background: WF.bg }}>
      <div
        style={{
          maxWidth: 920,
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        <div style={{ background: WF.panel, border: `1px solid ${WF.border}` }}>
          <PanelHead>manifest.json</PanelHead>
          <div
            style={{
              padding: "14px 16px",
              display: "grid",
              gridTemplateColumns: "160px 1fr",
              fontFamily: WF.mono,
              fontSize: 12,
              rowGap: 8,
              columnGap: 16,
            }}
          >
            {rows.map(([k, v]) => (
              <Cell key={k} label={k} value={v} />
            ))}
          </div>
        </div>

        <div style={{ background: WF.panel, border: `1px solid ${WF.border}` }}>
          <PanelHead
            right={
              <Pill tone="ghost">
                {manifest.specs.length} specs · {manifest.n_reps ?? "?"} reps each
              </Pill>
            }
          >
            specs[]
          </PanelHead>
          <SpecGridHeader />
          {specs.map((s) => (
            <div
              key={s.index}
              style={{
                display: "grid",
                gridTemplateColumns: "40px 1fr 1.5fr 70px 50px",
                padding: "8px 14px",
                columnGap: 14,
                alignItems: "center",
                fontFamily: WF.mono,
                fontSize: 11,
                borderBottom: `1px solid ${WF.divider}`,
              }}
            >
              <span style={{ color: WF.accent }}>{s.index}</span>
              <span>
                {Object.keys(s.spec.demands).length ? (
                  Object.entries(s.spec.demands).map(([name, level]) => (
                    <Pill key={name} style={{ marginRight: 4 }}>
                      {`${name}: ${level}`}
                    </Pill>
                  ))
                ) : (
                  <Pill tone="ghost">control</Pill>
                )}
              </span>
              <span style={{ color: WF.textMute, fontSize: 10 }}>
                {Object.entries(s.spec.params).map(([k, v]) => (
                  <span key={k} style={{ marginRight: 10 }}>
                    <span style={{ color: WF.textDim }}>{k}=</span>
                    <span style={{ color: WF.text }}>{paramValue(v)}</span>
                  </span>
                ))}
              </span>
              <span style={{ color: WF.textMute }}>{manifest.n_reps ?? "?"}</span>
              <span style={{ fontWeight: 600 }}>{s.count}</span>
            </div>
          ))}
        </div>

        <div style={{ background: WF.panel, border: `1px solid ${WF.border}` }}>
          <PanelHead>folder layout</PanelHead>
          <pre
            style={{
              margin: 0,
              padding: "12px 14px",
              fontFamily: WF.mono,
              fontSize: 11,
              lineHeight: 1.6,
              color: WF.text,
            }}
          >
            {folderTree}
          </pre>
        </div>
      </div>
    </div>
  );
}

function Cell({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <>
      <span style={{ color: WF.textMute }}>{label}</span>
      <span style={{ color: WF.text, wordBreak: "break-all" }}>{value}</span>
    </>
  );
}

function SpecGridHeader() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "40px 1fr 1.5fr 70px 50px",
        fontFamily: WF.mono,
        fontSize: 10,
        background: WF.panelAlt,
        borderBottom: `1px solid ${WF.border}`,
        textTransform: "uppercase",
        letterSpacing: 0.5,
        color: WF.textMute,
        padding: "6px 14px",
        columnGap: 14,
      }}
    >
      <span>#</span>
      <span>demands</span>
      <span>params</span>
      <span>n_reps</span>
      <span>total</span>
    </div>
  );
}

// Tree built from data we actually have: the manifest name, the stimulus count,
// and the asset references that appear in the stimuli.
function buildFolderTree(data: DatasetData): string {
  let inline = 0;
  let files = 0;
  for (const s of data.stimuli) {
    for (const m of s.messages) {
      if (typeof m.content === "string") continue;
      for (const c of m.content) {
        const ref =
          "image" in c
            ? c.image
            : "audio" in c
              ? c.audio
              : "video" in c
                ? c.video
                : "document" in c
                  ? c.document
                  : "";
        if (ref.startsWith("assets/inline/")) inline += 1;
        else if (ref.startsWith("assets/files/")) files += 1;
      }
    }
  }
  const lines = [
    `${data.manifest.name}/`,
    `  manifest.json`,
    `  stimuli.jsonl         (${data.stimuli.length} lines)`,
  ];
  if (inline || files) {
    lines.push(`  assets/`);
    if (inline) lines.push(`    inline/             (${inline} files)`);
    if (files) lines.push(`    files/              (${files} files)`);
  }
  return lines.join("\n");
}
