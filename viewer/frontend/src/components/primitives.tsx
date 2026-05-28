import type { CSSProperties, ReactNode } from "react";
import { WF } from "../tokens";

type Tone = "neutral" | "accent" | "ok" | "warn" | "ghost";

const PILL_TONES: Record<Tone, { bg: string; fg: string; bd: string }> = {
  neutral: { bg: WF.greybox, fg: WF.text, bd: WF.greyboxBorder },
  accent: { bg: WF.accentBg, fg: WF.accent, bd: WF.accentBorder },
  ok: { bg: WF.okBg, fg: WF.ok, bd: "transparent" },
  warn: { bg: WF.warnBg, fg: WF.warn, bd: "transparent" },
  ghost: { bg: "transparent", fg: WF.textMute, bd: WF.border },
};

export function Pill({
  children,
  tone = "neutral",
  style,
}: {
  children: ReactNode;
  tone?: Tone;
  style?: CSSProperties;
}) {
  const t = PILL_TONES[tone];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "1px 6px",
        borderRadius: 3,
        background: t.bg,
        color: t.fg,
        border: `1px solid ${t.bd}`,
        fontFamily: WF.mono,
        fontSize: 10,
        lineHeight: 1.5,
        whiteSpace: "nowrap",
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function PanelHead({
  children,
  right,
  style,
}: {
  children: ReactNode;
  right?: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 12px",
        background: WF.panelAlt,
        borderBottom: `1px solid ${WF.divider}`,
        fontFamily: WF.mono,
        fontSize: 10,
        textTransform: "uppercase",
        letterSpacing: 0.8,
        color: WF.textMute,
        ...style,
      }}
    >
      <span>{children}</span>
      <span style={{ flex: 1 }} />
      {right}
    </div>
  );
}

export function GhostButton({
  children,
  onClick,
  disabled,
  style,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  style?: CSSProperties;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: "transparent",
        border: `1px solid ${WF.border}`,
        cursor: disabled ? "default" : "pointer",
        padding: "3px 10px",
        borderRadius: 3,
        fontFamily: WF.mono,
        fontSize: 11,
        color: disabled ? WF.textDim : WF.textMute,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

// Window chrome bar shared by both pages.
export function Chrome({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 14px",
        background: WF.panelAlt,
        borderBottom: `1px solid ${WF.border}`,
        fontFamily: WF.mono,
        fontSize: 11,
        color: WF.textMute,
        flexShrink: 0,
      }}
    >
      <div style={{ display: "flex", gap: 5 }}>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            style={{
              width: 9,
              height: 9,
              borderRadius: 9,
              background: WF.greybox,
              border: `1px solid ${WF.greyboxBorder}`,
            }}
          />
        ))}
      </div>
      {children}
    </div>
  );
}
