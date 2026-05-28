import { useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import { fetchManifest, fetchStimuli } from "./api";
import { WF } from "./tokens";
import type { Manifest, StimulusRecord } from "./types";
import { DatasetPage } from "./pages/DatasetPage";
import { StimulusPage } from "./pages/StimulusPage";

export interface DatasetData {
  manifest: Manifest;
  stimuli: StimulusRecord[];
}

export function App() {
  const [data, setData] = useState<DatasetData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchManifest(), fetchStimuli()])
      .then(([manifest, stimuli]) => setData({ manifest, stimuli }))
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <Centered>Failed to load dataset: {error}</Centered>;
  if (!data) return <Centered>Loading dataset…</Centered>;

  return (
    <Routes>
      <Route path="/" element={<DatasetPage data={data} />} />
      <Route path="/stimuli/:sampleId" element={<StimulusPage data={data} />} />
      <Route path="*" element={<DatasetPage data={data} />} />
    </Routes>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: WF.mono,
        fontSize: 13,
        color: WF.textMute,
      }}
    >
      {children}
    </div>
  );
}
