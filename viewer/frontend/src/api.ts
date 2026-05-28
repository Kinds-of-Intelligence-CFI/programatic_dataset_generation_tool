import type { Manifest, StimulusRecord } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url} -> ${resp.status}`);
  return (await resp.json()) as T;
}

export function fetchManifest(): Promise<Manifest> {
  return getJson<Manifest>("/api/manifest");
}

export function fetchStimuli(): Promise<StimulusRecord[]> {
  return getJson<StimulusRecord[]>("/api/stimuli");
}
