export interface Spec {
  demands: Record<string, number>;
  params: Record<string, unknown>;
}

export type ContentItem =
  | { type: "text"; text: string }
  | { type: "image"; image: string; detail?: string }
  | { type: "audio"; audio: string; format: string }
  | { type: "video"; video: string; format: string }
  | {
      type: "document";
      document: string;
      filename?: string | null;
      mime_type?: string | null;
    };

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentItem[];
}

export interface StimulusRecord {
  sample_id: string;
  spec: Spec;
  functional: Spec | null;
  messages: Message[];
  target: unknown;
  metadata: Record<string, unknown>;
  validators_ran: string[];
  // Derived server-side by viewer/dataset_io.py:
  spec_index: number | null;
  modality: string;
}

export interface Manifest {
  name: string;
  library_version?: string;
  timestamp?: string;
  global_seed?: number;
  n_reps?: number;
  n_stimuli?: number;
  functional: Spec | null;
  specs: Spec[];
}

export type DemandMode = "any" | "all" | "none";

export interface Filters {
  search: string;
  demands: { mode: DemandMode; values: string[] };
  paramQuery: string;
  idFrom: string;
  idTo: string;
}

export interface Sort {
  key: string;
  dir: "asc" | "desc";
}
