export type CorpusSummary = {
  document_count: number;
  chunk_count: number;
  public_document_count: number;
  restricted_document_count: number;
  principals: string[];
};

export type SuggestedPrompt = {
  label: string;
  prompt: string;
};

export type ActiveModelSettings = {
  profile: string;
  source: "config" | "session";
  base_url: string;
  chat_model: string;
  embedding_model: string;
};

export type PendingModelChange = {
  profile: string;
  source: "config" | "session";
  base_url: string;
  chat_model: string;
  embedding_model: string;
};

export type ModelDiscoveryResponse = {
  base_url: string;
  reachable: boolean;
  models: string[];
  error: string | null;
};

export type BridgeHealthResponse = {
  base_url: string;
  reachable: boolean | null;
  model: string;
  error: string | null;
};

export type CorpusIngestSummary = {
  document_count: number;
  mode: string | null;
  ingest_model: string | null;
  ingest_fingerprint: string | null;
  chunking_strategy: string | null;
};

export type StatusResponse = {
  project_name: string;
  config_path: string;
  documents_path: string;
  documents_display_path: string;
  documents_source: "path" | "upload";
  local_models: {
    active: ActiveModelSettings;
    source: "config" | "session";
    pending_reindex: PendingModelChange | null;
    ollama: ModelDiscoveryResponse;
  };
  ingest: {
    mode: "local" | "bridge";
    bridge: BridgeHealthResponse;
    corpus: CorpusIngestSummary;
  };
  permissions_enabled: boolean;
  default_principals: string[];
  corpus: CorpusSummary;
  suggested_prompts: SuggestedPrompt[];
};

export type IngestResponse = {
  status: StatusResponse;
  report: {
    processed: string[];
    skipped: string[];
    deleted: string[];
    errors: Record<string, string>;
  };
};

export type AnswerCitation = {
  chunk_id: string;
  citation: string;
  reason: string;
};

export type RetrievedChunk = {
  chunk_id: string;
  title: string;
  section_path: string;
  location_label: string;
  text: string;
  access_scope: string;
  access_principals: string[];
};

export type AskResponse = {
  question: string;
  answer: string;
  grounded: boolean;
  citations: AnswerCitation[];
  trace: Record<string, unknown>;
  retrieved_chunks: RetrievedChunk[];
  status: string;
  blocked_principals: string[];
};

export type ApplyModelSettingsResponse = {
  status: StatusResponse;
  applied: boolean;
  reindex_required: boolean;
  message: string;
};

export type ReindexModelSettingsResponse = {
  status: StatusResponse;
  report: IngestResponse["report"];
  message: string;
};

export type CancelModelSettingsResponse = {
  status: StatusResponse;
  message: string;
};

export type SuggestedPromptsResponse = {
  prompts: SuggestedPrompt[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    headers,
    ...init,
  });
  const payload = (await response.json().catch(() => ({}))) as { error?: string };
  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed with status ${response.status}`);
  }
  return payload as T;
}

export function fetchStatus() {
  return request<StatusResponse>("/api/status");
}

export function fetchBridgeHealth() {
  return request<BridgeHealthResponse>("/api/ingest/bridge-health");
}

export function reloadRuntime() {
  return request<StatusResponse>("/api/reload", { method: "POST", body: "{}" });
}

export function discoverModelSettings(baseUrl: string) {
  return request<ModelDiscoveryResponse>("/api/model-settings/discover", {
    method: "POST",
    body: JSON.stringify({ base_url: baseUrl }),
  });
}

export function applyModelSettings(baseUrl: string, chatModel: string, embeddingModel: string) {
  return request<ApplyModelSettingsResponse>("/api/model-settings/apply", {
    method: "POST",
    body: JSON.stringify({
      base_url: baseUrl,
      chat_model: chatModel,
      embedding_model: embeddingModel,
    }),
  });
}

export function reindexPendingModelSettings() {
  return request<ReindexModelSettingsResponse>("/api/model-settings/reindex", {
    method: "POST",
    body: "{}",
  });
}

export function cancelPendingModelSettings() {
  return request<CancelModelSettingsResponse>("/api/model-settings/cancel", {
    method: "POST",
    body: "{}",
  });
}

export function ingestDocuments(documentsPath: string) {
  return request<IngestResponse>("/api/ingest", {
    method: "POST",
    body: JSON.stringify({ documents_path: documentsPath }),
  });
}

export function uploadDocumentFolder(files: File[], folderName: string) {
  const formData = new FormData();
  for (const file of files) {
    const relativePath = file.webkitRelativePath || file.name;
    formData.append("files", file, relativePath);
  }
  formData.append("folder_name", folderName);
  return request<IngestResponse>("/api/ingest", {
    method: "POST",
    body: formData,
  });
}

export function askQuestion(question: string, principals: string[]) {
  return request<AskResponse>("/api/ask", {
    method: "POST",
    body: JSON.stringify({ question, principals }),
  });
}

export function fetchSuggestedPrompts(principals: string[]) {
  return request<SuggestedPromptsResponse>("/api/suggested-prompts", {
    method: "POST",
    body: JSON.stringify({ principals }),
  });
}
