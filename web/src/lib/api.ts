export type CorpusSummary = {
  document_count: number;
  chunk_count: number;
  public_document_count: number;
  restricted_document_count: number;
  principals: string[];
};

export type StatusResponse = {
  project_name: string;
  config_path: string;
  documents_path: string;
  models: {
    profile: string;
    chat_model: string;
    embedding_model: string;
  };
  permissions_enabled: boolean;
  default_principals: string[];
  corpus: CorpusSummary;
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
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
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

export function reloadRuntime() {
  return request<StatusResponse>("/api/reload", { method: "POST", body: "{}" });
}

export function ingestDocuments(documentsPath: string) {
  return request<IngestResponse>("/api/ingest", {
    method: "POST",
    body: JSON.stringify({ documents_path: documentsPath }),
  });
}

export function askQuestion(question: string, principals: string[]) {
  return request<AskResponse>("/api/ask", {
    method: "POST",
    body: JSON.stringify({ question, principals }),
  });
}
