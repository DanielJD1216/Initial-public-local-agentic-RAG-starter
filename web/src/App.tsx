import { startTransition, useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Database,
  FolderSearch,
  LoaderCircle,
  RefreshCw,
  Shield,
  Sparkles,
  Waypoints,
} from "lucide-react";

import {
  askQuestion,
  fetchStatus,
  ingestDocuments,
  reloadRuntime,
  type AskResponse,
  type IngestResponse,
  type StatusResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

const DEMO_PROMPTS = [
  {
    label: "Simple lookup",
    prompt: "What is the standard support first response time?",
  },
  {
    label: "Cross-document",
    prompt: "Who owns support escalations and when is the postmortem due for a customer-facing incident?",
  },
  {
    label: "Permission check",
    prompt: "When is the salary adjustment review window planned?",
  },
];

type AccessMode = "default" | "public" | "custom";

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [ingestReport, setIngestReport] = useState<IngestResponse["report"] | null>(null);
  const [question, setQuestion] = useState(DEMO_PROMPTS[0].prompt);
  const [documentsPath, setDocumentsPath] = useState("");
  const [accessMode, setAccessMode] = useState<AccessMode>("default");
  const [selectedPrincipals, setSelectedPrincipals] = useState<string[]>([]);
  const [extraPrincipals, setExtraPrincipals] = useState("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(true);
  const [isReloading, setIsReloading] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const deferredResult = useDeferredValue(result);

  useEffect(() => {
    void loadStatus();
  }, []);

  const effectivePrincipals = useMemo(() => {
    if (!status) {
      return [];
    }
    if (!status.permissions_enabled || accessMode === "default") {
      return status.default_principals;
    }
    if (accessMode === "public") {
      return [];
    }
    const extra = extraPrincipals
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    return Array.from(new Set([...selectedPrincipals, ...extra]));
  }, [accessMode, extraPrincipals, selectedPrincipals, status]);

  async function loadStatus() {
    setIsLoadingStatus(true);
    setErrorMessage(null);
    try {
      const payload = await fetchStatus();
      startTransition(() => {
        setStatus(payload);
        setDocumentsPath(payload.documents_path);
        setSelectedPrincipals(payload.default_principals.filter((value) => value !== "*"));
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to load runtime status.");
    } finally {
      setIsLoadingStatus(false);
    }
  }

  async function handleReload() {
    setIsReloading(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await reloadRuntime();
      startTransition(() => {
        setStatus(payload);
        setDocumentsPath(payload.documents_path);
        setStatusMessage("Runtime reloaded from the current config.");
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to reload the runtime.");
    } finally {
      setIsReloading(false);
    }
  }

  async function handleIndex() {
    if (!documentsPath.trim()) {
      setErrorMessage("Paste a folder path before indexing.");
      return;
    }
    setIsIndexing(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await ingestDocuments(documentsPath.trim());
      startTransition(() => {
        setStatus(payload.status);
        setDocumentsPath(payload.status.documents_path);
        setIngestReport(payload.report);
        setResult(null);
        setStatusMessage(
          `Indexed ${payload.report.processed.length} file(s), skipped ${payload.report.skipped.length}, deleted ${payload.report.deleted.length}.`,
        );
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Indexing failed.");
    } finally {
      setIsIndexing(false);
    }
  }

  async function handleAsk() {
    if (!question.trim()) {
      setErrorMessage("Ask a question before running retrieval.");
      return;
    }
    setIsAsking(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await askQuestion(question.trim(), effectivePrincipals);
      startTransition(() => {
        setResult(payload);
        setStatusMessage(payload.grounded ? "Grounded answer ready." : "Grounding failed cleanly and the answer was withheld.");
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Question failed.");
    } finally {
      setIsAsking(false);
    }
  }

  function togglePrincipal(principal: string) {
    setSelectedPrincipals((current) =>
      current.includes(principal) ? current.filter((value) => value !== principal) : [...current, principal],
    );
  }

  return (
    <div className="min-h-screen px-4 py-5 md:px-6 lg:px-8">
      <div className="mx-auto grid max-w-[1560px] gap-4 lg:grid-cols-[320px_minmax(0,1fr)_360px]">
        <aside className="w-full lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)] lg:self-start">
          <div className="flex h-full flex-col gap-4">
            <Card className="glass-panel">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <Badge variant="outline" className="border-cyan-400/30 bg-cyan-400/10 text-cyan-100">
                    Local-only
                  </Badge>
                  <Badge variant={status?.permissions_enabled ? "success" : "secondary"}>
                    {status?.permissions_enabled ? "Permissions on" : "Permissions off"}
                  </Badge>
                </div>
                <CardTitle className="text-base">Runtime shell</CardTitle>
                <CardDescription>
                  {status?.project_name ?? "Loading runtime configuration..."}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <MetricGrid status={status} loading={isLoadingStatus} />
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Documents folder</p>
                  <Input
                    value={documentsPath}
                    onChange={(event) => setDocumentsPath(event.target.value)}
                    placeholder="/absolute/path/to/your/docs"
                  />
                  <p className="text-xs text-muted-foreground">
                    Paste a real folder path here, then index it without editing YAML by hand.
                  </p>
                </div>
                <div className="flex flex-col gap-2 sm:flex-row lg:flex-col">
                  <Button onClick={handleIndex} disabled={isIndexing} className="w-full">
                    {isIndexing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <FolderSearch className="h-4 w-4" />}
                    Index folder
                  </Button>
                  <Button variant="secondary" onClick={handleReload} disabled={isReloading} className="w-full">
                    {isReloading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                    Reload runtime
                  </Button>
                </div>
                <div className="rounded-lg border border-border/80 bg-background/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Model profile</p>
                  <div className="mt-2 space-y-1 text-sm">
                    <p>{status?.models.profile ?? "small"}</p>
                    <p className="text-muted-foreground">{status?.models.chat_model ?? "..."}</p>
                    <p className="text-muted-foreground">{status?.models.embedding_model ?? "..."}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-panel">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-cyan-300" />
                  <CardTitle className="text-base">Access view</CardTitle>
                </div>
                <CardDescription>Switch the active principals before you run a query.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-3 gap-2 rounded-lg border border-border/70 bg-background/50 p-1">
                  <ModeButton label="Config" active={accessMode === "default"} onClick={() => setAccessMode("default")} />
                  <ModeButton label="Public" active={accessMode === "public"} onClick={() => setAccessMode("public")} />
                  <ModeButton label="Custom" active={accessMode === "custom"} onClick={() => setAccessMode("custom")} />
                </div>
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Discovered principals</p>
                  <div className="flex flex-wrap gap-2">
                    {(status?.corpus.principals ?? []).map((principal) => (
                      <button
                        key={principal}
                        type="button"
                        onClick={() => togglePrincipal(principal)}
                        className={cn(
                          "rounded-full border px-3 py-1 text-xs transition",
                          selectedPrincipals.includes(principal)
                            ? "border-cyan-400/40 bg-cyan-400/10 text-cyan-100"
                            : "border-border bg-background/50 text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {principal}
                      </button>
                    ))}
                    {!status?.corpus.principals.length && (
                      <Badge variant="outline">No restricted principals indexed yet</Badge>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Extra principals</p>
                  <Input
                    value={extraPrincipals}
                    onChange={(event) => setExtraPrincipals(event.target.value)}
                    placeholder="owners, finance"
                    disabled={accessMode !== "custom"}
                  />
                </div>
                <div className="rounded-lg border border-border/80 bg-background/40 p-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Effective principals</p>
                  <p className="mt-2 text-sm text-foreground">
                    {effectivePrincipals.length > 0 ? effectivePrincipals.join(", ") : "public-only"}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <div className="flex h-full flex-col gap-4">
            <Card className="glass-panel metric-sheen">
              <CardHeader className="pb-4">
                <div className="flex flex-wrap items-center gap-3">
                  <Badge variant="outline" className="border-border/80 bg-background/40 text-muted-foreground">
                    shadcn-style web UI
                  </Badge>
                  <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-100">
                    Codex-like flow
                  </Badge>
                </div>
                <CardTitle className="text-2xl md:text-3xl">Grounded answers, local models, real folder testing.</CardTitle>
                <CardDescription className="max-w-2xl text-sm leading-6">
                  Start with the sample prompts, then switch the documents folder to your real corpus and re-index from this screen.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  {DEMO_PROMPTS.map((item) => (
                    <Button key={item.label} variant="outline" size="sm" onClick={() => setQuestion(item.prompt)}>
                      <Sparkles className="h-3.5 w-3.5" />
                      {item.label}
                    </Button>
                  ))}
                </div>
                {statusMessage ? (
                  <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100">
                    {statusMessage}
                  </div>
                ) : null}
                {errorMessage ? (
                  <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-100">
                    {errorMessage}
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card className="glass-panel">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Waypoints className="h-4 w-4 text-cyan-300" />
                  <CardTitle className="text-base">Ask a grounded question</CardTitle>
                </div>
                <CardDescription>
                  The answer panel only commits when the model is grounded in retrieved chunks and citations.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea value={question} onChange={(event) => setQuestion(event.target.value)} />
                <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                  <div className="flex min-w-0 flex-wrap gap-2 text-xs text-muted-foreground">
                    <Badge
                      variant="outline"
                      className="max-w-full gap-1 overflow-hidden text-ellipsis whitespace-nowrap"
                      title={status?.documents_path ?? undefined}
                    >
                      <Database className="h-3 w-3" />
                      {formatPathLabel(status?.documents_path)}
                    </Badge>
                    <Badge variant="outline" className="gap-1">
                      <Shield className="h-3 w-3" />
                      {effectivePrincipals.length > 0 ? effectivePrincipals.join(", ") : "public-only"}
                    </Badge>
                  </div>
                  <Button onClick={handleAsk} disabled={isAsking}>
                    {isAsking ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />}
                    Ask
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="glass-panel flex-1">
              <CardHeader>
                <div className="flex items-center gap-2">
                  {deferredResult?.grounded ? (
                    <Badge variant="success">Grounded</Badge>
                  ) : deferredResult ? (
                    <Badge variant="warning">Abstained</Badge>
                  ) : (
                    <Badge variant="outline">Waiting</Badge>
                  )}
                  <CardTitle className="text-base">Answer stream</CardTitle>
                </div>
                <CardDescription>
                  {deferredResult
                    ? `Status: ${deferredResult.status}`
                    : "Run a query to see the answer, citations, and retrieval trace."}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {deferredResult ? (
                  <div className="space-y-4">
                    {!deferredResult.grounded && (
                      <div className="flex items-start gap-3 rounded-lg border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-100">
                        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                        <span>The model found related material but withheld a definitive answer because grounding was insufficient.</span>
                      </div>
                    )}
                    <div className="rounded-xl border border-border/70 bg-background/55 p-5 text-sm leading-7 text-foreground">
                      {deferredResult.answer}
                    </div>
                    {ingestReport ? (
                      <div className="grid gap-3 md:grid-cols-4">
                        <StatTile label="Processed" value={String(ingestReport.processed.length)} />
                        <StatTile label="Skipped" value={String(ingestReport.skipped.length)} />
                        <StatTile label="Deleted" value={String(ingestReport.deleted.length)} />
                        <StatTile label="Errors" value={String(Object.keys(ingestReport.errors).length)} />
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="flex min-h-[240px] items-center justify-center rounded-xl border border-dashed border-border/70 bg-background/35 px-6 text-center text-sm text-muted-foreground">
                    Pick a demo question or paste your own, then run retrieval to populate this panel.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </main>

        <section className="w-full lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)] lg:self-start">
          <Card className="glass-panel h-full">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-cyan-300" />
                <CardTitle className="text-base">Inspector</CardTitle>
              </div>
              <CardDescription>Citations, retrieved chunks, and the raw trace live here.</CardDescription>
            </CardHeader>
            <CardContent className="h-[calc(100%-5.75rem)] overflow-hidden">
              <Tabs defaultValue="citations" className="flex h-full flex-col">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="citations">Citations</TabsTrigger>
                  <TabsTrigger value="chunks">Chunks</TabsTrigger>
                  <TabsTrigger value="trace">Trace</TabsTrigger>
                </TabsList>
                <TabsContent value="citations" className="min-h-0 flex-1 overflow-auto pr-1">
                  <div className="space-y-3">
                    {deferredResult?.citations.length ? (
                      deferredResult.citations.map((citation) => (
                        <div key={citation.chunk_id} className="rounded-lg border border-border/70 bg-background/45 p-4">
                          <p className="text-sm font-medium text-foreground">{citation.citation}</p>
                          <p className="mt-2 text-xs text-muted-foreground">Chunk: {citation.chunk_id}</p>
                          {citation.reason ? <p className="mt-3 text-sm text-muted-foreground">{citation.reason}</p> : null}
                        </div>
                      ))
                    ) : (
                      <EmptyState message="No citations yet. Grounded answers will list the exact supporting chunks here." />
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="chunks" className="min-h-0 flex-1 overflow-auto pr-1">
                  <div className="space-y-3">
                    {deferredResult?.retrieved_chunks.length ? (
                      deferredResult.retrieved_chunks.map((chunk) => (
                        <details key={chunk.chunk_id} className="rounded-lg border border-border/70 bg-background/45 p-4">
                          <summary className="cursor-pointer list-none">
                            <p className="text-sm font-medium">{chunk.title}</p>
                            <p className="mt-1 text-xs text-muted-foreground">{chunk.location_label}</p>
                          </summary>
                          <Separator className="my-3" />
                          <p className="text-xs text-muted-foreground">
                            access_scope={chunk.access_scope} | principals={chunk.access_principals.join(", ")}
                          </p>
                          <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-foreground/90">{chunk.text}</p>
                        </details>
                      ))
                    ) : (
                      <EmptyState message="Retrieved chunks will appear here after you ask a question." />
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="trace" className="min-h-0 flex-1 overflow-auto pr-1">
                  <div className="rounded-lg border border-border/70 bg-background/45 p-4">
                    <pre className="overflow-x-auto text-xs leading-6 text-muted-foreground">
                      {deferredResult ? JSON.stringify(deferredResult.trace, null, 2) : "No trace yet."}
                    </pre>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}

function MetricGrid({ status, loading }: { status: StatusResponse | null; loading: boolean }) {
  const metrics = [
    { label: "Docs", value: status?.corpus.document_count ?? 0 },
    { label: "Chunks", value: status?.corpus.chunk_count ?? 0 },
    { label: "Restricted", value: status?.corpus.restricted_document_count ?? 0 },
    { label: "Public", value: status?.corpus.public_document_count ?? 0 },
  ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {metrics.map((metric) => (
        <div key={metric.label} className="rounded-lg border border-border/80 bg-background/45 p-3">
          <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{metric.label}</p>
          <p className="mt-2 text-2xl font-semibold text-foreground">{loading ? "--" : metric.value}</p>
        </div>
      ))}
    </div>
  );
}

function ModeButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-md px-3 py-2 text-sm transition",
        active ? "bg-background text-foreground shadow" : "text-muted-foreground hover:text-foreground",
      )}
    >
      {label}
    </button>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/70 bg-background/45 p-3">
      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-xl font-semibold">{value}</p>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex min-h-[220px] items-center justify-center rounded-lg border border-dashed border-border/70 bg-background/35 px-4 text-center text-sm text-muted-foreground">
      {message}
    </div>
  );
}

function formatPathLabel(path: string | undefined) {
  if (!path) {
    return "loading docs path";
  }
  const normalized = path.replace(/\\/g, "/");
  const segments = normalized.split("/").filter(Boolean);
  if (segments.length <= 4) {
    return normalized;
  }
  return `.../${segments.slice(-4).join("/")}`;
}
