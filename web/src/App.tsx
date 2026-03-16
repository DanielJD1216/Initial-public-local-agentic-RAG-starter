import { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Database,
  FolderSearch,
  LoaderCircle,
  RefreshCw,
  Server,
  Shield,
  Sparkles,
  Waypoints,
} from "lucide-react";

import {
  applyModelSettings,
  askQuestion,
  cancelPendingModelSettings,
  discoverModelSettings,
  fetchSuggestedPrompts,
  fetchStatus,
  ingestDocuments,
  reloadRuntime,
  reindexPendingModelSettings,
  uploadDocumentFolder,
  type ActiveModelSettings,
  type AskResponse,
  type BridgeHealthResponse,
  type CancelModelSettingsResponse,
  type IngestResponse,
  type ModelDiscoveryResponse,
  type PendingModelChange,
  type ReindexModelSettingsResponse,
  type StatusResponse,
  type SuggestedPrompt,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

const DEFAULT_PROMPTS: SuggestedPrompt[] = [
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
type DirectoryFile = File & { webkitRelativePath?: string };

const SUPPORTED_DOC_EXTENSIONS = new Set([".pdf", ".md", ".markdown", ".txt", ".docx"]);
const SIDECAR_SUFFIXES = [".meta.yaml", ".meta.yml", ".meta.json"];

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [ingestReport, setIngestReport] = useState<IngestResponse["report"] | null>(null);
  const [question, setQuestion] = useState(DEFAULT_PROMPTS[0].prompt);
  const [suggestedPrompts, setSuggestedPrompts] = useState<SuggestedPrompt[]>(DEFAULT_PROMPTS);
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
  const [modelBaseUrl, setModelBaseUrl] = useState("");
  const [draftChatModel, setDraftChatModel] = useState("");
  const [draftEmbeddingModel, setDraftEmbeddingModel] = useState("");
  const [modelDiscovery, setModelDiscovery] = useState<ModelDiscoveryResponse | null>(null);
  const [isDiscoveringModels, setIsDiscoveringModels] = useState(false);
  const [isApplyingModelSettings, setIsApplyingModelSettings] = useState(false);
  const [isReindexingModels, setIsReindexingModels] = useState(false);
  const [isCancellingPendingModelChange, setIsCancellingPendingModelChange] = useState(false);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const deferredResult = useDeferredValue(result);

  useEffect(() => {
    void loadStatus();
  }, []);

  useEffect(() => {
    const input = folderInputRef.current;
    if (!input) {
      return;
    }
    input.setAttribute("webkitdirectory", "");
    input.setAttribute("directory", "");
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

  const permissionsAreInteractive = Boolean(status?.permissions_enabled);
  const principalsSelectionEnabled = permissionsAreInteractive && accessMode === "custom";
  const principalsKey = effectivePrincipals.join("\u0001");

  useEffect(() => {
    if (!status) {
      return;
    }
    void loadSuggestedPrompts(effectivePrincipals);
  }, [status, principalsKey]);

  function hydrateModelControls(payload: StatusResponse, options: { syncDiscovery: boolean }) {
    const { syncDiscovery } = options;
    const selection = payload.local_models.pending_reindex ?? payload.local_models.active;
    setModelBaseUrl(selection.base_url);
    setDraftChatModel(selection.chat_model);
    setDraftEmbeddingModel(selection.embedding_model);
    if (syncDiscovery) {
      if (payload.local_models.ollama.base_url === selection.base_url) {
        setModelDiscovery(payload.local_models.ollama);
      } else {
        setModelDiscovery(null);
      }
    }
  }

  async function loadStatus() {
    setIsLoadingStatus(true);
    setErrorMessage(null);
    try {
      const payload = await fetchStatus();
      startTransition(() => {
        setStatus(payload);
        setDocumentsPath(payload.documents_source === "path" ? payload.documents_path : "");
        setSelectedPrincipals(payload.default_principals.filter((value) => value !== "*"));
      });
      applySuggestedPrompts(payload.suggested_prompts);
      hydrateModelControls(payload, { syncDiscovery: true });
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
        setDocumentsPath(payload.documents_source === "path" ? payload.documents_path : "");
        setStatusMessage("Runtime reloaded from the current config.");
      });
      applySuggestedPrompts(payload.suggested_prompts);
      hydrateModelControls(payload, { syncDiscovery: true });
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
        setDocumentsPath(payload.status.documents_source === "path" ? payload.status.documents_path : "");
        setIngestReport(payload.report);
        setResult(null);
        setStatusMessage(
          `Indexed ${payload.report.processed.length} file(s), skipped ${payload.report.skipped.length}, deleted ${payload.report.deleted.length}.`,
        );
      });
      applySuggestedPrompts(payload.status.suggested_prompts);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Indexing failed.");
    } finally {
      setIsIndexing(false);
    }
  }

  async function handleFolderSelection(event: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []) as DirectoryFile[];
    event.target.value = "";
    if (!files.length) {
      return;
    }

    const supportedFiles = files.filter(isSupportedFolderUpload);
    const ignoredCount = files.length - supportedFiles.length;
    if (!supportedFiles.length) {
      setErrorMessage("The selected folder did not contain any supported documents or metadata sidecars.");
      return;
    }

    const folderName = inferFolderName(files);
    setIsIndexing(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await uploadDocumentFolder(supportedFiles, folderName);
      startTransition(() => {
        setStatus(payload.status);
        setDocumentsPath("");
        setIngestReport(payload.report);
        setResult(null);
        const ignoredNote = ignoredCount > 0 ? ` Ignored ${ignoredCount} unsupported file(s).` : "";
        setStatusMessage(`Indexed ${payload.report.processed.length} file(s) from ${folderName}.${ignoredNote}`);
      });
      applySuggestedPrompts(payload.status.suggested_prompts);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Folder indexing failed.");
    } finally {
      setIsIndexing(false);
    }
  }

  function openFolderPicker() {
    setErrorMessage(null);
    folderInputRef.current?.click();
  }

  async function handleCheckModelConnection() {
    setIsDiscoveringModels(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await discoverModelSettings(modelBaseUrl.trim());
      const targetSelection = status?.local_models.pending_reindex ?? status?.local_models.active ?? null;
      startTransition(() => {
        setModelBaseUrl(payload.base_url);
        setModelDiscovery(payload);
        setDraftChatModel((current) =>
          pickDiscoveredModel(current, payload.models, targetSelection?.chat_model ?? draftChatModel),
        );
        setDraftEmbeddingModel((current) =>
          pickDiscoveredModel(current, payload.models, targetSelection?.embedding_model ?? draftEmbeddingModel),
        );
      });
      if (!payload.reachable) {
        setErrorMessage(payload.error ?? `Could not reach Ollama at ${payload.base_url}.`);
      } else {
        setStatusMessage(
          payload.models.length > 0
            ? `Connected to Ollama. Found ${payload.models.length} installed model(s).`
            : payload.error ?? "Connected to Ollama.",
        );
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Model discovery failed.");
    } finally {
      setIsDiscoveringModels(false);
    }
  }

  async function handleApplyModelSettings() {
    if (!modelBaseUrl.trim()) {
      setErrorMessage("Enter an Ollama base URL before applying model settings.");
      return;
    }
    if (!draftChatModel || !draftEmbeddingModel) {
      setErrorMessage("Choose both a chat model and an embedding model before applying.");
      return;
    }

    setIsApplyingModelSettings(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await applyModelSettings(modelBaseUrl.trim(), draftChatModel, draftEmbeddingModel);
      startTransition(() => {
        setStatus(payload.status);
      });
      hydrateModelControls(payload.status, { syncDiscovery: !payload.reindex_required });
      setStatusMessage(payload.message);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Model settings could not be applied.");
    } finally {
      setIsApplyingModelSettings(false);
    }
  }

  async function handleReindexPendingModelChange() {
    setIsReindexingModels(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await reindexPendingModelSettings();
      applyModelReindexPayload(payload);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Reindex failed while activating the staged model.");
    } finally {
      setIsReindexingModels(false);
    }
  }

  async function handleCancelPendingModelChange() {
    setIsCancellingPendingModelChange(true);
    setErrorMessage(null);
    setStatusMessage(null);
    try {
      const payload = await cancelPendingModelSettings();
      applyModelCancelPayload(payload);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "The staged model change could not be cleared.");
    } finally {
      setIsCancellingPendingModelChange(false);
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
        setStatusMessage(answerStatusMessage(payload));
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

  function applyModelReindexPayload(payload: ReindexModelSettingsResponse) {
    startTransition(() => {
      setStatus(payload.status);
      setIngestReport(payload.report);
      setResult(null);
    });
    hydrateModelControls(payload.status, { syncDiscovery: true });
    setStatusMessage(payload.message);
  }

  function applyModelCancelPayload(payload: CancelModelSettingsResponse) {
    startTransition(() => {
      setStatus(payload.status);
    });
    hydrateModelControls(payload.status, { syncDiscovery: true });
    setStatusMessage(payload.message);
  }

  async function loadSuggestedPrompts(principals: string[]) {
    try {
      const payload = await fetchSuggestedPrompts(principals);
      applySuggestedPrompts(payload.prompts);
    } catch {
      // Keep the current chips if prompt generation is unavailable.
    }
  }

  function applySuggestedPrompts(nextPrompts: SuggestedPrompt[]) {
    const normalized = nextPrompts.length > 0 ? nextPrompts : DEFAULT_PROMPTS;
    const shouldReplaceQuestion = !question.trim() || suggestedPrompts.some((item) => item.prompt === question);
    startTransition(() => {
      setSuggestedPrompts(normalized);
      if (shouldReplaceQuestion && normalized[0]) {
        setQuestion(normalized[0].prompt);
      }
    });
  }

  const activeDocumentsLabel = status?.documents_display_path ?? status?.documents_path ?? "No indexed folder yet";
  const activeModelSettings = status?.local_models.active ?? null;
  const pendingModelChange = status?.local_models.pending_reindex ?? null;
  const modelSourceLabel = status?.local_models.source === "session" ? "Session override" : "Config default";
  const ingestStatus = status?.ingest ?? null;
  const bridgeStatus = ingestStatus?.bridge ?? null;
  const corpusIngest = ingestStatus?.corpus ?? null;
  const availableModels = modelDiscovery?.reachable ? modelDiscovery.models : [];
  const agentStatus = status?.agent ?? null;
  const selectionsAreDiscoverable =
    availableModels.includes(draftChatModel) && availableModels.includes(draftEmbeddingModel) && availableModels.length > 0;
  const comparisonTarget = pendingModelChange ?? activeModelSettings;
  const draftMatchesTarget = modelSelectionMatchesDraft(comparisonTarget, modelBaseUrl, draftChatModel, draftEmbeddingModel);
  const modelPanelBusy =
    isDiscoveringModels || isApplyingModelSettings || isReindexingModels || isCancellingPendingModelChange;
  const canApplyModelSettings = Boolean(modelBaseUrl.trim()) && selectionsAreDiscoverable && !draftMatchesTarget && !modelPanelBusy;

  return (
    <div className="min-h-screen px-4 py-5 md:px-6 lg:px-8 xl:px-10">
      <div className="mx-auto grid max-w-[1560px] gap-5 lg:grid-cols-[320px_minmax(0,1fr)_360px]">
        <aside className="w-full lg:sticky lg:top-4 lg:max-h-[calc(100vh-2rem)] lg:self-start">
          <div className="flex h-full flex-col gap-4 lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto lg:pr-1 sidebar-scroll">
            <Card className="glass-panel">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <Badge variant="outline" className="border-sky-200 bg-sky-50 text-sky-700">
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
                <div className="tonal-card space-y-2">
                  <p className="panel-kicker">Live control room</p>
                  <p className="text-sm leading-6 text-foreground">
                    Swap folders, inspect grounding, and tune the local answer stack without leaving this screen.
                  </p>
                </div>
                <div className="rounded-2xl border border-white/70 bg-white/58 p-4 shadow-[0_12px_30px_rgba(15,23,42,0.05)]">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <p className="panel-kicker">Agent runtime</p>
                      <p className="text-sm font-semibold text-foreground">
                        {agentStatus?.active_mode === "middleweight" ? "Middle-weight planner active" : "Lightweight fallback active"}
                      </p>
                      <p className="text-xs leading-5 text-muted-foreground">
                        Configured: {agentStatus?.configured_mode ?? "loading"} | artifacts:{" "}
                        {agentStatus?.planning_artifacts_available ? "ready" : "needs reindex"}
                      </p>
                    </div>
                    <Badge variant={agentStatus?.active_mode === "middleweight" ? "success" : "outline"}>
                      {agentStatus?.active_mode === "middleweight" ? "Middle-weight" : "Lightweight"}
                    </Badge>
                  </div>
                  {agentStatus?.downgrade_reason ? (
                    <div className="mt-3 rounded-lg border border-amber-200/90 bg-amber-50/85 px-3 py-2 text-xs text-amber-900">
                      {agentStatus.downgrade_reason}
                    </div>
                  ) : null}
                </div>
                <MetricGrid status={status} loading={isLoadingStatus} />
                <div className="space-y-2">
                  <p className="panel-kicker">Documents folder</p>
                  <Input
                    value={documentsPath}
                    onChange={(event) => setDocumentsPath(event.target.value)}
                    placeholder="/absolute/path/to/your/docs"
                  />
                  <input ref={folderInputRef} type="file" multiple className="hidden" onChange={handleFolderSelection} />
                  <p className="text-xs text-muted-foreground">
                    Choose a folder from a popup, or paste a real path if you want the backend to read directly from disk.
                  </p>
                </div>
                <div className="flex flex-col gap-2 sm:flex-row lg:flex-col">
                  <Button onClick={openFolderPicker} disabled={isIndexing} className="w-full">
                    {isIndexing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <FolderSearch className="h-4 w-4" />}
                    Choose folder
                  </Button>
                  <Button variant="secondary" onClick={handleIndex} disabled={isIndexing} className="w-full">
                    {isIndexing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <FolderSearch className="h-4 w-4" />}
                    Index pasted path
                  </Button>
                  <Button variant="secondary" onClick={handleReload} disabled={isReloading} className="w-full">
                    {isReloading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                    Reload runtime
                  </Button>
                </div>
                <div className="tonal-card">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Server className="h-4 w-4 text-sky-600" />
                        <p className="panel-kicker">Model settings</p>
                      </div>
                      <p className="text-sm font-semibold text-foreground">{modelSourceLabel}</p>
                      <p className="text-xs leading-5 text-muted-foreground">
                        Chat: {activeModelSettings?.chat_model ?? "..."} | embeddings: {activeModelSettings?.embedding_model ?? "..."}
                      </p>
                    </div>
                    <Badge variant={connectionBadgeVariant(modelDiscovery)}>{connectionBadgeLabel(modelDiscovery)}</Badge>
                  </div>

                  <div className="mt-4 space-y-3">
                    <div className="space-y-2">
                      <label htmlFor="ollama-base-url" className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Ollama base URL
                      </label>
                      <Input
                        id="ollama-base-url"
                        value={modelBaseUrl}
                        onChange={(event) => {
                          const nextValue = event.target.value;
                          setModelBaseUrl(nextValue);
                          setModelDiscovery((current) =>
                            current && current.base_url === nextValue.trim() ? current : null,
                          );
                        }}
                        placeholder="http://127.0.0.1:11434"
                      />
                      <p className="text-xs text-muted-foreground">
                        These changes stay in this running web session only. Restarting the backend returns to the checked-in config.
                      </p>
                    </div>

                    <Button variant="secondary" onClick={handleCheckModelConnection} disabled={modelPanelBusy} className="w-full">
                      {isDiscoveringModels ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                      Check connection
                    </Button>

                    {modelDiscovery?.error ? (
                      <div
                        className={cn(
                          "rounded-lg border px-3 py-2 text-xs",
                          modelDiscovery.reachable
                            ? "border-amber-200 bg-amber-50 text-amber-800"
                            : "border-red-200 bg-red-50 text-red-800",
                        )}
                      >
                        {modelDiscovery.error}
                      </div>
                    ) : null}

                    <ModelSelect
                      id="chat-model"
                      label="Chat model"
                      value={draftChatModel}
                      options={availableModels}
                      disabled={!modelDiscovery?.reachable || modelPanelBusy}
                      onChange={setDraftChatModel}
                      placeholder="Check connection to load models"
                    />

                    <ModelSelect
                      id="embedding-model"
                      label="Embedding model"
                      value={draftEmbeddingModel}
                      options={availableModels}
                      disabled={!modelDiscovery?.reachable || modelPanelBusy}
                      onChange={setDraftEmbeddingModel}
                      placeholder="Check connection to load models"
                    />

                    <p className="text-xs leading-5 text-muted-foreground">
                      Chat-model changes apply immediately. Embedding-model changes are staged until you confirm a reindex.
                    </p>

                    <Button onClick={handleApplyModelSettings} disabled={!canApplyModelSettings} className="w-full">
                      {isApplyingModelSettings ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Server className="h-4 w-4" />}
                      Apply model settings
                    </Button>

                    {pendingModelChange ? (
                      <div className="space-y-3 rounded-2xl border border-amber-200/90 bg-amber-50/90 p-4 text-sm text-amber-900">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                          <div className="space-y-1">
                            <p className="font-medium">Embedding change staged</p>
                            <p className="text-xs leading-5 text-amber-800">
                              The runtime is still using {activeModelSettings?.embedding_model ?? "the current embedding model"}.
                              Reindex to activate {pendingModelChange.embedding_model}.
                            </p>
                          </div>
                        </div>
                        <div className="rounded-lg border border-amber-200/80 bg-white/60 p-3 text-xs text-amber-900">
                          <p>Pending base URL: {pendingModelChange.base_url}</p>
                          <p className="mt-1">Pending chat: {pendingModelChange.chat_model}</p>
                          <p className="mt-1">Pending embeddings: {pendingModelChange.embedding_model}</p>
                        </div>
                        <div className="flex flex-col gap-2">
                          <Button onClick={handleReindexPendingModelChange} disabled={modelPanelBusy} className="w-full">
                            {isReindexingModels ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                            Reindex to activate
                          </Button>
                          <Button
                            variant="secondary"
                            onClick={handleCancelPendingModelChange}
                            disabled={modelPanelBusy}
                            className="w-full"
                          >
                            {isCancellingPendingModelChange ? (
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                            ) : (
                              <AlertTriangle className="h-4 w-4" />
                            )}
                            Keep current runtime
                          </Button>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>

                <div className="tonal-card">
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <Waypoints className="h-4 w-4 text-sky-600" />
                        <p className="panel-kicker">Ingest pipeline</p>
                      </div>
                      <p className="text-sm font-semibold text-foreground">
                        {ingestStatus?.mode === "bridge" ? "Bridge enrichment" : "Local heuristic ingest"}
                      </p>
                      <p className="text-xs leading-5 text-muted-foreground">
                        Bridge model: {bridgeStatus?.model ?? "..."} | corpus: {corpusIngest?.mode ?? "unindexed"}
                      </p>
                    </div>
                    <Badge variant={bridgeBadgeVariant(bridgeStatus)}>{bridgeBadgeLabel(bridgeStatus, ingestStatus?.mode)}</Badge>
                  </div>

                  <div className="mt-4 grid gap-3 text-xs text-muted-foreground sm:grid-cols-2 lg:grid-cols-1">
                    <div className="rounded-lg border border-border/70 bg-background/55 p-3">
                      <p className="uppercase tracking-[0.18em]">Bridge URL</p>
                      <p className="mt-2 break-all text-sm text-foreground">{bridgeStatus?.base_url ?? "http://127.0.0.1:8787"}</p>
                    </div>
                    <div className="rounded-lg border border-border/70 bg-background/55 p-3">
                      <p className="uppercase tracking-[0.18em]">Indexed corpus</p>
                      <p className="mt-2 text-sm text-foreground">
                        {corpusIngest?.document_count ? `${corpusIngest.document_count} docs via ${corpusIngest.mode}` : "No indexed docs yet"}
                      </p>
                      <p className="mt-1">
                        Chunking: {corpusIngest?.chunking_strategy ?? "n/a"}
                      </p>
                      <p className="mt-1">
                        Fingerprint: {shortFingerprint(corpusIngest?.ingest_fingerprint)}
                      </p>
                    </div>
                  </div>

                  {bridgeStatus?.error ? (
                    <div
                      className={cn(
                        "mt-3 rounded-lg border px-3 py-2 text-xs",
                        bridgeStatus.reachable ? "border-amber-200 bg-amber-50 text-amber-800" : "border-red-200 bg-red-50 text-red-800",
                      )}
                    >
                      {bridgeStatus.error}
                    </div>
                  ) : null}
                </div>
              </CardContent>
            </Card>

            <Card className="glass-panel">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-sky-600" />
                  <CardTitle className="text-base">Access view</CardTitle>
                </div>
                <CardDescription>
                  {permissionsAreInteractive
                    ? "Config uses the runtime defaults. Public hides restricted docs. Custom lets you simulate principals like owners or finance."
                    : "Permissions are off in this runtime, so these controls do not change retrieval yet."}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {!permissionsAreInteractive ? (
                  <StatusBanner
                    tone="warning"
                    message="Permissions are disabled in the current config. Queries and prompt suggestions are using the full visible corpus."
                  />
                ) : null}
                <div className="grid grid-cols-3 gap-2 rounded-lg border border-border/70 bg-background/50 p-1">
                  <ModeButton
                    label="Config"
                    active={accessMode === "default"}
                    disabled={!permissionsAreInteractive}
                    onClick={() => setAccessMode("default")}
                  />
                  <ModeButton
                    label="Public"
                    active={accessMode === "public"}
                    disabled={!permissionsAreInteractive}
                    onClick={() => setAccessMode("public")}
                  />
                  <ModeButton
                    label="Custom"
                    active={accessMode === "custom"}
                    disabled={!permissionsAreInteractive}
                    onClick={() => setAccessMode("custom")}
                  />
                </div>
                <div className="space-y-2">
                  <p className="panel-kicker">Discovered principals</p>
                  <div className="flex flex-wrap gap-2">
                    {(status?.corpus.principals ?? []).map((principal) => (
                      <button
                        key={principal}
                        type="button"
                        onClick={() => togglePrincipal(principal)}
                        disabled={!principalsSelectionEnabled}
                        className={cn(
                          "rounded-full border px-3 py-1 text-xs transition",
                          !principalsSelectionEnabled && "cursor-not-allowed opacity-55",
                          selectedPrincipals.includes(principal)
                            ? "border-sky-300 bg-sky-50 text-sky-700"
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
                  <p className="panel-kicker">Extra principals</p>
                  <Input
                    value={extraPrincipals}
                    onChange={(event) => setExtraPrincipals(event.target.value)}
                    placeholder="owners, finance"
                    disabled={!principalsSelectionEnabled}
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
            <Card className="glass-panel metric-sheen hero-panel overflow-hidden">
              <CardHeader className="pb-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline" className="border-sky-200/80 bg-white/75 text-sky-800">
                    Grounded retrieval desk
                  </Badge>
                  <Badge variant="secondary" className="bg-white/65">
                    {status?.corpus.document_count ?? 0} docs live
                  </Badge>
                  <Badge variant="secondary" className="bg-white/65">
                    {ingestStatus?.mode === "bridge" ? "Bridge ingest" : "Local ingest"}
                  </Badge>
                  <Badge variant="secondary" className="bg-white/65">
                    {agentStatus?.active_mode === "middleweight" ? "Middle-weight agent" : "Lightweight fallback"}
                  </Badge>
                </div>
                <CardTitle className="text-2xl md:text-3xl">Grounded answers, local models, real folder testing.</CardTitle>
                <CardDescription className="max-w-2xl text-sm leading-6">
                  Start with the sample prompts, then switch the documents folder to your real corpus and re-index from this screen.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_280px]">
                  <div className="flex flex-wrap gap-2">
                    {suggestedPrompts.map((item) => (
                      <Button key={item.label} variant="outline" size="sm" onClick={() => setQuestion(item.prompt)}>
                        <Sparkles className="h-3.5 w-3.5" />
                        {item.label}
                      </Button>
                    ))}
                  </div>
                  <div className="rounded-2xl border border-white/70 bg-white/55 p-4 shadow-[0_18px_50px_rgba(15,23,42,0.08)]">
                    <p className="panel-kicker">Best results</p>
                    <p className="mt-2 text-sm leading-6 text-foreground">
                      Name the document, the object you want, and the exact thing you need verified. Short ambiguous prompts are more likely to abstain.
                    </p>
                  </div>
                </div>
                {statusMessage ? <StatusBanner tone="success" message={statusMessage} /> : null}
                {errorMessage ? <StatusBanner tone="error" message={errorMessage} /> : null}
              </CardContent>
            </Card>

            <Card className="glass-panel">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Waypoints className="h-4 w-4 text-sky-600" />
                  <CardTitle className="text-base">Ask a grounded question</CardTitle>
                </div>
                <CardDescription>
                  The answer panel only commits when the model is grounded in retrieved chunks and citations.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  className="min-h-[154px] rounded-2xl bg-white/70"
                />
                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                  <Badge variant="outline" className="rounded-full bg-white/70 px-3 py-1">
                    Ask with document title + exact target
                  </Badge>
                  <Badge variant="outline" className="rounded-full bg-white/70 px-3 py-1">
                    The system abstains instead of guessing
                  </Badge>
                </div>
                <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                  <div className="flex min-w-0 flex-wrap gap-2 text-xs text-muted-foreground">
                    <Badge
                      variant="outline"
                      className="max-w-full gap-1 overflow-hidden text-ellipsis whitespace-nowrap"
                      title={activeDocumentsLabel}
                    >
                      <Database className="h-3 w-3" />
                      {formatPathLabel(activeDocumentsLabel)}
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
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Badge variant={answerBadgeVariant(deferredResult)}>{answerBadgeLabel(deferredResult)}</Badge>
                    <CardTitle className="text-base">Answer stream</CardTitle>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                    <Badge variant="outline" className="bg-white/70">
                      Citations {deferredResult?.citations.length ?? 0}
                    </Badge>
                    <Badge variant="outline" className="bg-white/70">
                      Chunks {deferredResult?.retrieved_chunks.length ?? 0}
                    </Badge>
                  </div>
                </div>
                <CardDescription>
                  {deferredResult
                    ? `Status: ${deferredResult.status}${deferredResult.stop_reason ? ` • stop: ${deferredResult.stop_reason}` : ""}`
                    : "Run a query to see the answer, citations, and retrieval trace."}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {deferredResult ? (
                  <div className="space-y-4">
                    {!deferredResult.grounded ? (
                      <StatusBanner tone={answerBannerTone(deferredResult)} message={answerBannerMessage(deferredResult)} />
                    ) : null}
                    <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                      <Badge variant="outline" className="bg-white/70">
                        Task {deferredResult.task_mode.replaceAll("_", " ")}
                      </Badge>
                      <Badge variant="outline" className="bg-white/70">
                        Agent {deferredResult.active_mode}
                      </Badge>
                      {deferredResult.failure_reason ? (
                        <Badge variant="outline" className="bg-white/70">
                          Reason {deferredResult.failure_reason.replaceAll("_", " ")}
                        </Badge>
                      ) : null}
                    </div>
                    <div className="rounded-[1.4rem] border border-white/70 bg-white/68 p-5 text-sm leading-7 text-foreground shadow-[0_16px_40px_rgba(15,23,42,0.06)]">
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
                  <EmptyState
                    title="No answer yet"
                    message="Pick a demo question or paste your own, then run retrieval to populate this panel."
                  />
                )}
              </CardContent>
            </Card>
          </div>
        </main>

        <section className="w-full lg:sticky lg:top-4 lg:h-[calc(100vh-2rem)] lg:self-start">
          <Card className="glass-panel h-full">
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-sky-600" />
                  <CardTitle className="text-base">Inspector</CardTitle>
                </div>
                <Badge variant="outline" className="bg-white/70">
                  {deferredResult ? "Live query details" : "Waiting for a run"}
                </Badge>
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
                        <div key={citation.chunk_id} className="rounded-2xl border border-border/70 bg-background/50 p-4">
                          <p className="text-sm font-medium text-foreground">{citation.citation}</p>
                          <p className="mt-2 text-xs text-muted-foreground">Chunk: {citation.chunk_id}</p>
                          {citation.reason ? <p className="mt-3 text-sm text-muted-foreground">{citation.reason}</p> : null}
                        </div>
                      ))
                    ) : (
                      <EmptyState
                        title="No citations yet"
                        message="Grounded answers will list the exact supporting chunks here."
                      />
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="chunks" className="min-h-0 flex-1 overflow-auto pr-1">
                  <div className="space-y-3">
                    {deferredResult?.retrieved_chunks.length ? (
                      deferredResult.retrieved_chunks.map((chunk) => (
                        <details key={chunk.chunk_id} className="rounded-2xl border border-border/70 bg-background/50 p-4">
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
                      <EmptyState
                        title="No chunks yet"
                        message="Retrieved chunks will appear here after you ask a question."
                      />
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="trace" className="min-h-0 flex-1 overflow-auto pr-1">
                  {deferredResult ? (
                    <div className="space-y-3">
                      <div className="grid gap-3 md:grid-cols-2">
                        <div className="rounded-2xl border border-border/70 bg-background/50 p-4">
                          <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Planner</p>
                          <p className="mt-2 text-sm text-foreground">
                            {deferredResult.active_mode} | stop: {deferredResult.stop_reason ?? "n/a"}
                          </p>
                          {deferredResult.downgrade_reason ? (
                            <p className="mt-2 text-xs leading-5 text-muted-foreground">{deferredResult.downgrade_reason}</p>
                          ) : null}
                        </div>
                        <div className="rounded-2xl border border-border/70 bg-background/50 p-4">
                          <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Verifier</p>
                          <p className="mt-2 text-sm text-foreground">{deferredResult.verifier_summary?.status ?? "n/a"}</p>
                          {deferredResult.verifier_summary?.notes?.length ? (
                            <p className="mt-2 text-xs leading-5 text-muted-foreground">
                              {deferredResult.verifier_summary.notes.join(" ")}
                            </p>
                          ) : null}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-border/70 bg-background/50 p-4">
                        <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Plan steps</p>
                        <div className="mt-3 space-y-3">
                          {deferredResult.plan_summary.length ? (
                            deferredResult.plan_summary.map((step) => (
                              <div key={step.step_id} className="rounded-xl border border-border/60 bg-white/55 p-3">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <p className="text-sm font-medium text-foreground">{step.title}</p>
                                  <Badge variant="outline" className="bg-white/70">
                                    {step.status}
                                  </Badge>
                                </div>
                                {step.subquestion ? <p className="mt-2 text-xs text-muted-foreground">{step.subquestion}</p> : null}
                                {step.notes.length ? (
                                  <p className="mt-2 text-xs leading-5 text-muted-foreground">{step.notes.join(" ")}</p>
                                ) : null}
                              </div>
                            ))
                          ) : (
                            <p className="text-sm text-muted-foreground">No plan steps recorded.</p>
                          )}
                        </div>
                      </div>
                      <div className="rounded-2xl border border-border/70 bg-background/50 p-4">
                        <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Tool events</p>
                        <div className="mt-3 space-y-3">
                          {deferredResult.tool_events.length ? (
                            deferredResult.tool_events.map((event, index) => (
                              <div key={`${event.tool_name}-${index}`} className="rounded-xl border border-border/60 bg-white/55 p-3">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <p className="text-sm font-medium text-foreground">{event.tool_name.replaceAll("_", " ")}</p>
                                  <Badge variant="outline" className="bg-white/70">
                                    {event.status}
                                  </Badge>
                                </div>
                                <p className="mt-2 text-xs leading-5 text-muted-foreground">{event.summary}</p>
                                {event.query ? <p className="mt-2 text-xs text-muted-foreground">query: {event.query}</p> : null}
                              </div>
                            ))
                          ) : (
                            <p className="text-sm text-muted-foreground">No tool events recorded.</p>
                          )}
                        </div>
                      </div>
                      <div className="rounded-lg border border-border/70 bg-background/45 p-4">
                        <pre className="overflow-x-auto text-xs leading-6 text-muted-foreground">
                          {JSON.stringify(deferredResult.trace, null, 2)}
                        </pre>
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-lg border border-border/70 bg-background/45 p-4">
                      <pre className="overflow-x-auto text-xs leading-6 text-muted-foreground">No trace yet.</pre>
                    </div>
                  )}
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
        <div key={metric.label} className="rounded-2xl border border-white/70 bg-white/62 p-3 shadow-[0_12px_30px_rgba(15,23,42,0.05)]">
          <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">{metric.label}</p>
          <p className="mt-2 text-2xl font-semibold text-foreground">{loading ? "--" : metric.value}</p>
        </div>
      ))}
    </div>
  );
}

function ModeButton({
  active,
  disabled,
  label,
  onClick,
}: {
  active: boolean;
  disabled?: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "rounded-xl px-3 py-2 text-sm transition",
        disabled && "cursor-not-allowed opacity-50",
        active
          ? "bg-background/90 text-foreground shadow-[0_10px_22px_rgba(15,23,42,0.08)]"
          : "text-muted-foreground hover:bg-white/50 hover:text-foreground",
      )}
    >
      {label}
    </button>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/70 bg-white/62 p-3 shadow-[0_12px_30px_rgba(15,23,42,0.05)]">
      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-xl font-semibold">{value}</p>
    </div>
  );
}

function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <div className="empty-state-shell flex min-h-[220px] flex-col items-center justify-center rounded-[1.4rem] px-6 text-center">
      <div className="rounded-full border border-sky-200/80 bg-white/78 p-3 shadow-[0_10px_24px_rgba(14,116,144,0.08)]">
        <Sparkles className="h-4 w-4 text-sky-700" />
      </div>
      <p className="mt-4 text-sm font-semibold text-foreground">{title}</p>
      <p className="mt-2 max-w-[28ch] text-sm leading-6 text-muted-foreground">{message}</p>
    </div>
  );
}

function StatusBanner({
  tone,
  message,
}: {
  tone: "success" | "error" | "warning";
  message: string;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border px-4 py-3 text-sm shadow-[0_12px_30px_rgba(15,23,42,0.04)]",
        tone === "success" && "border-emerald-200/90 bg-emerald-50/90 text-emerald-900",
        tone === "error" && "border-red-200/90 bg-red-50/90 text-red-900",
        tone === "warning" && "border-amber-200/90 bg-amber-50/90 text-amber-900",
      )}
    >
      {message}
    </div>
  );
}

function answerBadgeVariant(result: AskResponse | null): "outline" | "success" | "warning" {
  if (!result) {
    return "outline";
  }
  if (result.grounded) {
    return "success";
  }
  return result.failure_reason === "restricted" ? "warning" : "outline";
}

function answerBadgeLabel(result: AskResponse | null) {
  if (!result) {
    return "Waiting";
  }
  if (result.grounded) {
    return "Grounded";
  }
  if (result.failure_reason === "restricted") {
    return "Restricted";
  }
  if (result.failure_reason === "clarification_required") {
    return "Needs clarity";
  }
  if (result.failure_reason === "conflicting_sources") {
    return "Conflict";
  }
  return "Withheld";
}

function answerBannerTone(result: AskResponse): "success" | "error" | "warning" {
  if (result.failure_reason === "restricted") {
    return "error";
  }
  return "warning";
}

function answerBannerMessage(result: AskResponse) {
  if (result.failure_reason === "restricted") {
    return result.blocked_principals.length > 0
      ? `Restricted. Your current access view cannot use one or more matching documents. Try Custom and enable: ${result.blocked_principals.join(", ")}.`
      : "Restricted. Your current access view cannot use one or more matching documents.";
  }
  if (result.failure_reason === "clarification_required") {
    return result.clarification_prompt ?? "Clarification is required before the agent can continue.";
  }
  if (result.failure_reason === "conflicting_sources") {
    return "The agent found conflicting evidence across the matching documents, so it withheld a definitive answer.";
  }
  if (result.failure_reason === "generation_failure") {
    return "The runtime gathered evidence, but the active local chat model could not complete a structured grounded answer.";
  }
  if (result.failure_reason === "no_evidence") {
    return "The agent did not collect enough supporting evidence to answer safely.";
  }
  if (result.failure_reason === "partial_evidence") {
    return "The agent found related material, but the evidence was not strong enough to finalize the answer.";
  }
  return "The model found related material but withheld a definitive answer because grounding was insufficient.";
}

function answerStatusMessage(result: AskResponse) {
  if (result.grounded) {
    return "Grounded answer ready.";
  }
  if (result.failure_reason === "clarification_required") {
    return "Clarification required before the agent can continue.";
  }
  if (result.failure_reason === "restricted") {
    return "Access prevented the answer from using one or more matching documents.";
  }
  if (result.failure_reason === "conflicting_sources") {
    return "The answer was withheld because the matching documents conflicted.";
  }
  return "The answer was withheld cleanly after the planner and verifier checks.";
}

function ModelSelect({
  id,
  label,
  value,
  options,
  disabled,
  onChange,
  placeholder,
}: {
  id: string;
  label: string;
  value: string;
  options: string[];
  disabled: boolean;
  onChange: (nextValue: string) => void;
  placeholder: string;
}) {
  const normalizedValue = options.includes(value) ? value : "";
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
        {label}
      </label>
      <select
        id={id}
        value={normalizedValue}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background/60 px-3 py-2 text-sm text-foreground outline-none transition focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
          !normalizedValue && "text-muted-foreground",
        )}
      >
        <option value="">{placeholder}</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

function connectionBadgeVariant(discovery: ModelDiscoveryResponse | null): "outline" | "success" | "warning" | "secondary" {
  if (!discovery) {
    return "secondary";
  }
  if (!discovery.reachable) {
    return "warning";
  }
  if (!discovery.models.length) {
    return "outline";
  }
  return "success";
}

function connectionBadgeLabel(discovery: ModelDiscoveryResponse | null) {
  if (!discovery) {
    return "Unchecked";
  }
  if (!discovery.reachable) {
    return "Unreachable";
  }
  if (!discovery.models.length) {
    return "No models";
  }
  return `Connected (${discovery.models.length})`;
}

function bridgeBadgeVariant(status: BridgeHealthResponse | null): "outline" | "success" | "warning" | "secondary" {
  if (!status || status.reachable === null) {
    return "secondary";
  }
  if (!status.reachable) {
    return "warning";
  }
  return "success";
}

function bridgeBadgeLabel(status: BridgeHealthResponse | null, mode: StatusResponse["ingest"]["mode"] | undefined) {
  if (!status || status.reachable === null) {
    return mode === "local" ? "Inactive" : "Unchecked";
  }
  return status.reachable ? "Bridge ready" : "Bridge offline";
}

function modelSelectionMatchesDraft(
  selection: ActiveModelSettings | PendingModelChange | null,
  baseUrl: string,
  chatModel: string,
  embeddingModel: string,
) {
  if (!selection) {
    return false;
  }
  return (
    selection.base_url === baseUrl.trim() &&
    selection.chat_model === chatModel &&
    selection.embedding_model === embeddingModel
  );
}

function pickDiscoveredModel(currentValue: string, availableModels: string[], fallbackValue: string) {
  if (availableModels.includes(currentValue)) {
    return currentValue;
  }
  if (availableModels.includes(fallbackValue)) {
    return fallbackValue;
  }
  return availableModels[0] ?? "";
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

function shortFingerprint(fingerprint: string | null | undefined) {
  if (!fingerprint) {
    return "n/a";
  }
  if (fingerprint === "mixed") {
    return "mixed";
  }
  return `${fingerprint.slice(0, 8)}...`;
}

function inferFolderName(files: DirectoryFile[]) {
  const relativePath = files.find((file) => file.webkitRelativePath)?.webkitRelativePath ?? "";
  const folderName = relativePath.split("/")[0]?.trim();
  return folderName || "Selected folder";
}

function isSupportedFolderUpload(file: DirectoryFile) {
  const relativePath = (file.webkitRelativePath || file.name).replaceAll("\\", "/");
  const parts = relativePath.split("/").filter(Boolean);
  if (!parts.length || parts.some((part) => part.startsWith("."))) {
    return false;
  }
  const lowered = relativePath.toLowerCase();
  if (SIDECAR_SUFFIXES.some((suffix) => lowered.endsWith(suffix))) {
    return true;
  }
  const lastDot = lowered.lastIndexOf(".");
  if (lastDot === -1) {
    return false;
  }
  return SUPPORTED_DOC_EXTENSIONS.has(lowered.slice(lastDot));
}
