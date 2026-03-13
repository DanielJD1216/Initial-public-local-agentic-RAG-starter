from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ingest_bridge import IngestEnrichmentClient, LocalhostBridgeEnrichmentClient
from .agent import TransparentRAGAgent
from .clients import ChatClient, EmbeddingClient, OllamaChatClient, OllamaEmbeddingClient
from .config import AppConfig, load_config
from .ingestion import IngestionService
from .retrieval import HybridRetriever
from .storage import SQLiteStore
from .vector_index import VectorIndex


@dataclass(slots=True)
class AppRuntime:
    config: AppConfig
    store: SQLiteStore
    retriever: HybridRetriever
    ingestion: IngestionService
    agent: TransparentRAGAgent


def build_runtime(
    *,
    config_path: str | Path | None = None,
    embedding_client: EmbeddingClient | None = None,
    chat_client: ChatClient | None = None,
    ingest_enrichment_client: IngestEnrichmentClient | None = None,
    skip_retriever_validation: bool = False,
) -> AppRuntime:
    config = load_config(config_path)
    return build_runtime_from_config(
        config=config,
        embedding_client=embedding_client,
        chat_client=chat_client,
        ingest_enrichment_client=ingest_enrichment_client,
        skip_retriever_validation=skip_retriever_validation,
    )


def build_runtime_from_config(
    *,
    config: AppConfig,
    embedding_client: EmbeddingClient | None = None,
    chat_client: ChatClient | None = None,
    ingest_enrichment_client: IngestEnrichmentClient | None = None,
    skip_retriever_validation: bool = False,
) -> AppRuntime:
    config.ensure_runtime_directories()
    store = SQLiteStore(config.paths.database)
    store.initialize()
    vector_index = VectorIndex(
        backend=config.retrieval.vector_backend,
        index_path=config.paths.vector_index,
        metadata_path=config.paths.vector_metadata,
    )
    embedding_client = embedding_client or OllamaEmbeddingClient(
        base_url=config.models.base_url,
        model=config.models.embedding_model,
        timeout_seconds=config.models.request_timeout_seconds,
    )
    chat_client = chat_client or OllamaChatClient(
        base_url=config.models.base_url,
        model=config.models.chat_model,
        timeout_seconds=config.models.request_timeout_seconds,
        disable_thinking=config.models.disable_thinking,
    )
    if ingest_enrichment_client is None and config.ingest.mode == "bridge":
        ingest_enrichment_client = LocalhostBridgeEnrichmentClient(
            base_url=config.ingest.bridge_base_url,
            model=config.ingest.bridge_model,
            timeout_seconds=config.ingest.request_timeout_seconds,
        )
    retriever = HybridRetriever(config=config, store=store, vector_index=vector_index, embedding_client=embedding_client)
    if not skip_retriever_validation:
        retriever.ensure_index()
    ingestion = IngestionService(
        config=config,
        store=store,
        embedding_client=embedding_client,
        retriever=retriever,
        ingest_enrichment_client=ingest_enrichment_client,
    )
    agent = TransparentRAGAgent(config=config, retriever=retriever, chat_client=chat_client)
    return AppRuntime(config=config, store=store, retriever=retriever, ingestion=ingestion, agent=agent)
