"""End-to-end loop tests with the mock backbone + real ChromaDB store."""
import numpy as np
import pytest

from ergo.backbones import MockBackbone
from ergo.config import ErgoConfig
from ergo.orchestrator import ErgoOrchestrator
from ergo.rag import HashEmbedder, VectorStore, chunk_document, corpus_idf
from ergo.tokenization import SimpleTokenizer

DOCS = {
    "geo": "The capital city of France is Paris and it hosts the Louvre museum. "
           "The capital of Japan is Tokyo which hosts the Imperial Palace grounds.",
    "sci": "Marie Curie was awarded the Nobel Prize in Chemistry in 1911 for radium. "
           "Albert Einstein received the Nobel Prize in Physics in 1921.",
}


@pytest.fixture()
def small_cfg():
    cfg = ErgoConfig()
    cfg.canvas.len_thought = 16
    cfg.canvas.len_action = 2
    cfg.canvas.len_action_input = 8
    cfg.canvas.len_observation = 24
    cfg.canvas.len_critique = 12
    cfg.canvas.len_answer = 12
    cfg.loop.n_parallel = 2
    cfg.loop.c_max = 3
    cfg.loop.total_steps = 24
    cfg.retrieval.k = 2
    cfg.retrieval.relevance_floor = 0.0
    return cfg


@pytest.fixture()
def store(tmp_path):
    embedder = HashEmbedder()
    vs = VectorStore(str(tmp_path / "chroma"), "documents", embedder)
    chunks = [c for did, text in DOCS.items() for c in chunk_document(did, text, 2000, 0)]
    vs.add(chunks)
    return vs, corpus_idf([c.text for c in chunks])


def _orch(store_idf, cfg, facts):
    store, idf = store_idf
    tok = SimpleTokenizer()
    backbone = MockBackbone(tok, facts)
    return ErgoOrchestrator(backbone, tok, store, cfg, idf=idf), tok


def test_retrieve_then_finish(store, small_cfg):
    """Unknown without evidence -> cycle 0 retrieves; evidence arrives ->
    later cycle votes finish; final answer contains the fact.
    NB: the fact key appears only in the DOCUMENT text, never in the query,
    so the mock cannot 'know' before retrieval injects the chunk."""
    orch, _ = _orch(store, small_cfg, facts={"louvre museum": "Paris is the capital"})
    res = orch.run("what is the capital city of France")
    assert res.cycles[0].action == "retrieve"
    assert res.n_retrievals >= 1
    assert res.terminated_by == "finish"
    assert "paris" in res.answer.lower()


def test_no_retrieval_when_confident(store, small_cfg):
    """Fact key present in the query itself -> model 'knows' from cycle 0 ->
    finish immediately, zero retrievals."""
    orch, _ = _orch(store, small_cfg, facts={"sky color": "the sky is blue"})
    res = orch.run("tell me the sky color please")
    assert res.cycles[0].action == "finish"
    assert res.n_retrievals == 0


def test_dedup_never_reinjects_same_chunk(store, small_cfg):
    orch, _ = _orch(store, small_cfg, facts={"never-matching-key": "nope"})
    res = orch.run("what is the capital city of France")
    seen = [cid for log in res.cycles for cid in log.retrieved_ids]
    assert len(seen) == len(set(seen))          # monotone novelty


def test_exhaustion_forces_termination(tmp_path, small_cfg):
    """Empty store -> retrievals return nothing -> exhaustion brake ends loop."""
    vs = VectorStore(str(tmp_path / "empty"), "documents", HashEmbedder())
    tok = SimpleTokenizer()
    backbone = MockBackbone(tok, facts={"never": "no"})
    orch = ErgoOrchestrator(backbone, tok, vs, small_cfg, idf={})
    res = orch.run("unanswerable question about nothing indexed")
    assert res.terminated_by in ("exhausted", "c_max")
    assert res.cycles                            # loop ran and terminated cleanly


def test_closure_pass_fills_answer(store, small_cfg):
    """Whatever happens, the returned answer region is fully demasked."""
    orch, tok = _orch(store, small_cfg, facts={"capital city of France": "Paris is the capital"})
    res = orch.run("what is the capital city of France")
    assert "<mask>" not in res.answer


def test_rollback_triggers_on_contradicting_evidence(store, small_cfg):
    """Force a wrong early commit; mock's rescore crushes p' for tokens that
    contradict the evidence-informed target -> C1 clears them."""
    store_, idf = store
    tok = SimpleTokenizer()
    backbone = MockBackbone(tok, facts={"louvre": "Paris hosts the louvre museum"},
                            unknown_conf=0.9)   # confidently wrong pre-evidence
    orch = ErgoOrchestrator(backbone, tok, store_, small_cfg, idf=idf)
    res = orch.run("which european capital city hosts a famous art museum")
    rolled = sum(l.rollback.n_rolled for l in res.cycles if l.rollback)
    assert rolled > 0                            # contradicted commits were cleared
    assert "paris" in res.answer.lower()
