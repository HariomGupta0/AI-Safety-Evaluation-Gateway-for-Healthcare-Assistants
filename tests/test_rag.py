import pandas as pd
import pytest

from app.ai.rag import MedicalRAG, medical_tokenizer


def test_medical_tokenizer_stemming():
    tokens = medical_tokenizer("Patient with allergies and severe coughing")
    assert "allergy" in tokens
    assert "cough" in tokens


def test_rag_normal_retrieval():
    rag = MedicalRAG()
    results = rag.retrieve("What is asthma?", top_k=2)
    assert len(results) > 0
    assert "asthma" in results[0]["question"].lower() or "asthma" in results[0]["answer"].lower()
    assert results[0]["score"] > 0


def test_rag_missing_file_handled_gracefully(tmp_path, monkeypatch):
    non_existent = str(tmp_path / "does_not_exist.csv")
    # Point both target and fallback to non-existent
    rag = MedicalRAG(data_path=non_existent)
    # If sample_medquad exists in workspace, it might fall back to sample;
    # verify retrieve works without unhandled exceptions
    res = rag.retrieve("What is diabetes?")
    assert isinstance(res, list)


def test_rag_malformed_csv_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "malformed.csv"
    bad_csv.write_text("col_a,col_b\nval1,val2\n", encoding="utf-8")

    rag = MedicalRAG(data_path=str(bad_csv))
    assert rag.matrix is None
    assert rag.df is None

    # retrieve should safely return empty list without crashing
    res = rag.retrieve("What is diabetes?")
    assert res == []


def test_rag_empty_csv(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("question,answer\n", encoding="utf-8")

    rag = MedicalRAG(data_path=str(empty_csv))
    assert rag.retrieve("What is diabetes?") == []


def test_rag_build_prompt_with_and_without_docs():
    rag = MedicalRAG()
    docs = [{"question": "What is flu?", "answer": "Viral infection.", "score": 0.9}]
    prompt_with = rag.build_prompt("What is flu?", docs)
    assert "[Source 1]" in prompt_with
    assert "Viral infection" in prompt_with

    prompt_without = rag.build_prompt("Unknown rare question", [])
    assert "No specific reference documents found" in prompt_without

