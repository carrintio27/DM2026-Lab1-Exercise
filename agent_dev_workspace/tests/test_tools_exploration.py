import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools

def setup_fixture_dtm():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({
        "text_column": "text",
        "ngram_range": [1, 1]
    })
    
    tools = make_tools(session)
    cosine_sim_tool = next(t for t in tools if t.name == "cosine_similarity_tool")
    
    return session, cosine_sim_tool

def test_cosine_similarity_known_answer_16():
    session, cosine_sim_tool = setup_fixture_dtm()
    
    # Pair 1: Doc 0 vs 1 (both catA, near-identical) -> 0.9847
    res01 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 1})
    assert res01["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 2: Doc 0 vs 3 (catA vs catB, share only "always") -> 0.0909
    res03 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 3})
    assert res03["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)
    
    # Pair 3: Doc 3 vs 4 (both catB, near-identical) -> 0.9847
    res34 = cosine_sim_tool.invoke({"doc_index1": 3, "doc_index2": 4})
    assert res34["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 4: Doc 0 vs 7 (catA vs empty text row) -> 0.0
    res07 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 7})
    assert res07["cosine_similarity"] == 0.0

def test_cosine_similarity_out_of_bounds():
    session, cosine_sim_tool = setup_fixture_dtm()
    
    res = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 99})
    assert "error" in res
    assert "out of bounds" in res["error"].lower()

def test_feature_correlation_matrix_known_answer_19():
    session, _ = setup_fixture_dtm()
    tools = make_tools(session)
    feature_corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")
    
    res = feature_corr_tool.invoke({"top_k": 20})
    
    assert "error" not in res
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms"] == expected_terms
    
    expected_matrix = [
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ]
    
    corr_arr = np.array(res["correlation_matrix"])
    assert np.allclose(corr_arr, expected_matrix, atol=1e-3)
    assert session.pending_figure is not None
