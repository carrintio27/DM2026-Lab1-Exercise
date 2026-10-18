import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """
    Tests build_dtm_tool against Known Answer 4 in TEST_FIXTURE.md on sample_fixture.csv.
    Asserts:
    - Vocabulary size: 5 terms ['alpha', 'always', 'beta', 'delta', 'gamma']
    - 21 non-zero entries out of 40 total
    - Sparsity percentage: 47.5%
    - session.feature_matrix and session.feature_names populated
    - session.artifacts["count_vectorizer"] populated
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    result = build_dtm_tool.invoke({"text_column": "text", "ngram_range": [1, 1]})

    # Known Answer 4 Assertions
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert result["vocabulary"] == expected_vocab
    assert result["vocabulary_size"] == 5
    assert result["n_documents"] == 8
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Session state assertions
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == expected_vocab
    assert "count_vectorizer" in session.artifacts

    # Error handling check for invalid non-integer or invalid ngram_range
    invalid_result = build_dtm_tool.invoke({"text_column": "text", "ngram_range": [2, 1]})
    assert "error" in invalid_result


def test_term_frequency_tool_known_answer_14():
    """
    Tests term_frequency_tool against Known Answer 14 in TEST_FIXTURE.md on sample_fixture.csv.
    Asserts exact total term frequencies across all 8 documents:
    - alpha: 6
    - always: 7
    - beta: 3
    - delta: 4
    - gamma: 7
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # Must build DTM first
    build_dtm_tool.invoke({"text_column": "text", "ngram_range": [1, 1]})

    result = term_frequency_tool.invoke({})

    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    assert result["total_terms"] == 5
    assert result["term_frequencies"] == expected_freqs


def test_dtm_heatmap_tool_known_answer_21():
    """
    Tests dtm_heatmap_tool against Known Answer 21 in TEST_FIXTURE.md on sample_fixture.csv:
    - Default call (n_terms=20, n_documents=20) returns full 8x5 matrix.
    - Sliced call (n_terms=2, n_documents=3) returns top-left 3x2 corner [[3, 1], [2, 1], [1, 1]].
    - Verifies session.pending_figure is set.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({"text_column": "text", "ngram_range": [1, 1]})

    # Test full matrix slice
    full_result = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})
    assert full_result["n_documents"] == 8
    assert full_result["n_terms"] == 5
    assert full_result["included_terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert full_result["sliced_matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test sliced 3x2 corner
    slice_result = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert slice_result["n_documents"] == 3
    assert slice_result["n_terms"] == 2
    assert slice_result["included_terms"] == ["alpha", "always"]
    assert slice_result["sliced_matrix"] == [[3, 1], [2, 1], [1, 1]]
