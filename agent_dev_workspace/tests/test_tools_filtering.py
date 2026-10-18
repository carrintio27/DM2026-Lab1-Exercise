import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools

def test_variance_filter_tool_known_answer_5():
    # 1. Setup a single session instance
    session = SessionState()

    # 2. Load dataset with premade tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM using student's build_dtm_tool
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text", "ngram_range": [1, 1]})

    # 4. Invoke variance_filter_tool
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # 5. Assert known answer 5 numerical values (variances)
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375
    }

    for term, expected_var in expected_variances.items():
        assert term in res["variances"]
        assert res["variances"][term] == pytest.approx(expected_var, abs=1e-4)

    # 6. Assert kept vs removed with threshold 0.15
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_terms"] == ["always"]

    # 7. Verify session.feature_matrix shape is untouched
    assert session.feature_matrix.shape == (8, 5)


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text", "ngram_range": [1, 1]})

    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    # Known answer 6 values for target_class="catB"
    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307
    }

    for term, expected_r in expected_pearson.items():
        assert term in res["correlations"]
        assert res["correlations"][term] == pytest.approx(expected_r, abs=1e-3)

    assert set(res["kept_terms"]) == {"delta", "gamma"}
    assert set(res["removed_terms"]) == {"alpha", "always", "beta"}


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text", "ngram_range": [1, 1]})

    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    # Known answer 6 values for target_class="catB" (Spearman)
    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363
    }

    for term, expected_rho in expected_spearman.items():
        assert term in res["correlations"]
        assert res["correlations"][term] == pytest.approx(expected_rho, abs=1e-3)

    assert set(res["kept_terms"]) == {"delta", "gamma"}
    assert set(res["removed_terms"]) == {"alpha", "always", "beta"}
