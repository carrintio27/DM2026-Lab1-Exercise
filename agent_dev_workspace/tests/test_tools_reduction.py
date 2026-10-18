import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

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
    reduce_dimensions_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")
    
    return session, reduce_dimensions_tool

def test_reduce_dimensions_pca_known_answer_7():
    session, reduce_dimensions_tool = setup_fixture_dtm()
    
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})
    
    assert res["method"] == "pca"
    assert "explained_variance_ratio" in res
    
    expected_evr = [0.7532, 0.1965]
    assert res["explained_variance_ratio"] == pytest.approx(expected_evr, abs=1e-3)
    
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]
    
    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert session.pending_figure is not None

def test_reduce_dimensions_tsne_known_answer_8():
    session, reduce_dimensions_tool = setup_fixture_dtm()
    
    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    
    assert res["method"] == "tsne"
    
    expected_coords = [
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973]
    ]
    
    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-2)
    assert session.pending_figure is not None

def test_reduce_dimensions_umap_known_answer_9():
    session, reduce_dimensions_tool = setup_fixture_dtm()
    
    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    
    assert res["method"] == "umap"
    
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]
    
    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-1)
    assert session.pending_figure is not None
