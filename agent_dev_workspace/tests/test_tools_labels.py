import pandas as pd
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_labels import make_tools

def test_binarize_labels_tool_normal():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    
    assert "error" not in res
    assert res["classes"] == ["catA", "catB"]
    assert res["n_samples"] == 8
    assert res["shape"] == [8, 1]
    assert "binarized_labels" in session.artifacts
    assert "label_binarizer" in session.artifacts

def test_binarize_labels_tool_missing_labels():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["doc1", "doc2", "doc3"],
        "category_name": ["catA", None, "catB"]
    })
    session.labels = ["catA", None, "catB"]
    
    tools = make_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    
    assert "error" in res
    assert "missing" in res["error"].lower()
