import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def setup_fixture():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_tool

def test_mine_patterns_fpgrowth_known_answer_17():
    session, mine_tool = setup_fixture()
    
    # 1. Zero terms kept with filtering_method="variance"
    res_var = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["patterns"] == []
    assert "no terms survived filtering" in res_var["note"]
    
    # 2. tfidf filtering + fpgrowth for catA -> ['alpha'] support 3
    res_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3
    assert "patterns_catA_tfidf_fpgrowth" in session.artifacts
    
    # 3. tfidf filtering + fpgrowth for catB -> ['gamma'] support 4
    res_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4
    assert "patterns_catB_tfidf_fpgrowth" in session.artifacts

def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    session, mine_tool = setup_fixture()
    
    # topk algorithm with k=1 for catA -> ['alpha'] support 3
    res_topk = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk["patterns"]) == 1
    assert res_topk["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk["patterns"][0]["support"] == 3
    
    # maxfpgrowth algorithm for catA -> returns 0 patterns (empty result expected)
    res_max = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max["patterns"] == []

def test_mine_patterns_parameter_validation():
    session, mine_tool = setup_fixture()
    
    # Missing min_sup for fpgrowth
    res1 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert "error" in res1
    assert "positive integer" in res1["error"].lower() or "min_sup" in res1["error"].lower()

    # min_sup=0 and min_sup=-1 for fpgrowth
    for invalid_sup in [0, -1]:
        res = mine_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": invalid_sup
        })
        assert "error" in res
        assert "positive integer" in res["error"].lower()
    
    # Missing k for topk
    res2 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert "error" in res2
    assert "positive integer" in res2["error"].lower() or "k" in res2["error"].lower()

    # k=0 and k=-1 for topk
    for invalid_k in [0, -1]:
        res = mine_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": invalid_k
        })
        assert "error" in res
        assert "positive integer" in res["error"].lower()
