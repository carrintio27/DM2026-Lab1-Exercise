import json
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool_known_answer_10():
    """
    Tests tokenize_tool against Known Answer 10 in TEST_FIXTURE.md on sample_fixture.csv.
    Checks all 8 rows, including row 7 (empty text).
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
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({"text_column": "text"})

    assert "result_id" in result
    assert result["n_documents"] == 8

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []  # row 7 is empty text
    ]

    actual_tokens = list(session.dataframe["unigrams"])
    assert len(actual_tokens) == 8, f"Expected 8 document token lists, got {len(actual_tokens)}"

    for idx, (actual, expected) in enumerate(zip(actual_tokens, expected_tokens)):
        assert actual == expected, f"Row {idx} token mismatch. Got {actual}, expected {expected}"

    assert session.artifacts["tokens"] == expected_tokens


def test_check_missing_tool_known_answer_2():
    """
    Tests check_missing_tool against Known Answer 2 in TEST_FIXTURE.md on sample_fixture.csv.
    Asserts exactly 1 missing row at index 7.
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
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({"text_column": "text"})

    assert result["missing_count"] == 1, f"Expected missing_count 1, got {result.get('missing_count')}"
    assert result["missing_indices"] == [7], f"Expected missing_indices [7], got {result.get('missing_indices')}"
    assert result["total_documents"] == 8


def test_check_duplicates_tool_known_answer_3_no_drop():
    """Tests check_duplicates_tool without dropping (drop=False)."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    result = check_duplicates_tool.invoke({"text_column": "text", "drop": False})

    assert result["duplicate_count"] == 1
    assert result["dropped"] is False
    assert result["n_remaining_documents"] == 8
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_known_answer_3_with_drop():
    """Tests check_duplicates_tool with dropping (drop=True, keep=False)."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    result = check_duplicates_tool.invoke({"text_column": "text", "drop": True})

    assert result["duplicate_count"] == 1
    assert result["dropped"] is True
    # Crucial check: keep=False drops BOTH copies, leaving 6 rows out of 8
    assert result["n_remaining_documents"] == 6
    assert len(session.dataframe) == 6

    # Specific check: confirm that 'always gamma delta' is completely gone from session.dataframe
    remaining_texts = list(session.dataframe["text"])
    assert "always gamma delta" not in remaining_texts, (
        "'always gamma delta' should be completely removed under keep=False"
    )


def test_list_files_tool_known_answer_11():
    """
    Tests list_files_tool against Known Answer 11 in TEST_FIXTURE.md on subdirectory 'newdataset'.
    Also tests path traversal security rejection.
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Invoke with Known Answer 11 path
    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
    assert result["n_directories"] == 0
    assert result["n_files"] == 1

    # Security check: path traversal rejection
    traversal_result = list_files_tool.invoke({"subdirectory": "../"})
    assert "error" in traversal_result
    assert "Access denied" in traversal_result["error"]


def test_sample_data_tool_known_answer_13():
    """
    Tests sample_data_tool against Known Answer 13 in TEST_FIXTURE.md on sample_fixture.csv.
    Asserts n=4, random_state=42 returns sampled indices [1, 5, 0, 7].
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
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["sample_size"] == 4
    assert result["original_size"] == 8
    assert len(session.dataframe) == 4


def test_describe_data_tool_known_answer_18():
    """
    Tests describe_data_tool against Known Answer 18 in TEST_FIXTURE.md on sample_fixture.csv.
    Asserts text lengths, overall stats (including quartiles), per-category stats (min/max/quartiles), and pending figure generation.
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
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({"text_column": "text", "label_column": "category_name"})

    # 1. Text lengths check
    assert result["text_lengths"] == [29, 23, 17, 30, 24, 18, 18, 0]

    # 2. Overall stats check (including 25%, 50%, 75%)
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # 3. By-category stats check (including min/max/quartiles)
    by_cat = result["by_category_stats"]
    assert "catA" in by_cat
    assert "catB" in by_cat

    # catA assertions
    catA = by_cat["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29

    # catB assertions
    catB = by_cat["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30

    # 4. Matplotlib figure check
    assert session.pending_figure is not None


def test_inspect_data_tool_custom():
    """
    Tests inspect_data_tool custom behavior on sample_fixture.csv:
    - Verifies row count (8) and column names.
    - Verifies preview rows count for n=8.
    - Verifies empty/missing text at row 7 is converted safely to None (JSON null).
    - Verifies strict json.dumps(result, allow_nan=False) succeeds without error.
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
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Call with n=8 to include the empty text row
    result = inspect_tool.invoke({"n": 8})

    assert result["total_rows"] == 8
    assert "text" in result["columns"]
    assert result["preview_rows_count"] == 8

    # Row index 7 has empty/missing text; verify it is None rather than float('nan')
    empty_row_text = result["head"][7]["text"]
    assert empty_row_text is None or empty_row_text == ""

    # Strict JSON serialization check: allow_nan=False throws ValueError if any raw NaN exists
    json_str = json.dumps(result, allow_nan=False)
    assert isinstance(json_str, str)
