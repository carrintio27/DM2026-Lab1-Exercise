import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    """
    Test load_dataset_tool on agent_dev/sample_fixture.csv.
    Asserts known answer 1 from TEST_FIXTURE.md:
    - Exactly 8 documents loaded
    - Category split: {"catA": 4, "catB": 4}
    - session.dataframe populated with 8 rows
    """
    # Create single session state for the pipeline
    session = SessionState()

    # Load premade tools using load_workspace_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Invoke tool with sample fixture CSV
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Assertion 1: Check document count returned by the tool
    assert result["n_documents"] == 8, f"Expected 8 documents, got {result.get('n_documents')}"

    # Assertion 2: Check counts per category returned by the tool
    expected_counts = {"catA": 4, "catB": 4}
    assert result["counts_per_category"] == expected_counts, (
        f"Expected counts {expected_counts}, got {result.get('counts_per_category')}"
    )

    # Assertion 3: Verify session state side-effects
    assert session.dataframe is not None, "session.dataframe should not be None after loading"
    assert len(session.dataframe) == 8, f"Expected session.dataframe length 8, got {len(session.dataframe)}"
