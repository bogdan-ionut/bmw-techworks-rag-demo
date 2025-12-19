from app.rag.prompt import ALLOWED_FILTER_FIELDS, PLANNER_SYSTEM_PROMPT

def test_location_in_allowed_filters():
    assert "location_normalized" in ALLOWED_FILTER_FIELDS

    # Ensure other expected fields are still there
    assert "eyewear_present" in ALLOWED_FILTER_FIELDS
    assert "tech_tokens" in ALLOWED_FILTER_FIELDS
    assert "job_title" in ALLOWED_FILTER_FIELDS

def test_planner_prompt_allows_location_filter():
    prompt = PLANNER_SYSTEM_PROMPT

    # Check for the explicit positive instruction
    assert "**Location**: If the user explicitly asks for a city, use `location_normalized`" in prompt

    # Check that location_normalized is in the list of allowed fields in the prompt text
    assert "location_normalized (string, lowercase)" in prompt

def test_name_in_allowed_filters():
    assert "full_name_normalized" in ALLOWED_FILTER_FIELDS

def test_planner_prompt_allows_name_filter():
    prompt = PLANNER_SYSTEM_PROMPT

    # Check for the explicit positive instruction
    assert "**Name**: If the user asks for a specific person by name, use `full_name_normalized`" in prompt

    # Check that full_name_normalized is in the list of allowed fields in the prompt text
    assert "full_name_normalized (string, lowercase)" in prompt
