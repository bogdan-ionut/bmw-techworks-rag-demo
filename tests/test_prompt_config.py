from app.rag.prompt import ALLOWED_FILTER_FIELDS, PLANNER_SYSTEM_PROMPT

def test_location_not_in_allowed_filters():
    assert "location" not in ALLOWED_FILTER_FIELDS
    assert "location_normalized" not in ALLOWED_FILTER_FIELDS

    # Ensure other expected fields are still there
    assert "eyewear_present" in ALLOWED_FILTER_FIELDS
    assert "tech_tokens" in ALLOWED_FILTER_FIELDS

def test_planner_prompt_forbids_location_filter():
    prompt = PLANNER_SYSTEM_PROMPT

    # Check for the explicit negative instruction
    assert "Do NOT filter by location" in prompt
    assert "people from Berlin" in prompt

    # Check that location_normalized (string) is removed from the list
    # The original list had "location_normalized (string)"
    assert "location_normalized (string)" not in prompt
