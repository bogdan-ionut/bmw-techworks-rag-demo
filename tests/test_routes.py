from app.api import routes


def test_safe_json_loads_accepts_python_style_dict():
    raw = "{'answer': 'hi', 'top_matches': [{'full_name': 'Alice'}]}"
    parsed = routes._safe_json_loads(raw)

    assert parsed["answer"] == "hi"
    assert parsed["top_matches"][0]["full_name"] == "Alice"


def test_coerce_answer_obj_renames_summary_to_answer():
    # Simulate an LLM response where the summary is under the "summary" key
    llm_output = {
        "summary": "This is a summary.",
        "top_matches": [{"full_name": "Bob"}],
    }

    # The expected output after coercion
    expected = {
        "answer": "This is a summary.",
        "top_matches": [{"full_name": "Bob"}],
        "filters_applied": None,
    }

    # Call the function to coerce the object
    coerced = routes._coerce_answer_obj(
        query="test",
        flt=None,
        sources=[{"full_name": "Bob"}],
        answer_obj=llm_output,
        top_n=1,
    )

    assert "summary" not in coerced
    assert coerced["answer"] == expected["answer"]
    assert coerced["top_matches"] == expected["top_matches"]
