from app.api import routes


def test_safe_json_loads_accepts_python_style_dict():
    raw = "{'answer': 'hi', 'top_matches': [{'full_name': 'Alice'}]}"
    parsed = routes._safe_json_loads(raw)

    assert parsed["answer"] == "hi"
    assert parsed["top_matches"][0]["full_name"] == "Alice"
