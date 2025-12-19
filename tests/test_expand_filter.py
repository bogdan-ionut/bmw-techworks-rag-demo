
import pytest
from app.rag.service import expand_name_filter

def test_expand_name_filter_no_change():
    """Test that filters without full_name_normalized are unchanged."""
    f = {"eyewear_present": {"$eq": True}}
    assert expand_name_filter(f) == f

def test_expand_name_filter_simple_eq():
    """Test expansion of full_name_normalized strict equality."""
    f = {"full_name_normalized": {"$eq": "ionut buraga"}}
    expanded = expand_name_filter(f)

    assert "full_name_normalized" in expanded
    assert "$in" in expanded["full_name_normalized"]
    variations = expanded["full_name_normalized"]["$in"]

    assert "ionut buraga" in variations
    assert "Ionut Buraga" in variations
    assert "IONUT BURAGA" in variations

def test_expand_name_filter_nested_and():
    """Test expansion inside $and."""
    f = {
        "$and": [
            {"full_name_normalized": {"$eq": "test name"}},
            {"eyewear_present": {"$eq": True}}
        ]
    }
    expanded = expand_name_filter(f)

    name_filter = expanded["$and"][0]["full_name_normalized"]
    assert "$in" in name_filter
    assert "Test Name" in name_filter["$in"]

    # Check other filter remains untouched
    assert expanded["$and"][1] == {"eyewear_present": {"$eq": True}}

def test_expand_name_filter_handles_none():
    assert expand_name_filter(None) is None
