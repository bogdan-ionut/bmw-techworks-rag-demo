
import unittest
from app.rag.service import safe_json_loads

class TestSafeJsonLoads(unittest.TestCase):
    def test_clean_json(self):
        s = '{"answer": "hello"}'
        self.assertEqual(safe_json_loads(s), {"answer": "hello"})

    def test_markdown_json(self):
        s = '```json\n{"answer": "hello"}\n```'
        self.assertEqual(safe_json_loads(s), {"answer": "hello"})

    def test_markdown_json_with_text(self):
        s = 'Here is the result:\n```json\n{"answer": "hello"}\n```\nHope it helps.'
        self.assertEqual(safe_json_loads(s), {"answer": "hello"})

    def test_trailing_comma(self):
        s = '{"answer": "hello",}'
        self.assertEqual(safe_json_loads(s), {"answer": "hello"})

    def test_chain_of_thought_before(self):
        s = 'Thinking: {check constraints}\n{"answer": "hello"}'
        self.assertEqual(safe_json_loads(s), {"answer": "hello"})

    def test_chain_of_thought_valid_json_before(self):
        s = '{"thought": "checking"}\n{"answer": "hello"}'
        self.assertEqual(safe_json_loads(s), {"answer": "hello"})

    def test_chain_of_thought_after(self):
        s = '{"answer": "hello"}\nI checked constraints.'
        self.assertEqual(safe_json_loads(s), {"answer": "hello"})

    def test_nested_braces(self):
        s = '{"answer": "hello {world}"}'
        self.assertEqual(safe_json_loads(s), {"answer": "hello {world}"})

    def test_multiple_candidates_priority(self):
        # Should prefer the one with "answer" key
        s = '{"info": "test"}\n{"answer": "real answer"}\n{"other": "ignore"}'
        self.assertEqual(safe_json_loads(s), {"answer": "real answer"})

if __name__ == '__main__':
    unittest.main()
