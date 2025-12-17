from pathlib import Path

from app import main


def test_resolve_web_dir(tmp_path: Path):
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("hello")

    assert main.resolve_web_dir(tmp_path) == web_dir


def test_cohere_reranker_skips_when_no_key():
    assert main.init_cohere_reranker("", model="", top_n=5) is None
