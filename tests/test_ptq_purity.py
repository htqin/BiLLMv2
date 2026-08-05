import ast
from pathlib import Path


def test_pure_ptq_has_no_optimizer_or_backward_call() -> None:
    source = Path(__file__).resolve().parents[1] / "run_ptq.py"
    tree = ast.parse(source.read_text())
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "backward" not in called_attributes
    assert "Optimizer" not in called_names
    assert "Adam" not in called_names
    assert "AdamW" not in called_names
