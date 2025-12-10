import ast

from ndel.py_analyzer import analyze_python_source
from ndel.semantic_model import Dataset, Transformation


def test_custom_detector_adds_dataset() -> None:
    def detector(ctx, node):  # type: ignore[override]
        if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "special_df":
                ctx.datasets[node.targets[0].id] = Dataset(name="special_df")

    source = """
special_df = 1
"""

    pipeline = analyze_python_source(source, custom_detectors=[detector])

    assert any(ds.name == "special_df" for ds in pipeline.datasets)


def test_custom_detector_adds_transformation() -> None:
    def detector(ctx, node):  # type: ignore[override]
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "custom_op":
                ctx.transformations.append(
                    Transformation(
                        name="custom_op",
                        description="custom detector transformation",
                        kind="other",
                        inputs=[],
                        outputs=[],
                    )
                )

    source = """
def run():
    custom_op()
"""

    pipeline = analyze_python_source(source, custom_detectors=[detector])

    assert any(t.name == "custom_op" for t in pipeline.transformations)
