from ndel.py_analyzer import analyze_python_source


def test_sklearn_pipeline_detects_transform_and_model() -> None:
    source = """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline(steps=[
    ("scale", StandardScaler()),
    ("clf", LogisticRegression())
])
"""

    pipeline = analyze_python_source(source)

    kinds = [t.kind for t in pipeline.transformations]
    assert "feature_engineering" in kinds
    assert any(m.algorithm_family == "LogisticRegression" for m in pipeline.models)


def test_column_transformer_detects_transformations() -> None:
    source = """
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preproc = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(), ["country"])
])
"""

    pipeline = analyze_python_source(source)

    names = [t.name for t in pipeline.transformations]
    descriptions = [t.description for t in pipeline.transformations]
    assert any("num" in n for n in names)
    assert any("cat" in n for n in names)
    assert any("age" in d or "country" in d for d in descriptions)
