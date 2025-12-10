from ndel.py_analyzer import analyze_python_source


def test_analyze_python_source_detects_dataset_and_model() -> None:
    source = """
import pandas as pd
from xgboost import XGBClassifier

df = pd.read_parquet("s3://bucket/data.parquet")
model = XGBClassifier()
model.fit(df[["a", "b"]], df["y"])
"""

    pipeline = analyze_python_source(source)

    assert pipeline.name == "python_pipeline"
    assert any(ds.name == "df" for ds in pipeline.datasets)
    assert any(m.algorithm_family == "XGBClassifier" for m in pipeline.models)
    assert all(t.inputs for t in pipeline.transformations)
    assert all(t.outputs for t in pipeline.transformations)
