from ndel.py_analyzer import analyze_python_source


def test_detect_filter_transformation() -> None:
    source = """
import pandas as pd
df = pd.read_csv("file.csv")
df = df[df["x"] > 0]
"""

    pipeline = analyze_python_source(source)
    kinds = [t.kind for t in pipeline.transformations]
    assert "filter" in kinds


def test_detect_new_column_transformation() -> None:
    source = """
import pandas as pd
df = pd.read_csv("file.csv")
df["y"] = df["x"] * 2
"""

    pipeline = analyze_python_source(source)
    kinds = [t.kind for t in pipeline.transformations]
    outputs = [o for t in pipeline.transformations for o in t.outputs]
    assert "feature_engineering" in kinds
    assert "y" in outputs


def test_detect_groupby_aggregation() -> None:
    source = """
import pandas as pd
df = pd.read_csv("file.csv")
df = df.groupby("a").agg({"b": "sum"})
"""

    pipeline = analyze_python_source(source)
    kinds = [t.kind for t in pipeline.transformations]
    assert "aggregation" in kinds


def test_detect_merge_join() -> None:
    source = """
import pandas as pd
df1 = pd.read_csv("file1.csv")
df2 = pd.read_csv("file2.csv")
df3 = df1.merge(df2, on="id")
"""

    pipeline = analyze_python_source(source)
    kinds = [t.kind for t in pipeline.transformations]
    outputs = [t.outputs for t in pipeline.transformations if t.kind == "join"]
    assert "join" in kinds
    assert any("df3" in out for out in outputs)
