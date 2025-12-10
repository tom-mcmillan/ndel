from ndel.py_analyzer import analyze_python_source


def test_chained_query_assign_groupby() -> None:
    source = """
import pandas as pd
df = pd.read_csv("file.csv")
df = (
    df
      .query("x > 0")
      .assign(new_col=lambda d: d["x"] + 1)
      .groupby("g")
      .agg({"new_col": "mean"})
)
"""

    pipeline = analyze_python_source(source)
    kinds = [t.kind for t in pipeline.transformations]
    assert kinds == ["filter", "feature_engineering", "aggregation"]
    outputs = [t.outputs for t in pipeline.transformations]
    inputs = [t.inputs for t in pipeline.transformations]
    assert any("new_col" in out for out in outputs)
    assert all(inp for inp in inputs)


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


def test_detect_concat_join() -> None:
    source = """
import pandas as pd
df1 = pd.read_csv("file1.csv")
df2 = pd.read_csv("file2.csv")
df3 = pd.concat([df1, df2])
"""

    pipeline = analyze_python_source(source)
    kinds = [t.kind for t in pipeline.transformations]
    outputs = [t.outputs for t in pipeline.transformations if t.kind == "join"]
    assert "join" in kinds
    assert any("df3" in out for out in outputs)
