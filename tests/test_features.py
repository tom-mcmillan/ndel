from ndel.py_analyzer import analyze_python_source


def test_feature_from_column_assignment() -> None:
    source = """
import pandas as pd
df = pd.read_csv("file.csv")
df["new_col"] = df["a"] + 1
"""

    pipeline = analyze_python_source(source)

    names = [f.name for f in pipeline.features]
    origins = [f.origin for f in pipeline.features]
    assert "new_col" in names
    assert "df" in origins


def test_features_from_fit_args() -> None:
    source = """
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("file.csv")
model = LogisticRegression()
model.fit(df[["a", "b"]], df["y"])
"""

    pipeline = analyze_python_source(source)

    names = [f.name for f in pipeline.features]
    assert "a" in names and "b" in names
