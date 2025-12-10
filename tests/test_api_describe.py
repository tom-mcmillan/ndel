from ndel import describe_callable, describe_python_source


def test_describe_python_source_includes_model_and_dataset() -> None:
    source = """
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
model = LogisticRegression()
model.fit(df[["x"]], df["y"])
"""

    ndel_text = describe_python_source(source)

    assert "pipeline" in ndel_text
    assert "df" in ndel_text
    assert "LogisticRegression" in ndel_text


def test_describe_callable_uses_function_source() -> None:
    def _toy_pipeline():
        import pandas as pd
        from sklearn.linear_model import LogisticRegression

        df = pd.read_csv("data.csv")
        model = LogisticRegression()
        model.fit(df[["x"]], df["y"])

    ndel_text = describe_callable(_toy_pipeline)

    assert "LogisticRegression" in ndel_text
    assert "df" in ndel_text
