from ndel.py_analyzer import analyze_python_source


def test_detect_common_metrics() -> None:
    source = """
from sklearn.metrics import roc_auc_score, mean_squared_error

y_true = [0, 1]
y_pred = [0.1, 0.9]
auc = roc_auc_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
"""

    pipeline = analyze_python_source(source)

    names = [m.name for m in pipeline.metrics]
    hib_map = {m.name: m.higher_is_better for m in pipeline.metrics}

    assert "roc_auc_score" in names
    assert "mean_squared_error" in names
    assert hib_map.get("roc_auc_score") is True
    assert hib_map.get("mean_squared_error") is False
