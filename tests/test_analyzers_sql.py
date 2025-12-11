from src.analysis import analyze_sql_source


def test_analyze_sql_with_sqlglot_parsing():
    sql = """
    SELECT u.id as user_id, count(o.id) as order_count
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE u.active = 1
    GROUP BY u.id
    """
    pipeline = analyze_sql_source(sql)

    ds_names = {ds.name for ds in pipeline.datasets}
    assert "users" in ds_names or "u" in ds_names
    assert any(t.kind == "join" for t in pipeline.transformations)
    assert any(t.kind == "filter" for t in pipeline.transformations)
    assert any(t.kind == "aggregation" for t in pipeline.transformations)

    feature_names = {f.name for f in pipeline.features}
    assert "user_id" in feature_names
    assert "order_count" in feature_names
