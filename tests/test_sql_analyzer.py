from ndel.api import describe_sql_source
from ndel.sql_analyzer import analyze_sql_source


def test_analyze_sql_source_detects_join_filter_agg() -> None:
    sql = """
SELECT u.id, COUNT(*) AS sessions
FROM users u
JOIN events e ON u.id = e.user_id
WHERE e.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY u.id
"""

    pipeline = analyze_sql_source(sql, config=None)

    names = [ds.name for ds in pipeline.datasets]
    kinds = [t.kind for t in pipeline.transformations]

    assert "users" in names
    assert "events" in names
    assert "join" in kinds
    assert "filter" in kinds
    assert "aggregation" in kinds


def test_analyze_sql_source_detects_projection() -> None:
    sql = """
SELECT id, revenue * 1.2 AS revenue_adj
FROM revenue
WHERE revenue > 0
"""

    pipeline = analyze_sql_source(sql, config=None)

    kinds = [t.kind for t in pipeline.transformations]
    assert "filter" in kinds
    assert any(t.kind == "feature_engineering" for t in pipeline.transformations)
    assert any(f.name == "revenue_adj" for f in pipeline.features)


def test_describe_sql_source_renders_pipeline() -> None:
    sql = "SELECT name FROM users"

    output = describe_sql_source(sql)

    assert "pipeline" in output
    assert "users" in output
