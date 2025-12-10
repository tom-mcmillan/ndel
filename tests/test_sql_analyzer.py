from ndel.api import describe_sql_source
from ndel.sql_analyzer import analyze_sql_source


def test_analyze_sql_source_detects_table_and_where() -> None:
    sql = "SELECT * FROM analytics.users WHERE active = 1"

    pipeline = analyze_sql_source(sql, config=None)

    assert any(ds.name == "analytics.users" for ds in pipeline.datasets)
    assert any(t.kind == "filter" for t in pipeline.transformations)


def test_describe_sql_source_renders_pipeline() -> None:
    sql = "SELECT name FROM users"

    output = describe_sql_source(sql)

    assert "pipeline" in output
    assert "users" in output
