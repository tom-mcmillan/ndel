from ndel.api import describe_sql_and_python


def test_sql_and_python_lineage_merge() -> None:
    sql = """
SELECT user_id, COUNT(*) AS sessions
FROM users
GROUP BY user_id
"""

    py_source = """
import pandas as pd
df = pd.read_parquet("users")
df = df[df["sessions"] > 1]
"""

    output = describe_sql_and_python(sql, py_source)

    assert "users" in output
    assert "filter" in output or "sessions" in output
    assert "group" in output.lower()
