"""Tests for Python -> NDEL translation."""

import pytest
from ndel import describe, translate, DomainConfig


class TestTranslate:
    def test_simple_select(self):
        code = 'pd.read_sql("SELECT name, age FROM users", conn)'
        result = translate(code)
        assert "FIND name, age FROM users" in result

    def test_select_with_where(self):
        code = '''
        query = """
        SELECT player_name, goals
        FROM agg_player_season
        WHERE team_name = 'Spirit' AND season_year = 2025
        """
        df = pd.read_sql(query, conn)
        '''
        result = translate(code)
        assert "FIND player_name, goals FROM agg_player_season" in result
        assert "WHERE" in result
        assert "Spirit" in result

    def test_order_by_becomes_rank_by(self):
        code = '''
        query = """
        SELECT name FROM players
        ORDER BY goals DESC
        LIMIT 10
        """
        '''
        result = translate(code)
        assert "RANK BY goals DESC" in result
        assert "LIMIT 10" in result

    def test_pandas_computed_column(self):
        code = """
        df['goals_per_90'] = df['goals'] / df['minutes'] * 90
        """
        result = translate(code)
        assert "COMPUTE goals_per_90" in result

    def test_pandas_sort_values(self):
        code = "df = df.sort_values('points', ascending=False)"
        result = translate(code)
        assert "RANK BY points DESC" in result

    def test_invalid_code_returns_fallback(self):
        result = translate("this is not valid python {{{")
        assert result == "Analysis performed"

    def test_no_sql_returns_fallback(self):
        result = translate("x = 1 + 2")
        assert result == "Analysis performed"


class TestDescribe:
    def test_without_domain(self):
        code = 'pd.read_sql("SELECT id FROM internal_table", conn)'
        result = describe(code)
        assert "internal_table" in result

    def test_with_domain(self):
        code = 'pd.read_sql("SELECT id FROM internal_table", conn)'
        domain: DomainConfig = {
            "tables": {"internal_table": "My Data"}
        }
        result = describe(code, domain=domain)
        assert "My Data" in result
        assert "internal_table" not in result

    def test_column_replacement(self):
        code = 'pd.read_sql("SELECT user_id, created_at FROM events", conn)'
        domain: DomainConfig = {
            "columns": {"user_id": "user", "created_at": "date"}
        }
        result = describe(code, domain=domain)
        assert "user" in result
        assert "date" in result

    def test_never_throws(self):
        # Should not raise, should return fallback
        result = describe(None)  # type: ignore
        assert result == "Analysis performed"
