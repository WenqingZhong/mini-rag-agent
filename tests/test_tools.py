"""
Unit tests for llm/tools.py

Tests every branch of execute_tool() and validates TOOLS schema structure.
No external calls — tools are pure local logic.
"""

import re
from llm.tools import execute_tool, TOOLS


class TestExecuteToolDatetime:
    def test_returns_utc_string(self):
        result = execute_tool("get_current_datetime", {})
        # Format: "YYYY-MM-DD HH:MM:SS UTC"
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", result)

    def test_no_args_required(self):
        # Should not raise even with empty dict
        result = execute_tool("get_current_datetime", {})
        assert isinstance(result, str)


class TestExecuteToolCalculate:
    def test_addition(self):
        assert execute_tool("calculate", {"expression": "1 + 1"}) == "2"

    def test_multiplication(self):
        assert execute_tool("calculate", {"expression": "6 * 7"}) == "42"

    def test_complex_expression(self):
        assert execute_tool("calculate", {"expression": "(100 + 50) * 2"}) == "300"

    def test_float_result(self):
        assert execute_tool("calculate", {"expression": "10 / 4"}) == "2.5"

    def test_invalid_expression_returns_error_string(self):
        result = execute_tool("calculate", {"expression": "not_a_number + 1"})
        assert result.startswith("Error evaluating")

    def test_no_builtins_allowed(self):
        # __import__ should not be available inside eval
        result = execute_tool("calculate", {"expression": "__import__('os')"})
        assert result.startswith("Error evaluating")

    def test_missing_expression_key(self):
        # args.get("expression", "") falls back to empty string
        result = execute_tool("calculate", {})
        assert "Error" in result


class TestExecuteToolCountWords:
    def test_basic_count(self):
        assert execute_tool("count_words", {"text": "hello world"}) == "2"

    def test_single_word(self):
        assert execute_tool("count_words", {"text": "hello"}) == "1"

    def test_empty_string(self):
        assert execute_tool("count_words", {"text": ""}) == "0"

    def test_extra_whitespace(self):
        # str.split() handles multiple spaces
        assert execute_tool("count_words", {"text": "one  two   three"}) == "3"

    def test_missing_text_key(self):
        # args.get("text", "") falls back to empty string
        assert execute_tool("count_words", {}) == "0"


class TestExecuteToolUnknown:
    def test_unknown_tool_name(self):
        result = execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result


class TestToolsSchema:
    def test_tools_is_a_list(self):
        assert isinstance(TOOLS, list)

    def test_three_tools_defined(self):
        assert len(TOOLS) == 3

    def test_each_tool_has_required_fields(self):
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_tool_names(self):
        names = {t["function"]["name"] for t in TOOLS}
        assert names == {"get_current_datetime", "calculate", "count_words"}
