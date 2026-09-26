"""The arithmetic tool.

Whisper emits words, not symbols, so most of this file is about turning
"seven times twenty three" into something Python can evaluate -- and about
refusing everything that is not arithmetic.
"""

from __future__ import annotations

import pytest

from server.tools.calc import (
    CalcError,
    evaluate,
    normalize_expression,
    try_fast_path,
)


class TestDigitExpressions:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("7 * 23", 161.0),
            ("144 / 12", 12.0),
            ("2 + 2", 4.0),
            ("10 - 3", 7.0),
            ("9 % 4", 1.0),
            ("2 ** 10", 1024.0),
            ("2 ** 0.5", pytest.approx(1.41421356, rel=1e-6)),
            ("2 + 3 * 4", 14.0),  # precedence
            ("(2 + 3) * 4", 20.0),  # parentheses
            ("-5 + 8", 3.0),
            ("1.5 * 4", 6.0),
        ],
    )
    def test_evaluates(self, expression: str, expected) -> None:
        assert evaluate(expression).value == pytest.approx(expected)


class TestSpokenExpressions:
    @pytest.mark.parametrize(
        "spoken,expected",
        [
            ("7 times 23", 161.0),
            ("twenty three times seven", 161.0),
            ("144 divided by 12", 12.0),
            ("twelve divided by four", 3.0),
            ("2 plus 2", 4.0),
            ("ten minus three", 7.0),
            ("2 to the power of 10", 1024.0),
            ("what is 7 times 23", 161.0),
            ("what is 144 divided by 12", 12.0),
            ("can you calculate 50 percent of 200", 100.0),
            ("what is the square root of 16", 4.0),
            ("square root of 81", 9.0),
            ("what is 5 squared", 25.0),
            ("what is 3 cubed", 27.0),
        ],
    )
    def test_evaluates(self, spoken: str, expected: float) -> None:
        assert evaluate(spoken).value == pytest.approx(expected)


class TestNumberWords:
    @pytest.mark.parametrize(
        "spoken,expected",
        [
            ("one", 1.0),
            ("ten", 10.0),
            ("twenty three", 23.0),
            ("ninety nine", 99.0),
            ("one hundred", 100.0),
            ("two hundred fifty", 250.0),
            ("one thousand", 1000.0),
            ("three million", 3_000_000.0),
            ("twenty three times seven", 161.0),
        ],
    )
    def test_compound_numbers(self, spoken: str, expected: float) -> None:
        assert evaluate(spoken).value == pytest.approx(expected)

    def test_hyphenated_and_comma_forms(self) -> None:
        assert evaluate("twenty-three plus seven").value == pytest.approx(30.0)
        assert evaluate("1,024 plus 1").value == pytest.approx(1025.0)


class TestFunctionsAndConstants:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("sqrt(144)", 12.0),
            ("abs(-5)", 5.0),
            ("round(3.14159, 2)", 3.14),
            ("floor(3.9)", 3.0),
            ("ceil(3.1)", 4.0),
            ("log(1)", 0.0),
            ("log2(1024)", 10.0),
            ("log10(1000)", 3.0),
            ("min(3, 7, 2)", 2.0),
            ("max(3, 7, 2)", 7.0),
            ("pow(2, 8)", 256.0),
        ],
    )
    def test_functions(self, expression: str, expected: float) -> None:
        assert evaluate(expression).value == pytest.approx(expected)

    def test_constants(self) -> None:
        assert evaluate("pi").value == pytest.approx(3.14159265, rel=1e-7)
        assert evaluate("tau").value == pytest.approx(6.2831853, rel=1e-7)

    def test_constants_arithmetic(self) -> None:
        assert evaluate("pi * 2").value == pytest.approx(6.2831853, rel=1e-7)


class TestRejectsUnsafeInput:
    """The evaluator is a tool the LLM can call with anything at all."""

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('id')",
            "().__class__.__bases__",
            "open('/etc/passwd')",
            "os.system('rm -rf /')",
            "[1,2,3]",
            "{'a': 1}",
            "lambda: 1",
            "x = 5",
            "print(1)",
            "1; 2",
            "1 if 1 else 2",
            "1 and 2",
            "a.b.c",
        ],
    )
    def test_rejected(self, expression: str) -> None:
        with pytest.raises(CalcError):
            evaluate(expression)

    def test_no_attribute_access_sneaks_through(self) -> None:
        with pytest.raises(CalcError):
            evaluate("().__class__")

    def test_name_lookup_rejects_unknown_identifiers(self) -> None:
        """An unrecognised word must be refused, never quietly evaluated."""
        with pytest.raises(CalcError, match="cannot interpret word"):
            evaluate("mystery + 1")

    def test_huge_exponent_is_refused(self) -> None:
        with pytest.raises(CalcError, match="exponent"):
            evaluate("2 ** 9999")

    def test_exponent_within_limit_is_allowed(self) -> None:
        assert evaluate("2 ** 50").value == pytest.approx(1.125899906842624e15)

    def test_result_out_of_range_is_refused(self) -> None:
        with pytest.raises(CalcError, match="out of range"):
            evaluate("10 ** 18 * 10")


class TestDisplay:
    def test_integers_have_no_decimal_point(self) -> None:
        assert evaluate("144 / 12").display == "12"
        assert evaluate("7 * 23").display == "161"

    def test_non_integers_keep_precision(self) -> None:
        assert evaluate("10 / 4").display == "2.5"

    def test_float_noise_is_trimmed(self) -> None:
        """0.1 + 0.2 must not be read out as 0.30000000000000004."""
        assert evaluate("0.1 + 0.2").display == "0.3"

    def test_expression_is_echoed_normalised(self) -> None:
        assert evaluate("7 times 23").expression == "7 * 23"
        assert evaluate("7 * 23").expression == "7 * 23"


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("what is 7 times 23", "7 * 23"),
            ("What is 144 divided by 12?", "144 / 12"),
            ("calculate 2 plus 2 please", "2 + 2"),
            ("7 ** 3", "7 ** 3"),
            ("2 ^ 3", "2 ** 3"),
        ],
    )
    def test_known_normalisations(self, raw: str, expected: str) -> None:
        assert normalize_expression(raw) == expected

    def test_implicit_multiplication_is_inserted(self) -> None:
        assert normalize_expression("2 pi") == "2 * pi"

    def test_units_are_dropped(self) -> None:
        assert normalize_expression("5 kilometers") == "5"

    def test_question_scaffolding_is_removed(self) -> None:
        assert normalize_expression("hey friday what is 6 times 7") == "6 * 7"


class TestFastPath:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("what is 7 times 23?", "161"),
            ("What is 144 divided by 12?", "12"),
            ("what is 2 plus 2", "4"),
            ("what is 2 to the power of 10?", "1024"),
            # What the pipeline actually sees: normalise_transcript has already
            # turned the spoken operators into symbols by this point.
            ("what is 7 * 23", "161"),
            ("what is 144 / 12", "12"),
            ("what is 7 * 23 in seconds", "161"),  # the unit is dropped, the maths holds
            ("square root of 144", "12"),
        ],
    )
    def test_hits(self, question: str, expected: str) -> None:
        hit = try_fast_path(question)
        assert hit is not None
        assert hit.display == expected

    @pytest.mark.parametrize(
        "question",
        [
            "what is the weather in Pune?",
            "what time is it?",
            "how much memory does this machine have?",
            "blah blah wubble frobnicate",
            "",
            "tell me a joke",
            # A number, but not arithmetic: the model has to pick the tool.
            "how many people live in 5 million",
            "what is the population of 5 million people",
            # A bare number is not a question.
            "what is 12",
            "7",
        ],
    )
    def test_misses(self, question: str) -> None:
        assert try_fast_path(question) is None
