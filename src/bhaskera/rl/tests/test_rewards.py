from __future__ import annotations

from fractions import Fraction

import pytest

from bhaskera.rl.rewards import (
    extract_boxed, get_reward, gold_answer, math_boxed, register_reward, to_number,
)


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed("so \\boxed{42}") == "42"

    def test_takes_last(self):
        assert extract_boxed("\\boxed{1} then \\boxed{2}") == "2"

    def test_nested_braces(self):
        assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_missing_or_unbalanced(self):
        assert extract_boxed("the answer is 5") is None
        assert extract_boxed("\\boxed{5") is None


class TestToNumber:
    @pytest.mark.parametrize("text,expected", [
        ("42", Fraction(42)), ("1,234", Fraction(1234)), ("$12.50", Fraction(25, 2)),
        ("75%", Fraction(75)), ("\\frac{1}{2}", Fraction(1, 2)), ("3/4", Fraction(3, 4)),
        ("18 \\text{ dollars}", Fraction(18)), ("-7", Fraction(-7)), (" 5. ", Fraction(5)),
    ])
    def test_parses(self, text, expected):
        assert to_number(text) == expected

    @pytest.mark.parametrize("text", ["", "abc", "1/0", "\\frac{1}{0}"])
    def test_rejects(self, text):
        assert to_number(text) is None


class TestMathBoxed:
    def test_correct(self):
        assert math_boxed("reasoning... \\boxed{72}", "72") == 1.0

    def test_wrong_number(self):
        assert math_boxed("\\boxed{71}", "72") == 0.0

    def test_no_box_gets_nothing_even_if_answer_appears(self):
        assert math_boxed("The answer is 72.", "72") == 0.0

    def test_gsm8k_gold_with_hashes(self):
        assert math_boxed("\\boxed{18}", "Janet sells 9 eggs at 2 each.\n#### 18") == 1.0

    def test_numeric_equivalence(self):
        assert math_boxed("\\boxed{1,000}", "1000") == 1.0
        assert math_boxed("\\boxed{0.5}", "1/2") == 1.0

    def test_non_numeric_gold_falls_back_to_exact_string(self):
        assert math_boxed("\\boxed{yes}", "yes") == 1.0
        assert math_boxed("\\boxed{no}", "yes") == 0.0


def test_gold_answer():
    assert gold_answer("blah #### 7") == "7"
    assert gold_answer("  7 ") == "7"


class TestRegistry:
    def test_builtin_available(self):
        assert get_reward("math_boxed") is math_boxed

    def test_custom_registration(self):
        @register_reward("always_one_for_test")
        def r(completion, answer):
            return 1.0
        assert get_reward("always_one_for_test")("x", "y") == 1.0

    def test_unknown_lists_available(self):
        with pytest.raises(ValueError, match="math_boxed"):
            get_reward("nope")
