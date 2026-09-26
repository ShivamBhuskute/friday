"""Safe arithmetic evaluation.

Speech-to-text emits words far more often than symbols ("7 times 23", "twelve
divided by 4"), so this module runs a real tokeniser that understands number
words, compound numbers ("twenty three" -> 23) and spoken operators, then
evaluates the result with an AST whitelist. ``eval`` is never used.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

# --------------------------------------------------------------- vocabulary

NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "nought": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    # Frequent Whisper mishearings of digits.
    "won": "1", "to": "2", "too": "2", "for": "4", "fore": "4",
    "ate": "8", "sicks": "6", "won't": "1",
}

# Longest phrases first so "multiplied by" beats "multiplied".
OPERATOR_PHRASES: dict[str, str] = {
    "multiplied by": "*",
    "divided by": "/",
    "raised to the power of": "**",
    "raised to the power": "**",
    "to the power of": "**",
    "to the power": "**",
    "subtracted from": "-",
    "plus or minus": "+",
    "modulo": "%",
    "remainder": "%",
    "plus": "+",
    "minus": "-",
    "times": "*",
    "multiplied": "*",
    "divided": "/",
    "over": "/",
    "power": "**",
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "percent of": "* 0.01",
}

# Phrases that attach to the value on their left ("7 squared", "the square
# root of 16" said after the number). Applied to whichever side has a value.
POSTFIX_PHRASES: dict[str, str] = {
    "the square root of": "** 0.5",
    "square root of": "** 0.5",
    "squared": "** 2",
    "cubed": "** 3",
}

# Question scaffolding that carries no arithmetic meaning.
FILLER_WORDS = frozenset({
    "what", "whats", "what's", "is", "are", "was", "were", "the", "a", "an",
    "please", "tell", "me", "much", "many", "equals", "equal", "calculate",
    "compute", "how", "of", "to", "in", "then", "give", "us", "hey", "friday",
    "ok", "okay", "just", "do", "you", "i", "we", "can", "could", "would",
    "result", "answer", "value", "total", "sum", "product",
})

UNITS = frozenset({
    "degrees", "degree", "celsius", "fahrenheit", "percent", "%", "dollars",
    "rupees", "meters", "metres", "kilometers", "kilometres", "miles", "feet",
    "inches", "seconds", "minutes", "hours", "days", "years", "people",
    "times", "units", "square", "cubic", "cm", "km", "kg", "g", "m", "s", "l",
})

SCALES: dict[str, int] = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "min": min,
    "max": max,
    "pow": pow,
}

# (min_args, max_args); ``None`` for max means "no upper bound".
FUNCTIONS_ARITY: dict[str, tuple[int, int | None]] = {
    "round": (1, 2),
    "pow": (2, 2),
    "min": (1, None),
    "max": (1, None),
}

MAX_EXPONENT = 128
MAX_ABS_VALUE = 1e18
MAX_POW_BASE = 1e9

# token kinds
_NUM, _WORD, _SYM = "num", "word", "sym"

_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)|([a-z][a-z0-9]*)|([+\-*/%()^,])", re.IGNORECASE)
_STRIP_RE = re.compile(r"[^\w.+\-*/%()^,]")

# "1,024" is one number; "round(3.14, 2)" is two arguments. Only the former has
# exactly three digits following the comma.
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# A hyphen between two letters joins a compound number ("twenty-three"); a
# hyphen between digits is subtraction, which speech renders as "minus".
_HYPHEN_RE = re.compile(r"(?<=[a-zA-Z])-(?=[a-zA-Z])")

# Characters with no meaning in arithmetic. A string containing one of these is
# not an arithmetic question, and silently reinterpreting "[1,2,3]" as
# "1 * 2 * 3" would be a confident wrong answer.
_REJECT_RE = re.compile(r"[\[\]{};:'\"\\!~|&<>@`]")
# Boolean keywords look like arithmetic but are not: "1 and 2" is not 2.
# "is" and "in" are absent on purpose: both are far more often connective than
# boolean in speech ("what is 7 in seconds"), and FILLER_WORDS already has "is".
_REJECT_WORDS = frozenset({"and", "or", "not", "if", "else", "elif"})


class CalcError(ValueError):
    """Anything that is not a safe, evaluable arithmetic expression."""


@dataclass(slots=True)
class CalcResult:
    expression: str
    value: float
    display: str


# -------------------------------------------------------------- tokenising


def _tokenize(text: str) -> list[tuple[str, str]]:
    # A hyphen between letters is a compound-number joiner; a comma between
    # digit groups is a thousands separator. Both must go before tokenising.
    text = _HYPHEN_RE.sub(" ", text)
    text = _THOUSANDS_RE.sub("", text)
    cleaned = _STRIP_RE.sub(" ", text)
    toks: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(cleaned):
        num, word, sym = m.groups()
        if num is not None:
            toks.append((_NUM, num))
        elif word is not None:
            toks.append((_WORD, word.lower()))
        else:
            if sym == "^":
                sym = "**"
            # The regex sees one character at a time, so "**" arrives as two
            # "*" tokens. Rejoin them or every power of becomes a syntax error.
            if toks and toks[-1] == (_SYM, sym) and sym in {"*", "/"}:
                toks[-1] = (_SYM, sym * 2)
            else:
                toks.append((_SYM, sym))
    return toks


def _match_operator(tokens: list[tuple[str, str]], i: int) -> tuple[str, int] | None:
    """Longest-match a spoken operator phrase starting at ``i``."""
    for n in range(min(5, len(tokens) - i), 0, -1):
        if all(tokens[i + k][0] == _WORD for k in range(n)):
            phrase = " ".join(tokens[i + k][1] for k in range(n))
            if phrase in OPERATOR_PHRASES:
                return OPERATOR_PHRASES[phrase], i + n
    return None


def _match_postfix(tokens: list[tuple[str, str]], i: int) -> tuple[str, int] | None:
    """Longest-match a postfix phrase ("squared", "square root of") at ``i``."""
    for n in range(min(4, len(tokens) - i), 0, -1):
        if all(tokens[i + k][0] == _WORD for k in range(n)):
            phrase = " ".join(tokens[i + k][1] for k in range(n))
            if phrase in POSTFIX_PHRASES:
                return POSTFIX_PHRASES[phrase], i + n
    return None


def _match_number(tokens: list[tuple[str, str]], i: int) -> tuple[str, int] | None:
    """Parse a (possibly compound) number word starting at ``i``.

    Handles the shapes speech actually produces: "three", "twenty three",
    "one hundred", "two hundred and fifty", "one thousand five hundred",
    "three million". Returns ``(literal, next_index)`` or ``None``.
    """
    if tokens[i][0] != _WORD:
        return None
    head = tokens[i][1]
    if head in FUNCTIONS or head in CONSTANTS or head in FILLER_WORDS or head in UNITS:
        return None

    total = 0  # scale groups already committed ("three million" -> 3000000)
    group = 0  # the current sub-thousand group ("two hundred fifty" -> 250)
    seen = False
    j = i
    while j < len(tokens) and tokens[j][0] == _WORD:
        word = tokens[j][1]
        if word == "and":  # "two hundred and fifty"
            j += 1
            continue
        if word in SCALES:
            scale = SCALES[word]
            if scale == 100:
                group = (group or 1) * 100
            else:
                total += (group or 1) * scale
                group = 0
            seen = True
            j += 1
            continue
        literal = NUMBER_WORDS.get(word)
        if literal is None:
            break  # an operator, filler or function name ends the number
        group += int(literal)
        seen = True
        j += 1

    if not seen:
        return None
    return str(total + group), j


def normalize_expression(text: str) -> str:
    """Turn a spoken phrase into a symbolic arithmetic expression.

    >>> normalize_expression("7 times 23")
    '7 * 23'
    >>> normalize_expression("what is twelve divided by 4")
    '12 / 4'
    """
    if _REJECT_RE.search(text or ""):
        bad = _REJECT_RE.search(text or "").group()  # type: ignore[union-attr]
        raise CalcError(f"character {bad!r} is not valid in an arithmetic expression")

    tokens = _tokenize(text or "")
    out: list[str] = []
    # A postfix phrase seen before any value ("square root of 16") waits here
    # until the value it applies to arrives.
    pending: list[str] | None = None
    depth = 0  # parenthesis nesting; a comma only separates arguments inside
    i = 0
    while i < len(tokens):
        kind, val = tokens[i]

        if kind == _NUM:
            out.append(val)
            if pending:
                out.extend(pending)
                pending = None
            i += 1
            continue

        if kind == _SYM:
            if val == "(":
                depth += 1
            elif val == ")":
                depth = max(0, depth - 1)
            elif val == ",":
                if depth == 0:
                    # A comma outside brackets is prose punctuation.
                    i += 1
                    continue
            out.append(val)
            i += 1
            continue

        if op := _match_postfix(tokens, i):
            sym, nxt = op
            if out and _ends_value(out[-1]):
                out.extend(sym.split())
            else:
                pending = sym.split()
            i = nxt
            continue

        if op := _match_operator(tokens, i):
            sym, nxt = op
            out.extend(sym.split())
            i = nxt
            continue

        if num := _match_number(tokens, i):
            literal, nxt = num
            out.append(literal)
            if pending:
                out.extend(pending)
                pending = None
            i = nxt
            continue

        if val in FILLER_WORDS or val in UNITS:
            i += 1
            continue

        if val in CONSTANTS or val in FUNCTIONS:
            out.append(val)
            i += 1
            continue

        if val in _REJECT_WORDS:
            raise CalcError(f"{val!r} is not an arithmetic operator")

        # Unknown word: refuse rather than silently mangling the expression.
        raise CalcError(f"cannot interpret word {val!r} in an arithmetic expression")

    if pending:
        raise CalcError("expression ends with an operator and no value")

    return " ".join(_join_implicit_products(out))


def _join_implicit_products(tokens: list[str]) -> list[str]:
    """Insert the ``*`` that spoken forms leave out.

    "50 percent of 200" normalises to ``50 * 0.01 200``; the multiplier needs
    an explicit join before the trailing operand.
    """
    out: list[str] = []
    for tok in tokens:
        if out and _ends_value(out[-1]) and _starts_value(tok):
            out.append("*")
        out.append(tok)
    return out


def _ends_value(tok: str) -> bool:
    if tok == ")":
        return True
    try:
        float(tok)
    except ValueError:
        return tok in CONSTANTS
    return True


def _starts_value(tok: str) -> bool:
    if tok == "(":
        return True
    try:
        float(tok)
    except ValueError:
        # A name is only a value when it is a constant; a function needs parens.
        return tok in CONSTANTS
    return True


# --------------------------------------------------------------- evaluation

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Call,
)

_BIN_OPS: dict[type, object] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPS: dict[type, object] = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}


def evaluate(expression: str) -> CalcResult:
    """Safely evaluate an arithmetic expression and format the result."""
    expr = normalize_expression(expression)
    if not expr:
        raise CalcError("no arithmetic expression found")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"could not parse expression {expr!r}: {exc.msg}") from exc
    value = _eval_node(tree)
    return CalcResult(expression=expr, value=value, display=format_value(value))


def _eval_node(node: ast.AST) -> float:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise CalcError(f"disallowed expression element: {type(child).__name__}")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalcError(f"unsupported constant: {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalcError(f"unknown name: {node.id!r}")
        return CONSTANTS[node.id]

    if isinstance(node, ast.Call):
        return _eval_call(node)

    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise CalcError(f"disallowed unary operator: {type(node.op).__name__}")
        return float(fn(_eval_node(node.operand)))  # type: ignore[operator]

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        fn = _BIN_OPS.get(op_type)
        if fn is None:
            raise CalcError(f"disallowed operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT:
                raise CalcError(f"exponent too large (max {MAX_EXPONENT})")
            if abs(left) > MAX_POW_BASE:
                raise CalcError("base too large to exponentiate")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise CalcError("division by zero")
        result = float(fn(left, right))  # type: ignore[operator]
        if not math.isfinite(result) or abs(result) > MAX_ABS_VALUE:
            raise CalcError("result out of range")
        return result

    raise CalcError(f"unsupported node: {type(node).__name__}")


def _eval_call(node: ast.Call) -> float:
    if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
        raise CalcError("only whitelisted math functions may be called")
    if node.keywords:
        raise CalcError("keyword arguments are not supported")
    name = node.func.id
    args = [_eval_node(a) for a in node.args]
    lo, hi = FUNCTIONS_ARITY.get(name, (1, 1))
    if len(args) < lo or (hi is not None and len(args) > hi):
        want = f"at least {lo}" if hi is None else (f"exactly {lo}" if lo == hi else f"{lo} to {hi}")
        raise CalcError(f"{name}() takes {want} argument(s), got {len(args)}")
    call_args: list[Any] = list(args)
    if name == "round" and len(call_args) == 2:
        # round() rejects a float digit count, and we evaluate everything as float.
        call_args[1] = int(call_args[1])
    try:
        return float(FUNCTIONS[name](*call_args))
    except ValueError as exc:
        raise CalcError(str(exc)) from exc
    except OverflowError as exc:
        raise CalcError("numeric overflow") from exc


# ------------------------------------------------------------------ helpers


def format_value(value: float) -> str:
    """Render a result the way a person would say it: 161, not 161.0."""
    if not math.isfinite(value):
        return str(value)
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e15:
        return str(int(round(value)))
    return f"{round(value, 10):.10f}".rstrip("0").rstrip(".")


_NON_MATH = re.compile(
    r"\b(weather|temperature|forecast|news|who|where|when|why|convert|"
    r"currency|stock|define|explain|translate|play|open|search)\b",
    re.IGNORECASE,
)
_HAS_NUMBER = re.compile(
    r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion)\b",
    re.IGNORECASE,
)
# Only characters a normalised expression can legitimately contain.
_PURE_EXPR = re.compile(r"^[\w\s+\-*/%().^,]+$")


def is_arithmetic_question(text: str) -> bool:
    """Heuristic gating the deterministic fast path.

    The question is "does this whole phrase reduce to an expression?", not "does
    it contain the word times?". Transcripts reach here already rewritten into
    symbols -- ``normalize_transcript`` turns "seven times twenty three" into
    "7 * 23" -- so a keyword list would miss the very case the fast path exists
    to serve. Asking the tokeniser also means an unrecognised word ("population
    of") is rejected instead of being quietly dropped.
    """
    s = (text or "").strip()
    if not s or len(s) > 140:
        return False
    if _NON_MATH.search(s):
        return False
    if not _HAS_NUMBER.search(s):
        return False
    try:
        expr = normalize_expression(s)
    except CalcError:
        return False
    if not expr or not _PURE_EXPR.match(expr):
        return False
    # A lone number is not a question worth bypassing the model for; an
    # expression with an operator in it is.
    return bool(re.search(r"[+\-*/%^]", expr))


def try_fast_path(text: str) -> CalcResult | None:
    """Exact answer for unambiguous arithmetic, else ``None``."""
    if not is_arithmetic_question(text):
        return None
    try:
        return evaluate(text)
    except CalcError:
        return None


def exact_arithmetic(text: str) -> float | None:
    """Evaluate with exact rationals so float drift cannot change the answer."""
    try:
        expr = normalize_expression(text)
    except CalcError:
        return None
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    try:
        return float(_eval_exact(tree))
    except (CalcError, ZeroDivisionError, OverflowError, ValueError):
        return None


def _eval_exact(node: ast.AST) -> Fraction:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise CalcError("disallowed element")
        if isinstance(child, ast.Call):
            raise CalcError("functions are not supported in exact mode")
    if isinstance(node, ast.Expression):
        return _eval_exact(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalcError("unsupported constant")
        return Fraction(node.value).limit_denominator(10**12)
    if isinstance(node, ast.Name):
        raise CalcError("names are not supported in exact mode")
    if isinstance(node, ast.UnaryOp):
        v = _eval_exact(node.operand)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):
        left = _eval_exact(node.left)
        right = _eval_exact(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise CalcError("exponent too large")
            return left**right
    raise CalcError("unsupported node")
