"""
Frame selector text, 1-based and inclusive, used by the Image Sequence
selector and the Mesh Template Track's 'Copy selected frames':

    1:30, 61:90, 100        ranges and single frames (',' ';' or spaces between)
    1:1000:10               every 10th frame
    700:   :50              to the last / from the first
    (31-30):(80+50-30)      every number may be an integer expression with
    201 : 480 + 20          + - * and ( ); ':' binds weakest, so 1:2+3 is
                            1:5 (not 4:5). Spaces inside an expression or
                            around an operator do not separate items.

parse_selector(text, total) -> (0-based indices in the given order, number of
requested frames beyond `total`), or (None, 0) for an empty text (= all).
ValueError with a message for the user when the text is malformed.
No eval(): a small parser of integers, + - * and parentheses.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\s*(?:(\d+)|(.))")


def _items(text: str) -> list[str]:
    """Split into items at ',' ';' or whitespace -- but not inside
    parentheses, and not where the whitespace touches an operator or ':'
    ('1 + 2', '201 : 480', '( 3 )' stay one item)."""
    out, cur, depth = [], [], 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("')' without '('")
        if depth == 0 and (ch in ",;" or ch.isspace()):
            j = i
            while j < n and (text[j] in ",;" or text[j].isspace()):
                j += 1
            gap = text[i:j]
            prev = "".join(cur).rstrip()[-1:] if cur else ""
            nxt = text[j:j + 1]
            joins = ("," not in gap and ";" not in gap and
                     (prev in "+-*:(" or nxt in "+-*:)") and prev != "" and nxt != "")
            if not joins and cur:
                out.append("".join(cur))
                cur = []
            elif joins:
                cur.append(" ")
            i = j
            continue
        cur.append(ch)
        i += 1
    if depth > 0:
        raise ValueError("'(' without ')'")
    if cur:
        out.append("".join(cur))
    return [s for s in (x.strip() for x in out) if s]


class _Expr:
    """Integer expression: + - * ( ) and unary minus."""

    def __init__(self, text: str):
        self.toks = []
        pos = 0
        for m in _TOKEN.finditer(text):
            if m.start() != pos and text[pos:m.start()].strip():
                break
            pos = m.end()
            num, op = m.groups()
            if num is not None:
                self.toks.append(int(num))
            elif op is not None and op.strip():
                if op not in "+-*()":
                    raise ValueError(f"'{op}' is not allowed (only numbers, + - * and parentheses)")
                self.toks.append(op)
        self.i = 0
        self.text = text

    def parse(self) -> int:
        if not self.toks:
            raise ValueError("a number is missing")
        v = self._sum()
        if self.i != len(self.toks):
            raise ValueError(f"'{self.text}' is not a valid expression")
        return v

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _sum(self) -> int:
        v = self._prod()
        while self._peek() in ("+", "-"):
            op = self.toks[self.i]
            self.i += 1
            w = self._prod()
            v = v + w if op == "+" else v - w
        return v

    def _prod(self) -> int:
        v = self._unary()
        while self._peek() == "*":
            self.i += 1
            v *= self._unary()
        return v

    def _unary(self) -> int:
        t = self._peek()
        if t == "-":
            self.i += 1
            return -self._unary()
        if t == "+":
            self.i += 1
            return self._unary()
        if t == "(":
            self.i += 1
            v = self._sum()
            if self._peek() != ")":
                raise ValueError(f"')' is missing in '{self.text}'")
            self.i += 1
            return v
        if isinstance(t, int):
            self.i += 1
            return t
        raise ValueError(f"'{self.text}' is not a valid expression")


def evaluate(expr: str) -> int:
    """Value of one integer expression ('(80+50-30)' -> 100)."""
    return _Expr(expr).parse()


def _split_colons(item: str) -> list[str]:
    parts, cur, depth = [], [], 0
    for ch in item:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == ":" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def parse_selector(text: str, total: int) -> tuple[list[int] | None, int]:
    if not (text or "").strip():
        return None, 0
    out: list[int] = []
    dropped = 0
    for item in _items(text.strip()):
        parts = _split_colons(item)
        if len(parts) > 3:
            raise ValueError(f"'{item}': at most a:b:step")
        try:
            if len(parts) == 1:
                a = b = evaluate(parts[0])
                step = 1
            else:
                a = evaluate(parts[0]) if parts[0].strip() else 1
                b = evaluate(parts[1]) if parts[1].strip() else max(total, a)
                step = evaluate(parts[2]) if len(parts) == 3 and parts[2].strip() else 1
        except ValueError as e:
            raise ValueError(f"'{item}': {e}") from None
        if a < 1 or b < 1:
            raise ValueError(f"'{item}': numbers start at 1 (it gives {a}:{b})" if len(parts) > 1
                             else f"'{item}': numbers start at 1 (it gives {a})")
        if step < 1:
            raise ValueError(f"'{item}': step must be 1 or more")
        if a > b:
            raise ValueError(f"'{item}': the first number ({a}) is larger than the last ({b})")
        requested = (b - a) // step + 1
        stop = min(b, total)                  # never build a huge range past the source
        if a <= stop:
            kept = range(a - 1, stop, step)
            out.extend(kept)
            dropped += requested - len(kept)
        else:
            dropped += requested
    return out, dropped
