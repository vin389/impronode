"""Shared pytest setup.

Tests must never open real dialog boxes: a failing or half-mocked test that
reaches messagebox.show*/ask* would pop a window up on the developer's
screen (and an ask* dialog would block the test run until it is clicked).
Every tkinter.messagebox function is replaced, for every test, by a stub
that records the call; tests can inspect the calls through the `dialogs`
fixture. A test that needs a specific answer still monkeypatches the
function itself (that later patch wins).
"""

import tkinter.messagebox as _messagebox

import pytest

_SHOW = ("showinfo", "showwarning", "showerror")
_ASK = ("askyesno", "askokcancel", "askyesnocancel", "askretrycancel", "askquestion")


@pytest.fixture(autouse=True)
def dialogs(monkeypatch):
    """Record dialog calls instead of showing them. ask* answer "no/cancel"
    (False; "no" for askquestion), so an unexpected question never makes a
    test go ahead with a destructive action."""
    calls = []

    def stub(name, answer):
        def _dialog(*args, **kwargs):
            calls.append((name, args, kwargs))
            return answer
        return _dialog

    for name in _SHOW:
        monkeypatch.setattr(_messagebox, name, stub(name, "ok"))
    for name in _ASK:
        monkeypatch.setattr(_messagebox, name, stub(name, "no" if name == "askquestion" else False))
    return calls
