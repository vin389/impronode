import pytest

from node_editor.frame_selector import evaluate, parse_selector
from node_editor.nodes.image_sequence_node import ImageSequenceNode


def one_based(text, total=10_000):
    sel, _d = parse_selector(text, total)
    return None if sel is None else [k + 1 for k in sel]


def test_plain_syntax_unchanged():
    assert one_based("1:3, 6:7 9") == [1, 2, 3, 6, 7, 9]
    assert one_based("1:10:4") == [1, 5, 9]
    assert one_based("8:", 10) == [8, 9, 10]
    assert one_based(":3") == [1, 2, 3]
    assert one_based("") is None and one_based("   ") is None
    assert parse_selector("5:20", 10) == ([4, 5, 6, 7, 8, 9], 10)


@pytest.mark.parametrize("expr, value", [("1+2*3", 7), ("(1+2)*3", 9), ("80+50-30", 100), ("-(2-5)", 3),
                                         ("2*(3+(4-1))", 12), (" 7 ", 7), ("10-2-3", 5)])
def test_expressions(expr, value):
    assert evaluate(expr) == value


def test_expressions_in_ranges_and_colon_binds_weakest():
    assert one_based("(31-30):(80+50-30) 201:(480+50-30)") == list(range(1, 101)) + list(range(201, 501))
    assert one_based("1:2+3") == [1, 2, 3, 4, 5]                          # 1:5, not 4:5
    assert one_based("201 : 480 + 20") == list(range(201, 501))          # spaces around operators join
    assert one_based("( 2 + 3 ) : 2*4") == [5, 6, 7, 8]
    assert one_based("1:20:2*5") == [1, 11]
    assert one_based("3*2, 1+1") == [6, 2]


@pytest.mark.parametrize("bad, msg", [("3:1", "larger"), ("(1+2", "without"), ("1+2)", "without"),
                                      ("1:2:3:4", "a:b:step"), ("2-5", "start at 1"), ("1/2", "not allowed"),
                                      ("1:5:0", "step"), ("1+", "valid"), ("a:3", "not allowed")])
def test_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        parse_selector(bad, 100)


def test_image_sequence_uses_it():
    assert ImageSequenceNode._parse_selector("1:2+3", 10) == ([0, 1, 2, 3, 4], 0)
