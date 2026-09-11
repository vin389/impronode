from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema


class DummyTitleNode(BaseNode):
    NODE_TYPE = "dummy_title"
    DISPLAY_NAME = "Dummy Node"
    CATEGORY = "misc"
    NODE_HEIGHT = 100

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(inputs=[], outputs=[])

    def build_body(self) -> None:
        pass


def test_default_height_keeps_room_for_multiline_title():
    node = DummyTitleNode("dummy_title", None)
    assert node.get_default_height() >= 118
