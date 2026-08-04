# node_editor/pin_types.py

from enum import Enum, auto
from dataclasses import dataclass, field


class PinType(Enum):
    SCALAR = auto()   # float / int
    ARRAY  = auto()   # numpy ndarray (any shape)
    IMAGE  = auto()   # numpy ndarray (HxW or HxWxC, uint8)
    STRING = auto()   # str (file paths, parameters, etc.)
    TRIGGER = auto()  # event pulse
    ANY    = auto()   # compatible with all types


# Compatibility table: dst_type → set of accepted src_types
PIN_COMPATIBILITY: dict[PinType, set[PinType]] = {
    PinType.SCALAR: {PinType.SCALAR},
    PinType.ARRAY:  {PinType.ARRAY},
    PinType.IMAGE:  {PinType.IMAGE, PinType.ARRAY},  # IMAGE is a special array
    PinType.STRING: {PinType.STRING},
    PinType.TRIGGER: {PinType.TRIGGER},
    PinType.ANY:    {PinType.SCALAR, PinType.ARRAY,
                     PinType.IMAGE,  PinType.STRING, PinType.TRIGGER, PinType.ANY},
}


def pins_compatible(src_type: PinType, dst_type: PinType) -> bool:
    """Return True if src_type can connect to dst_type."""
    return src_type in PIN_COMPATIBILITY.get(dst_type, set())


@dataclass
class PinDef:
    name:     str
    type:     PinType
    label:    str        = ""
    optional: bool       = False
    shape:    tuple | None = None   # e.g. (-1, 3) means (N, 3); None means any shape
    dtype:    str  | None = None   # e.g. "float32", "uint8"; None means any dtype


@dataclass
class PinSchema:
    inputs:  list[PinDef] = field(default_factory=list)
    outputs: list[PinDef] = field(default_factory=list)