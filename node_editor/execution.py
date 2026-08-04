# execution.py

from enum import Enum

class ExecutionMode(Enum):
    SYNC        = "sync"
    BACKGROUND  = "background"
    STREAMING   = "streaming"