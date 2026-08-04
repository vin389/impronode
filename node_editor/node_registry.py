# node_editor/node_registry.py

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path

from node_editor.base_node import BaseNode


def discover_nodes(nodes_package_path: str | Path
                   ) -> dict[str, list[tuple[str, type]]]:
    """
    Scan all .py files under nodes_package_path,
    import them, and collect all BaseNode subclasses.

    Returns a dict grouped by CATEGORY:
    {
        "source":    [("Webcam Input", WebcamInputNode), ...],
        "process":   [("Gaussian Blur", GaussianBlurNode), ...],
        "visualize": [("Video Output", VideoPlayOutputNode), ...],
        "misc":      [...],
    }
    """
    nodes_path = Path(nodes_package_path)
    found: dict[str, list[tuple[str, type]]] = {}

    for py_file in sorted(nodes_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue    # skip __init__.py, __pycache__ etc.

        module_name = f"node_editor.nodes.{py_file.stem}"
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec   = importlib.util.spec_from_file_location(
                    module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except Exception as e:
            print(f"[NodeRegistry] failed to import "
                  f"{py_file.name}: {e}")
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (cls is BaseNode
                    or not issubclass(cls, BaseNode)
                    or not cls.NODE_TYPE          # empty string guard
                    or cls.NODE_TYPE == "base"):  # BaseNode itself
                continue

            # avoid registering the same class twice
            # (can happen if a class is imported into multiple modules)
            category = cls.CATEGORY or "misc"
            entry    = (cls.DISPLAY_NAME or cls.NODE_TYPE, cls)

            already  = any(
                c is cls
                for entries in found.values()
                for _, c in entries)
            if already:
                continue

            found.setdefault(category, []).append(entry)

    return found