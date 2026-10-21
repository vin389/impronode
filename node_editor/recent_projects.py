"""
List of recently opened / saved project files, kept per user in
~/.impronode/recent_projects.json (most recent first).
"""

import json
import os
from pathlib import Path

MAX_RECENT_PROJECTS = 10


def _default_store() -> Path:
    return Path.home() / ".impronode" / "recent_projects.json"


class RecentProjects:
    def __init__(self, store_path: str | Path | None = None,
                 max_items: int = MAX_RECENT_PROJECTS):
        self.store_path = Path(store_path) if store_path else _default_store()
        self.max_items = max_items
        self.paths: list[str] = []
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        if isinstance(data, dict):
            data = data.get("recent_projects", [])
        self.paths = [str(p) for p in data if isinstance(p, str) and p][: self.max_items]

    def save(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store_path.with_name(self.store_path.name + ".tmp")
            tmp.write_text(json.dumps({"recent_projects": self.paths}, indent=2), encoding="utf-8")
            os.replace(tmp, self.store_path)
        except OSError as e:
            print(f"Could not save the recent project list: {e}")

    @staticmethod
    def _key(path: str) -> str:
        return os.path.normcase(os.path.abspath(path))

    def add(self, path: str) -> None:
        path = os.path.abspath(path)
        key = self._key(path)
        self.paths = [path] + [p for p in self.paths if self._key(p) != key]
        del self.paths[self.max_items:]
        self.save()

    def remove(self, path: str) -> None:
        key = self._key(path)
        self.paths = [p for p in self.paths if self._key(p) != key]
        self.save()

    def clear(self) -> None:
        self.paths = []
        self.save()
