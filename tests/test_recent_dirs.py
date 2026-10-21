import os
from types import SimpleNamespace

from node_editor import project_context, recent_dirs
from node_editor.recent_projects import RecentProjects


def test_recent_dirs_saved_relative_to_project_and_restored(tmp_path):
    proj = tmp_path / "proj"
    data = tmp_path / "data" / "imgs"
    proj.mkdir()
    data.mkdir(parents=True)
    project_context.set_project_file_path(str(proj / "p.xlsx"))
    try:
        node = SimpleNamespace()
        recent_dirs.remember_dir(node, "load", str(data))
        saved = recent_dirs.dirs_to_params(node)
        assert saved == {"load": os.path.join("..", "data", "imgs")}

        restored = SimpleNamespace()
        recent_dirs.dirs_from_params(restored, saved)
        assert recent_dirs.get_recent_dir(restored, "load") == str(data)
        # The remembered folder wins over the caller's default.
        kwargs = recent_dirs._with_initialdir(restored, "load", {"initialdir": "elsewhere"})
        assert kwargs["initialdir"] == str(data)
        # Nothing remembered for "save": falls back to the caller, then the project folder.
        assert recent_dirs._with_initialdir(restored, "save", {"initialdir": "x"})["initialdir"] == "x"
        assert recent_dirs._with_initialdir(restored, "save", {})["initialdir"] == str(proj)
    finally:
        project_context.set_project_file_path(None)


def test_recent_projects_most_recent_first_without_duplicates(tmp_path):
    store = tmp_path / "recent.json"
    recent = RecentProjects(store, max_items=3)
    for name in "abcda":
        recent.add(str(tmp_path / f"{name}.xlsx"))
    names = [os.path.basename(p) for p in RecentProjects(store, max_items=3).paths]
    assert names == ["a.xlsx", "d.xlsx", "c.xlsx"]
