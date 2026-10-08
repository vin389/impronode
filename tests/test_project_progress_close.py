import tkinter as tk
from types import SimpleNamespace

from node_editor import node_editor_app as app_module
from node_editor.node_editor_app import NodeEditorApp


def test_project_progress_close_waits_for_load_and_save(monkeypatch, tmp_path):
    root = tk.Tk()
    root.withdraw()
    try:
        app = NodeEditorApp.__new__(NodeEditorApp)
        app.root = root
        app.engine = SimpleNamespace(node_compute_callback=None, execution_complete_callback=None)
        app._project_progress_window = None
        app._project_progress_text = None
        app._project_progress_close_button = None
        app._project_progress_start = None
        app._project_node_load_starts = {}
        app._file_load_in_progress = False
        app._file_load_last_finished_at = None
        app._dirty = False
        app._suspend_dirty = False
        app._project_path = None
        # One node, so a load really waits for the graph's execution-complete
        # callback (a project with NO nodes finishes at once: nothing to run).
        app.canvas_nodes = {"node_0": object()}
        app._serialize_graph = lambda: ([], [], {})
        app._set_clean = lambda: None
        app._restore_canvas_size = lambda _meta: None
        app._set_project_path = lambda path: setattr(app, "_project_path", path)
        app._show_load_report = lambda _issues: None
        monkeypatch.setattr(app_module.filedialog, "askopenfilename", lambda **_kwargs: str(tmp_path / "load.xlsx"))
        monkeypatch.setattr(app_module, "load_project", lambda _path: ([], [], {}))

        def start_background_load(_nodes, _links):
            app.engine.execution_complete_callback = app._on_project_load_execution_complete
            return []

        app._load_graph = start_background_load
        app._file_load()
        window = app._project_progress_window
        button = app._project_progress_close_button
        assert button.cget("state") == tk.DISABLED
        app._close_project_progress()  # the title-bar X must not interrupt a load
        assert window.winfo_exists()
        app.engine.execution_complete_callback()
        assert button.cget("state") == tk.NORMAL
        button.invoke()
        assert not window.winfo_exists()

        def finish_during_graph_load(_nodes, _links):
            app._on_project_load_execution_complete()
            assert app._project_progress_close_button.cget("state") == tk.DISABLED
            return []

        app._load_graph = finish_during_graph_load
        app._file_load_last_finished_at = None
        app._file_load()
        assert app._project_progress_close_button.cget("state") == tk.NORMAL
        app._close_project_progress()

        def failed_load(_path):
            assert app._project_progress_close_button.cget("state") == tk.DISABLED
            raise OSError("cannot read")

        errors = []
        monkeypatch.setattr(app_module.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
        monkeypatch.setattr(app_module, "load_project", failed_load)
        app._file_load_last_finished_at = None
        app._file_load()
        assert app._project_progress_close_button.cget("state") == tk.NORMAL
        assert "failed" in app._project_progress_text.get("1.0", tk.END).lower()
        assert errors
        app._close_project_progress()

        app._project_path = str(tmp_path / "save.xlsx")

        def save_while_busy(_path, _nodes, _links, _meta):
            assert app._project_progress_close_button.cget("state") == tk.DISABLED
            app._close_project_progress()
            assert app._project_progress_window.winfo_exists()

        monkeypatch.setattr(app_module, "save_project", save_while_busy)
        assert app._file_save() is True
        assert app._project_progress_close_button.cget("state") == tk.NORMAL
        app._project_progress_close_button.invoke()
        assert app._project_progress_window is None

        def failed_save(_path, _nodes, _links, _meta):
            raise OSError("cannot write")

        monkeypatch.setattr(app_module, "save_project", failed_save)
        assert app._file_save() is False
        assert app._project_progress_close_button.cget("state") == tk.NORMAL
        assert "failed" in app._project_progress_text.get("1.0", tk.END).lower()
        assert errors
        app._close_project_progress()
        assert app._project_progress_window is None
    finally:
        root.destroy()