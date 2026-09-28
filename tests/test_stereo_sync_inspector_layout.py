import tkinter as tk

import numpy as np

from node_editor.nodes.stereo_sync_node import StereoSyncNode


def _build(root):
    node = StereoSyncNode("stereo_sync", tk.Canvas(root))
    frame = tk.Frame(root)
    node.build_inspector(frame)
    root.update_idletasks()
    return node, frame


def test_plots_are_scrollable_and_sit_right_of_controls():
    root = tk.Tk()
    root.withdraw()
    try:
        node, frame = _build(root)

        main_row = frame.winfo_children()[1]
        left, plot_frame = main_row.winfo_children()[0], main_row.winfo_children()[1]
        assert left.pack_info()["side"] == "left"
        assert plot_frame.pack_info()["side"] == "left"
        # plot_frame packed after left -> renders to its right.
        assert plot_frame is main_row.winfo_children()[1]

        assert any(isinstance(w, tk.Canvas) for w in plot_frame.winfo_children())
        assert any(isinstance(w, __import__("tkinter.ttk", fromlist=["Scrollbar"]).Scrollbar)
                   for w in plot_frame.winfo_children())
    finally:
        root.destroy()


def test_ctrl_wheel_links_both_time_domain_plots_when_checkbox_checked():
    root = tk.Tk()
    root.withdraw()
    try:
        node, _frame = _build(root)
        for ax in (node._ax_traj, node._ax_err_time, node._ax_point_summary, node._ax_scan):
            ax.set_xlim(0, 10)

        node._link_xaxis_var.set(True)
        event = type("E", (), {
            "inaxes": node._ax_traj, "xdata": 5.0, "ydata": 0.0,
            "button": "up", "key": "control",
        })()
        node._on_plot_scroll(event)

        assert node._ax_traj.get_xlim() == node._ax_err_time.get_xlim()
        assert node._ax_point_summary.get_xlim() == (0.0, 10.0)
        assert node._ax_scan.get_xlim() == (0.0, 10.0)
    finally:
        root.destroy()


def test_ctrl_wheel_stays_independent_when_checkbox_unchecked():
    root = tk.Tk()
    root.withdraw()
    try:
        node, _frame = _build(root)
        for ax in (node._ax_traj, node._ax_err_time, node._ax_scan):
            ax.set_xlim(0, 10)

        node._link_xaxis_var.set(False)
        event = type("E", (), {
            "inaxes": node._ax_traj, "xdata": 5.0, "ydata": 0.0,
            "button": "up", "key": "control",
        })()
        node._on_plot_scroll(event)

        assert node._ax_traj.get_xlim() != (0.0, 10.0)
        assert node._ax_err_time.get_xlim() == (0.0, 10.0)
        assert node._ax_scan.get_xlim() == (0.0, 10.0)
    finally:
        root.destroy()


def test_shift_wheel_zooms_vertical_only():
    root = tk.Tk()
    root.withdraw()
    try:
        node, _frame = _build(root)
        node._ax_traj.set_xlim(0, 10)
        node._ax_traj.set_ylim(-2, 2)

        event = type("E", (), {
            "inaxes": node._ax_traj, "xdata": 5.0, "ydata": 0.0,
            "button": "up", "key": "shift",
        })()
        node._on_plot_scroll(event)

        assert node._ax_traj.get_xlim() == (0.0, 10.0)
        assert node._ax_traj.get_ylim() != (-2.0, 2.0)
    finally:
        root.destroy()


def test_relative_trajectory_checkbox_starts_both_camera_series_at_zero():
    root = tk.Tk()
    root.withdraw()
    try:
        node, frame = _build(root)
        node._t1 = np.array([0.0, 1.0, 2.0])
        node._t2 = np.array([0.0, 1.0, 2.0])
        node._imgpts1 = np.array([[[100.0, 103.0, 108.0], [20.0, 21.0, 23.0]]])
        node._imgpts2 = np.array([[[450.0, 454.0, 459.0], [70.0, 72.0, 75.0]]])
        node._t_shift_trial = np.array([0.0])

        node._relative_trajectory_var.set(True)
        node._redraw_trajectory_plot()

        assert len(node._ax_traj.lines) == 2
        np.testing.assert_allclose(node._ax_traj.lines[0].get_ydata(), [0.0, 3.0, 8.0])
        np.testing.assert_allclose(node._ax_traj.lines[1].get_ydata(), [0.0, 4.0, 9.0])
        assert "relative to index [0]" in node._ax_traj.get_ylabel()

        main_row = frame.winfo_children()[1]
        plot_frame = main_row.winfo_children()[1]
        checkbox_rows = [child for child in plot_frame.winfo_children()
                         if isinstance(child, tk.Frame)]
        texts = [widget.cget("text") for row in checkbox_rows
                 for widget in row.winfo_children() if isinstance(widget, tk.Checkbutton)]
        assert texts[:2] == [
            "Link horizontal axis across time-domain plots (Ctrl+wheel)",
            "Plot camera values relative to time index [0]",
        ]
    finally:
        root.destroy()
