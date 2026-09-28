import tkinter as tk

import numpy as np

from node_editor.nodes.signal_generator_node import SignalGeneratorNode


def test_inspector_builds_all_channels_and_preview():
    root = tk.Tk()
    root.withdraw()
    try:
        node = SignalGeneratorNode("sig", tk.Canvas(root))
        frame = tk.Frame(root)
        node.build_inspector(frame)
        root.update_idletasks()

        assert len(node._sine_widget_vars) == 6
        for channel in node._sine_widget_vars:
            assert set(channel["entries"]) == {
                "amplitude", "frequency", "decay", "time_shift",
            }
            for key, entry in channel["entries"].items():
                assert entry.winfo_exists()
                assert str(entry.cget("textvariable")) == str(channel[key])
        assert set(node._noise_widget_vars["entries"]) == {"amplitude", "seed"}
        assert len(node._ax.lines) == 1
        outputs = node.compute({})
        np.testing.assert_allclose(node._ax.lines[0].get_xdata(), outputs["t"])
        np.testing.assert_allclose(node._ax.lines[0].get_ydata(), outputs["signal"])

        # Apply must read pending edits even without Return/FocusOut,
        # regenerate the preview, and preserve the user's pan/zoom.
        node._duration_var.set(2.0)
        node._sine_widget_vars[0]["amplitude"].set(3.0)
        node._ax.set_xlim(0.2, 0.8)
        node._ax.set_ylim(-4.0, 4.0)
        node._ax.lines[0].set_ydata(np.zeros_like(outputs["signal"]))
        apply_button = next(
            widget for row in frame.winfo_children()
            for widget in row.winfo_children()
            if isinstance(widget, tk.Button) and widget.cget("text") == "Apply"
        )
        apply_button.invoke()
        root.update_idletasks()
        outputs = node.compute({})
        assert len(outputs["t"]) == 200
        assert np.max(outputs["signal"]) > 2.9
        np.testing.assert_allclose(node._ax.lines[0].get_xdata(), outputs["t"])
        np.testing.assert_allclose(node._ax.lines[0].get_ydata(), outputs["signal"])
        assert node._ax.get_xlim() == (0.2, 0.8)
        assert node._ax.get_ylim() == (-4.0, 4.0)
    finally:
        root.destroy()