# Recycle bin

Retired code, kept in the repository (with its git history) instead of being deleted.
Nothing here is loaded by the program: the node toolbox only scans `node_editor/nodes/`.

Files keep their original path under `recycle_bin/`, so restoring one is a plain move back.

## Contents

| File | Was | Replaced by |
|---|---|---|
| `node_editor/nodes/calib_nodes.py` | node **Camera Calibration** (`camera_calib`) | **Camera Calibration** (`camera_calibration`, `node_editor/nodes/camera_calibration_node.py`) |
| `node_editor/nodes/chessboard_calibration_node.py` | node **Chessboard Calibration** (`chessboard_calibration`) | the same node (Chessboard tab, *Search settings...*) |
| `tests/test_camera_calib_node.py` | tests of `calib_nodes.py` | — |

Retired 2026-10-21. The helpers the new node still uses were moved, unchanged, to the
active code: corner detection and image ranking to `node_editor/chessboard_detect.py`,
the number-list parser (`7:9`) to `node_editor/chessboard_search.py` (`parse_number_list`).

Projects that contain these retired nodes still open; those nodes are reported as an
unknown node type in the load report and left out.

## Restoring a file

```
git mv recycle_bin/node_editor/nodes/calib_nodes.py node_editor/nodes/calib_nodes.py
git mv recycle_bin/node_editor/nodes/chessboard_calibration_node.py node_editor/nodes/chessboard_calibration_node.py
git mv recycle_bin/tests/test_camera_calib_node.py tests/test_camera_calib_node.py
```

`chessboard_calibration_node.py` imports `CALIB_FLAG_TIPS` from `calib_nodes.py`, so restore
both together.

## Retiring a file

Move it with `git mv` (not a plain move or delete) to the same path under `recycle_bin/`,
add a row to the table above, and check that nothing active still imports it
(`grep -rn "<module name>" node_editor tests`).
