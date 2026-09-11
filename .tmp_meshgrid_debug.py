import tkinter as tk
from node_editor.nodes.meshgrid_node import MeshgridNode

root = tk.Tk(); root.withdraw()
canvas = tk.Canvas(root, width=200, height=200)
node = MeshgridNode('mesh', canvas)
node._init_state()
node.open_inspector()
root.update_idletasks()
win = node._inspector_win
body = node._inspector_body

def walk(widget, depth=0):
    indent = '  ' * depth
    cls = widget.winfo_class()
    print(f'{indent}{widget} class={cls} children={len(widget.winfo_children())}')
    for ch in widget.winfo_children():
        walk(ch, depth+1)

print('WIN_EXISTS', win is not None and win.winfo_exists())
print('BODY', body)
walk(body)
root.destroy()
