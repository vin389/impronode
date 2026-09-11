import tkinter as tk
from node_editor.node_editor_app import NodeEditorApp
from node_editor.nodes.meshgrid_node import MeshgridNode

root = tk.Tk()
root.geometry('600x400')
app = NodeEditorApp(root)
app._node_counter = 1
node = MeshgridNode('node_0', app.canvas)
node.x = 120
node.y = 80
node.width = 200
node.height = node.get_default_height()
node.build_body()
app.canvas_nodes['node_0'] = node
app.engine.add_node(node)
app.canvas.update_idletasks()
print('before open', node.is_inspector_open())
print('item_at center', app._item_at(node.x + 40, node.y + 40))
print('tags', app.canvas.gettags(app._item_at(node.x + 40, node.y + 40)))
class E:
    x = node.x + 40
    y = node.y + 40
    widget = app.canvas

e = E()
try:
    print('result', app._on_canvas_double_click(e))
except Exception as exc:
    print('EXC', type(exc).__name__, exc)
print('after open', node.is_inspector_open())
if node._inspector_win:
    print('win exists', node._inspector_win.winfo_exists())
    print('body children', [str(c) for c in node._inspector_body.winfo_children()])
root.after(200, root.destroy)
root.mainloop()
