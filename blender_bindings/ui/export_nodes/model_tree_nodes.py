from collections import defaultdict, deque

import bpy
from bpy.types import Node, NodeTree, Operator

from . import nodes


class SourceIO_OP_EvaluateNodeTree(Operator):
    bl_idname = "sourceio.evaluate_nodetree"
    bl_label = "Evaluate tree"
    tmp_file: bpy.types.Text

    def execute(self, context: bpy.types.Context):
        if not bpy.data.texts.get('qc', False):
            self.tmp_file = bpy.data.texts.new('qc')
        else:
            self.tmp_file = bpy.data.texts['qc']
        all_nodes = context.space_data.node_tree.nodes
        outputs = []  # type:list[Node]
        for node in all_nodes:  # type: Node
            if node.bl_idname == "SourceIOModelNode":
                outputs.append(node)
        for output in outputs:  # type:nodes.SourceIOModelNode
            self.traverse_tree(output)
        return {'FINISHED'}

    def traverse_tree(self, start_node: nodes.SourceIOModelNode):
        start_node.write(self.tmp_file)
        objects = start_node.inputs['Objects']
        bodygroups = start_node.inputs['Bodygroups']
        skins = start_node.inputs['Skin']
        textures = start_node.inputs['Textures']

        if textures.is_linked:
            for link in textures.links:
                texture_node: nodes.SourceIOTextureInputNode = link.from_node
                texture_data = texture_node.execute()
                print(f"Texture data shape: {texture_data}")
        if objects.is_linked:
            for link in objects.links:
                object_node: nodes.SourceIOObjectNode = link.from_node
                object_node.write(self.tmp_file)

        if bodygroups.is_linked:
            for link in bodygroups.links:
                bodygroup_node: nodes.SourceIOBodygroupNode = link.from_node
                bodygroup_node.write(self.tmp_file)
        if skins.is_linked:
            skin_node = skins.links[0].from_node  # type: nodes.SourceIOSkinNode
            self.tmp_file.write(str(skin_node.get_value()))
            self.tmp_file.write('\n')


class SourceIO_NT_ModelTree(NodeTree):
    bl_idname = 'sourceio.model_definition'
    bl_label = "SourceIO model definition"
    bl_icon = 'NODETREE'

    def update(self, ):
        for node in self.nodes:
            node.update()
        for link in self.links:  # type:bpy.types.NodeLink
            if link.from_socket.bl_idname != link.to_socket.bl_idname:
                self.links.remove(link)
        self.check_link_duplicates()

    def check_link_duplicates(self):
        to_remove = []
        for link in self.links:
            for link2 in self.links:
                if link == link2 or link in to_remove:
                    continue

                if (
                        link.from_node == link2.from_node and
                        link.to_node == link2.to_node and
                        link.from_socket == link2.to_socket
                ):
                    to_remove.append(link2)
                    break
        for link in to_remove:
            self.links.remove(link)


class EvalContext:
    """Per-pass state holding computed socket values and per-node caches."""

    def __init__(self, tree):
        self.tree = tree
        self.values = {}

    def get_output_result(self, node, socket_name):
        """Get the computed value for a given output socket of a node."""
        for sock in node.outputs:
            if sock.name == socket_name:
                return self.values.get((node, sock))
        return None


def _is_frame(n):
    """Return True if node is a frame."""
    return n.type == "FRAME"


def _collapse_reroute_to(link):
    """Follow downstream through reroute nodes and return (node, socket)."""
    n, s = link.to_node, link.to_socket
    while n.type == "REROUTE":
        outs = n.outputs
        if not outs or not outs[0].links:
            break
        link = outs[0].links[0]
        n, s = link.to_node, link.to_socket
    return n, s

def _collapse_upstream(link, collapse_reroutes=True, collapse_muted=True):
    """Follow upstream across REROUTE and muted nodes using internal_links; return (node, socket)."""
    n, s = link.from_node, link.from_socket
    while True:
        if collapse_reroutes and n.type == "REROUTE":
            ins = n.inputs
            if not ins or not ins[0].links:
                break
            link = ins[0].links[0]
            n, s = link.from_node, link.from_socket
            continue

        if collapse_muted and n.mute:
            # Find which input feeds this muted node's output socket `s`
            mapped = None
            for il in n.internal_links:
                if il.to_socket == s and il.from_socket and il.from_socket.links:
                    mapped = il.from_socket.links[0]
                    break
            if mapped is None:
                break
            link = mapped
            n, s = link.from_node, link.from_socket
            continue

        break
    return n, s

def _collect_inputs(ctx, node, collapse_reroutes=True):
    """Return a mapping of input socket names to resolved values for a node."""
    out = {}
    for sock in node.inputs:
        if sock.is_linked and sock.links:
            link = sock.links[0]
            upn, ups = _collapse_upstream(link, collapse_reroutes=collapse_reroutes, collapse_muted=True)
            out[sock.name] = ctx.values.get((upn, ups))
        else:
            out[sock.name] = getattr(sock, "value", None) or getattr(sock, "default_value", None)
    return out



def _build_graph(tree, collapse_reroutes=True, skip_frames=True):
    """Build adjacency and indegree for the node DAG."""
    adj = defaultdict(set)
    indeg = defaultdict(int)
    for n in tree.nodes:
        if skip_frames and _is_frame(n):
            continue
        adj.setdefault(n, set())
    for n in list(adj.keys()):
        for out_sock in n.outputs:
            for link in out_sock.links:
                m, _ = _collapse_reroute_to(link) if collapse_reroutes else (link.to_node, link.to_socket)
                if skip_frames and _is_frame(m):
                    continue
                if m not in adj[n]:
                    adj[n].add(m)
                    indeg[m] += 1
                    indeg.setdefault(n, 0)
    for n in adj:
        indeg.setdefault(n, 0)
    return adj, indeg


def _restrict_to_upstream(tree, targets, collapse_reroutes=True, skip_frames=True):
    """Return the set of nodes that can reach any of the targets."""
    rev = defaultdict(set)
    for n in tree.nodes:
        if skip_frames and _is_frame(n):
            continue
        for out_sock in n.outputs:
            for link in out_sock.links:
                m, _ = _collapse_reroute_to(link) if collapse_reroutes else (link.to_node, link.to_socket)
                if skip_frames and _is_frame(m):
                    continue
                rev[m].add(n)
        rev.setdefault(n, set())
    seen = set()
    dq = deque(targets)
    while dq:
        x = dq.popleft()
        if x in seen:
            continue
        seen.add(x)
        for p in rev.get(x, ()):
            if p not in seen:
                dq.append(p)
    return seen


def compute_order(tree, subset=None, collapse_reroutes=True, skip_frames=True):
    """Compute a topological order using a deque; subset restricts to an induced subgraph."""
    adj, indeg = _build_graph(tree, collapse_reroutes, skip_frames)
    if subset is not None:
        keep = set(subset)
        adj = {u: {v for v in vs if v in keep} for u, vs in adj.items() if u in keep}
        indeg = {u: 0 for u in adj}
        for u, vs in adj.items():
            for v in vs:
                indeg[v] = indeg.get(v, 0) + 1
    q = deque([n for n, d in indeg.items() if d == 0 and n in adj])
    order, seen = [], set()
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if any(d > 0 for d in indeg.values() if isinstance(d, int)):
        raise ValueError("Cycle detected in node graph")
    return order


def evaluate_tree(tree, targets=None, collapse_reroutes=True, skip_frames=True) -> EvalContext:
    ctx = EvalContext(tree)
    subset = _restrict_to_upstream(tree, targets, collapse_reroutes, skip_frames) if targets else None
    order = compute_order(tree, subset=subset, collapse_reroutes=collapse_reroutes, skip_frames=skip_frames)
    for node in order:
        if node.mute:                           # hard-skip muted nodes
            continue
        props = _collect_inputs(ctx, node, collapse_reroutes=collapse_reroutes)
        outputs = node.process(props) or {}
        for sock in node.outputs:
            if sock.name in outputs:
                ctx.values[(node, sock)] = outputs[sock.name]
    return ctx

