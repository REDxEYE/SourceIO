from bpy.types import NodeTree


class SourceIO_NT_ModelTree(NodeTree):
    bl_idname = 'sourceio.model_definition'
    bl_label = "SourceIO model definition"
    bl_icon = 'NODETREE'

    def update(self):
        from .flex_graph import (
            migrate_flex_geometry, repair_flex_groups,
            sync_connected_flex_groups, sync_flex_tree)
        from .nodes.sfm_rig_nodes import migrate_rig_nodes
        repair_flex_groups(self)
        migrate_flex_geometry(self)
        migrate_rig_nodes(self)
        for node in self.nodes:
            if (node.bl_idname == 'SourceIOModelEntryNode'
                    and not node.inputs.get('Flex Rules')):
                node.inputs.new('SourceIOFlexRulesSocket', 'Flex Rules')
            elif node.bl_idname == 'SourceIOPropModelNode':
                if not node.inputs.get('Flex Rules'):
                    node.inputs.new('SourceIOFlexRulesSocket', 'Flex Rules',
                                    use_multi_input=True)
                if not node.inputs.get('Bone names'):
                    node.inputs.new('SourceIOBoneNamesSocket', 'Bone names')
                if not node.inputs.get('SFM rig'):
                    node.inputs.new('SourceIOSfmRigSocket', 'SFM rig')
        seen = set()
        invalid = []
        wrappers = []
        vtf_outputs = []
        for link in self.links:
            endpoints = (link.from_socket, link.to_socket)
            if endpoints in seen:
                invalid.append(link)
                continue
            seen.add(endpoints)
            if (link.from_node.bl_idname == 'SourceIOPropGeometryNode'
                    and link.from_socket.name == 'Geometry'
                    and link.to_node.bl_idname == 'SourceIOPropModelNode'
                    and link.to_socket.name in {'Models', 'Bodygroups'}):
                rank = sum(
                    other.multi_input_sort_id > link.multi_input_sort_id
                    for other in link.to_socket.links)
                wrappers.append((link, link.from_node, link.to_node, rank,
                                 'SourceIOModelEntryNode'
                                 if link.to_socket.name == 'Models'
                                 else 'SourceIOPropBodygroupNode'))
            elif (link.from_socket.bl_idname == 'SourceIOMaterialSocket'
                  and link.to_node.bl_idname == 'SourceIOPropModelNode'
                  and link.to_socket.name == 'Skin families'):
                rank = sum(
                    other.multi_input_sort_id > link.multi_input_sort_id
                    for other in link.to_socket.links)
                wrappers.append((link, link.from_node, link.to_node, rank,
                                 'SourceIOPropSkinFamilyNode'))
            elif (link.from_socket.bl_idname == 'SourceIOTextureSocket'
                  and link.to_socket.bl_idname == 'SourceIOTextureVtfSocket'):
                parameter = next(
                    (key for _, label, key
                     in getattr(link.to_node, '_textures', ())
                     if label == link.to_socket.name), None)
                if parameter is None:
                    invalid.append(link)
                else:
                    vtf_outputs.append((link, link.from_socket,
                                        link.to_socket, link.from_node,
                                        link.to_node, parameter))
            elif (link.from_node.type != 'REROUTE'
                  and link.to_node.type != 'REROUTE'
                  and link.from_socket.bl_idname
                  != link.to_socket.bl_idname):
                invalid.append(link)
        for link in invalid:
            self.links.remove(link)
        for link, source, output, rank, node_type in wrappers:
            source_socket = link.from_socket
            self.links.remove(link)
            wrapper = self.nodes.new(node_type)
            wrapper.location = (source.location_absolute
                                + output.location_absolute) * 0.5
            if node_type in {'SourceIOModelEntryNode',
                             'SourceIOPropBodygroupNode'}:
                selected = (source.obj if source.source_type == 'OBJECT'
                            else source.collection)
                base = source.basename.strip() or (
                    selected.name if selected else source.name)
                if node_type == 'SourceIOModelEntryNode':
                    property_name, input_name, output_name, target_name = (
                        'entry_name', 'Geometry', 'Model', 'Models')
                else:
                    property_name, input_name, output_name, target_name = (
                        'bodygroup_name', 'Choices', 'Bodygroup',
                        'Bodygroups')
                used = {getattr(node, property_name) for node in self.nodes
                        if node.bl_idname == node_type and node != wrapper}
                name = base
                suffix = 1
                while name in used:
                    name = f'{base}.{suffix:03d}'
                    suffix += 1
                setattr(wrapper, property_name, name)
            else:
                input_name, output_name, target_name = (
                    'Materials', 'Skin family', 'Skin families')
            self.links.new(source_socket, wrapper.inputs[input_name])
            new_link = self.links.new(wrapper.outputs[output_name],
                                      output.inputs[target_name])
            # Preserve the dropped link's position on the multi-input socket.
            ordered = sorted(output.inputs[target_name].links,
                             key=lambda item: item.multi_input_sort_id,
                             reverse=True)
            rank = min(rank, len(ordered) - 1)
            index = ordered.index(new_link)
            while index != rank:
                next_index = index + (1 if index < rank else -1)
                new_link.swap_multi_input_sort_id(ordered[next_index])
                ordered[index], ordered[next_index] = (ordered[next_index],
                                                       ordered[index])
                index = next_index

        for (link, source_socket, target_socket, source, material,
             parameter) in vtf_outputs:
            self.links.remove(link)
            vtf = self.nodes.new('SourceIOVTFOutputNode')
            vtf.location = (source.location_absolute
                            + material.location_absolute) * 0.5
            base = material.vmt_path.strip().replace('\\', '/')
            if base.lower().startswith('materials/'):
                base = base[len('materials/'):]
            if base.lower().endswith('.vmt'):
                base = base[:-4]
            suffix = ('albedo' if parameter == '$basetexture'
                      else parameter[1:])
            vtf.vtf_name = f'{base or "models/props/material"}_{suffix}.vtf'
            vtf.is_normal = parameter in {'$bumpmap', '$normalmap'}
            self.links.new(source_socket, vtf.inputs['Texture'])
            self.links.new(vtf.outputs['VTF'], target_socket)
        # Topology changes may add/remove a geometry or rule link without a
        # Key update.
        sync_flex_tree(self)
        sync_connected_flex_groups(self)
