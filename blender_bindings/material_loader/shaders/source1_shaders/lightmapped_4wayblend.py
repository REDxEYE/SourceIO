import bpy
from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class Lightmapped4WayBlend(Source1ShaderBase):
    SHADER = 'lightmapped_4wayblend'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.0, 0.3, 1.0))
        return None

    @property
    def basetexture2(self):
        texture_path = self._vmt.get_string('$basetexture2', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None

    @property
    def basetexture3(self):
        texture_path = self._vmt.get_string('$basetexture3', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None

    @property
    def basetexture4(self):
        texture_path = self._vmt.get_string('$basetexture4', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None
    
    @property
    def detail(self):
        texture_path = self._vmt.get_string('$detail', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0.3, 0.0, 1.0))
        return None
    
    @property
    def texture1_uvscale(self):
        vector, _ = self._vmt.get_vector('$texture1_uvscale', None)
        if vector:
            vector = list(vector)
            return self.ensure_length(vector, 3, vector[0])
        return vector

    @property
    def texture2_uvscale(self):
        vector, _ = self._vmt.get_vector('$texture2_uvscale', None)
        if vector:
            vector = list(vector)
            return self.ensure_length(vector, 3, vector[0])
        return vector
    
    @property
    def texture3_uvscale(self):
        vector, _ = self._vmt.get_vector('$texture3_uvscale', None)
        if vector:
            vector = list(vector)
            return self.ensure_length(vector, 3, vector[0])
        return vector
    
    @property
    def texture4_uvscale(self):
        vector, _ = self._vmt.get_vector('$texture4_uvscale', None)
        if vector:
            vector = list(vector)
            return self.ensure_length(vector, 3, vector[0])
        return vector
    
    @property
    def detailscale(self):
        vector, _ = self._vmt.get_vector('$detailscale', None)
        #print(vector)
        if vector:
            vector = list(vector)
            vector = self.ensure_length(vector, 3, vector[0])
            print(vector)
            return vector
        return vector

    @property
    def ssbump(self):
        return self._vmt.get_int('$ssbump', 0) == 1

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            if self.ssbump:
                image = self.convert_ssbump(image)
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumpmap2(self):
        texture_path = self._vmt.get_string('$bumpmap2', None) or self._vmt.get_string(
            '$basenormalmap2', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            if self.ssbump:
                image = self.convert_ssbump(image)
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumpmap3(self):
        texture_path = self._vmt.get_string('$basenormalmap3', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            if self.ssbump:
                image = self.convert_ssbump(image)
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumpmap4(self):
        texture_path = self._vmt.get_string('$basenormalmap4', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            if self.ssbump:
                image = self.convert_ssbump(image)
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        
        self.do_arrange = True
        
        vars = [
            '$texture1_lumstart',
            '$texture1_lumend',
            '$texture2_lumstart',
            '$texture2_lumend',
            '$texture2_blendstart',
            '$texture2_blendend',
            '$lumblendfactor2',
            '$texture3_lumstart',
            '$texture3_lumend',
            '$texture3_blendstart',
            '$texture3_blendend',
            '$texture3_bumpblendfactor',
            #'$texture4_blendmode',
            '$texture4_lumstart',
            '$texture4_lumend',
            '$texture4_blendstart',
            '$texture4_blendend',
            '$texture4_bumpblendfactor',
            '$lumblendfactor3',
            '$lumblendfactor4',
            '$detailblendfactor',
            '$detailblendfactor2',
            '$detailblendfactor3',
            '$detailblendfactor4'
        ]

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        if bpy.app.version >= (4, 0, 0):
            shader.inputs['Specular IOR Level'].default_value = 0.0
        else:
            shader.inputs['Specular'].default_value = 0.0
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        Fway: bpy.types.ShaderNodeGroup
        Fway = self.create_node_group('4wayBlend')
        self.connect_nodes(Fway.outputs[0], shader.inputs['Base Color'])
        normalMap = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(Fway.outputs[1], normalMap.inputs['Color'])
        self.connect_nodes(Fway.outputs[2], normalMap.inputs['Strength'])
        self.connect_nodes(normalMap.outputs[0], shader.inputs['Normal'])

        basetexture1 = self.basetexture
        basetexture2 = self.basetexture2
        basetexture3 = self.basetexture3
        basetexture4 = self.basetexture4
        normal_texture1 = self.bumpmap
        normal_texture2 = self.bumpmap2
        bases = [None, None, None, None]
        normals = [None, None]
        if basetexture1:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture1')
            basetexture_node.image = basetexture1
            bases[0] = basetexture_node
            self.connect_nodes(basetexture_node.outputs[0], Fway.inputs['$basetexture'])
            scale = self.texture1_uvscale
            #print('scale1', scale)
            if scale:
                uv = self.create_node(Nodes.ShaderNodeUVMap, name='UV Map', location=[-760, -700])
                scaler = self.create_node(Nodes.ShaderNodeVectorMath, name='$texture1_uvscale', location=[-580, -700])
                scaler.inputs[1].default_value = scale
                scaler.operation = 'MULTIPLY'

                self.connect_nodes(uv.outputs[0], scaler.inputs[0])
                self.connect_nodes(scaler.outputs[0], basetexture_node.inputs[0])

        if basetexture2:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture2')
            basetexture_node.image = basetexture2
            bases[1] = basetexture_node
            self.connect_nodes(basetexture_node.outputs[0], Fway.inputs['$basetexture2'])
            scale = self.texture2_uvscale
            #print('scale2', scale)
            if scale:
                uv = self.create_node(Nodes.ShaderNodeUVMap, name='UV Map', location=[-760, -700])
                scaler = self.create_node(Nodes.ShaderNodeVectorMath, name='$texture2_uvscale', location=[-580, -700])
                scaler.inputs[1].default_value = scale
                scaler.operation = 'MULTIPLY'

                self.connect_nodes(uv.outputs[0], scaler.inputs[0])
                self.connect_nodes(scaler.outputs[0], basetexture_node.inputs[0])

        if basetexture3:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture3')
            basetexture_node.image = basetexture3
            bases[2] = basetexture_node
            scale = self.texture3_uvscale
            #print('scale3', scale)
            if scale:
                uv = self.create_node(Nodes.ShaderNodeUVMap, name='UV Map', location=[-760, -700])
                scaler = self.create_node(Nodes.ShaderNodeVectorMath, name='$texture3_uvscale', location=[-580, -700])
                scaler.inputs[1].default_value = scale
                scaler.operation = 'MULTIPLY'

                self.connect_nodes(uv.outputs[0], scaler.inputs[0])
                self.connect_nodes(scaler.outputs[0], basetexture_node.inputs[0])

            self.connect_nodes(basetexture_node.outputs[0], Fway.inputs['$basetexture3'])
        if basetexture4:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture4')
            basetexture_node.image = basetexture4
            bases[3] = basetexture_node
            self.connect_nodes(basetexture_node.outputs[0], Fway.inputs['$basetexture4'])
            scale = self.texture4_uvscale
            #print('scale4', scale)
            if scale:
                uv = self.create_node(Nodes.ShaderNodeUVMap, name='UV Map', location=[-760, -700])
                scaler = self.create_node(Nodes.ShaderNodeVectorMath, name='$texture4_uvscale', location=[-580, -700])
                scaler.inputs[1].default_value = scale
                scaler.operation = 'MULTIPLY'

                self.connect_nodes(uv.outputs[0], scaler.inputs[0])
                self.connect_nodes(scaler.outputs[0], basetexture_node.inputs[0])

        if self.detail:
            detailtexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$detail')
            detailtexture_node.image = self.detail
            self.connect_nodes(detailtexture_node.outputs[0], Fway.inputs['$detail'])

            if self.detailscale:
                uv = self.create_node(Nodes.ShaderNodeUVMap, name='UV Map', location=[-760, -700])
                scaler = self.create_node(Nodes.ShaderNodeVectorMath, name='$detailscale', location=[-580, -700])
                scaler.inputs[1].default_value = self.detailscale
                scaler.operation = 'MULTIPLY'

                self.connect_nodes(uv.outputs[0], scaler.inputs[0])
                self.connect_nodes(scaler.outputs[0], detailtexture_node.inputs[0])

        if normal_texture1:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap1')
            bumpmap_node.image = normal_texture1
            normals[0] = bumpmap_node
            self.connect_nodes(bumpmap_node.outputs[0], Fway.inputs['$bumpmap'])
        if normal_texture2:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap2')
            bumpmap_node.image = normal_texture2
            normals[1] = bumpmap_node
            self.connect_nodes(bumpmap_node.outputs[0], Fway.inputs['$bumpmap2'])

        for var in vars:
            value = self._vmt.get_float(var, 0)
            Fway.inputs[var].default_value = value