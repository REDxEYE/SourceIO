from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class EyeRefract(Source1ShaderBase):
    SHADER: str = 'eyerefract'

    @property
    def iris(self):
        texture_path = self._vmt.get_string('$iris', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        iris = self.iris
        if iris:
            self.create_and_connect_texture_node(iris, shader.inputs['Base Color'], name='$iris')
