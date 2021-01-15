from pathlib import Path
from typing import Dict, Any, Optional

import bpy
import numpy as np

from .node_arranger import nodes_iterate
from ...bpy_utilities.logging import BPYLoggingManager


class Nodes:
    ShaderNodeAddShader = 'ShaderNodeAddShader'
    ShaderNodeAmbientOcclusion = 'ShaderNodeAmbientOcclusion'
    ShaderNodeAttribute = 'ShaderNodeAttribute'
    ShaderNodeBackground = 'ShaderNodeBackground'
    ShaderNodeBevel = 'ShaderNodeBevel'
    ShaderNodeBlackbody = 'ShaderNodeBlackbody'
    ShaderNodeBrightContrast = 'ShaderNodeBrightContrast'
    ShaderNodeBsdfAnisotropic = 'ShaderNodeBsdfAnisotropic'
    ShaderNodeBsdfDiffuse = 'ShaderNodeBsdfDiffuse'
    ShaderNodeBsdfGlass = 'ShaderNodeBsdfGlass'
    ShaderNodeBsdfGlossy = 'ShaderNodeBsdfGlossy'
    ShaderNodeBsdfHair = 'ShaderNodeBsdfHair'
    ShaderNodeBsdfHairPrincipled = 'ShaderNodeBsdfHairPrincipled'
    ShaderNodeBsdfPrincipled = 'ShaderNodeBsdfPrincipled'
    ShaderNodeBsdfRefraction = 'ShaderNodeBsdfRefraction'
    ShaderNodeBsdfToon = 'ShaderNodeBsdfToon'
    ShaderNodeBsdfTranslucent = 'ShaderNodeBsdfTranslucent'
    ShaderNodeBsdfTransparent = 'ShaderNodeBsdfTransparent'
    ShaderNodeBsdfVelvet = 'ShaderNodeBsdfVelvet'
    ShaderNodeBump = 'ShaderNodeBump'
    ShaderNodeCameraData = 'ShaderNodeCameraData'
    ShaderNodeClamp = 'ShaderNodeClamp'
    ShaderNodeCombineHSV = 'ShaderNodeCombineHSV'
    ShaderNodeCombineRGB = 'ShaderNodeCombineRGB'
    ShaderNodeCombineXYZ = 'ShaderNodeCombineXYZ'
    ShaderNodeCustomGroup = 'ShaderNodeCustomGroup'
    ShaderNodeDisplacement = 'ShaderNodeDisplacement'
    ShaderNodeEeveeSpecular = 'ShaderNodeEeveeSpecular'
    ShaderNodeEmission = 'ShaderNodeEmission'
    ShaderNodeFresnel = 'ShaderNodeFresnel'
    ShaderNodeGamma = 'ShaderNodeGamma'
    ShaderNodeGroup = 'ShaderNodeGroup'
    ShaderNodeHairInfo = 'ShaderNodeHairInfo'
    ShaderNodeHoldout = 'ShaderNodeHoldout'
    ShaderNodeHueSaturation = 'ShaderNodeHueSaturation'
    ShaderNodeInvert = 'ShaderNodeInvert'
    ShaderNodeLayerWeight = 'ShaderNodeLayerWeight'
    ShaderNodeLightFalloff = 'ShaderNodeLightFalloff'
    ShaderNodeLightPath = 'ShaderNodeLightPath'
    ShaderNodeMapRange = 'ShaderNodeMapRange'
    ShaderNodeMapping = 'ShaderNodeMapping'
    ShaderNodeMath = 'ShaderNodeMath'
    ShaderNodeMixRGB = 'ShaderNodeMixRGB'
    ShaderNodeMixShader = 'ShaderNodeMixShader'
    ShaderNodeNewGeometry = 'ShaderNodeNewGeometry'
    ShaderNodeNormal = 'ShaderNodeNormal'
    ShaderNodeNormalMap = 'ShaderNodeNormalMap'
    ShaderNodeObjectInfo = 'ShaderNodeObjectInfo'
    ShaderNodeOutputAOV = 'ShaderNodeOutputAOV'
    ShaderNodeOutputLight = 'ShaderNodeOutputLight'
    ShaderNodeOutputLineStyle = 'ShaderNodeOutputLineStyle'
    ShaderNodeOutputMaterial = 'ShaderNodeOutputMaterial'
    ShaderNodeOutputWorld = 'ShaderNodeOutputWorld'
    ShaderNodeParticleInfo = 'ShaderNodeParticleInfo'
    ShaderNodeRGB = 'ShaderNodeRGB'
    ShaderNodeRGBCurve = 'ShaderNodeRGBCurve'
    ShaderNodeRGBToBW = 'ShaderNodeRGBToBW'
    ShaderNodeScript = 'ShaderNodeScript'
    ShaderNodeSeparateHSV = 'ShaderNodeSeparateHSV'
    ShaderNodeSeparateRGB = 'ShaderNodeSeparateRGB'
    ShaderNodeSeparateXYZ = 'ShaderNodeSeparateXYZ'
    ShaderNodeShaderToRGB = 'ShaderNodeShaderToRGB'
    ShaderNodeSqueeze = 'ShaderNodeSqueeze'
    ShaderNodeSubsurfaceScattering = 'ShaderNodeSubsurfaceScattering'
    ShaderNodeTangent = 'ShaderNodeTangent'
    ShaderNodeTexBrick = 'ShaderNodeTexBrick'
    ShaderNodeTexChecker = 'ShaderNodeTexChecker'
    ShaderNodeTexCoord = 'ShaderNodeTexCoord'
    ShaderNodeTexEnvironment = 'ShaderNodeTexEnvironment'
    ShaderNodeTexGradient = 'ShaderNodeTexGradient'
    ShaderNodeTexIES = 'ShaderNodeTexIES'
    ShaderNodeTexImage = 'ShaderNodeTexImage'
    ShaderNodeTexMagic = 'ShaderNodeTexMagic'
    ShaderNodeTexMusgrave = 'ShaderNodeTexMusgrave'
    ShaderNodeTexNoise = 'ShaderNodeTexNoise'
    ShaderNodeTexPointDensity = 'ShaderNodeTexPointDensity'
    ShaderNodeTexSky = 'ShaderNodeTexSky'
    ShaderNodeTexVoronoi = 'ShaderNodeTexVoronoi'
    ShaderNodeTexWave = 'ShaderNodeTexWave'
    ShaderNodeTexWhiteNoise = 'ShaderNodeTexWhiteNoise'
    ShaderNodeUVAlongStroke = 'ShaderNodeUVAlongStroke'
    ShaderNodeUVMap = 'ShaderNodeUVMap'
    ShaderNodeValToRGB = 'ShaderNodeValToRGB'
    ShaderNodeValue = 'ShaderNodeValue'
    ShaderNodeVectorCurve = 'ShaderNodeVectorCurve'
    ShaderNodeVectorDisplacement = 'ShaderNodeVectorDisplacement'
    ShaderNodeVectorMath = 'ShaderNodeVectorMath'
    ShaderNodeVectorRotate = 'ShaderNodeVectorRotate'
    ShaderNodeVectorTransform = 'ShaderNodeVectorTransform'
    ShaderNodeVertexColor = 'ShaderNodeVertexColor'
    ShaderNodeVolumeAbsorption = 'ShaderNodeVolumeAbsorption'
    ShaderNodeVolumeInfo = 'ShaderNodeVolumeInfo'
    ShaderNodeVolumePrincipled = 'ShaderNodeVolumePrincipled'
    ShaderNodeVolumeScatter = 'ShaderNodeVolumeScatter'
    ShaderNodeWavelength = 'ShaderNodeWavelength'
    ShaderNodeWireframe = 'ShaderNodeWireframe'


log_manager = BPYLoggingManager()


class ShaderBase:
    SHADER: str = "Unknown"

    @classmethod
    def all_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in c.all_subclasses()])

    def __init__(self):
        self.logger = log_manager.get_logger(f'{self.SHADER}_handler')
        self.bpy_material: bpy.types.Material = None

    @staticmethod
    def get_missing_texture(texture_name: str, fill_color: tuple = (1.0, 1.0, 1.0, 1.0)):
        assert len(fill_color) == 4, 'Fill color should be in RGBA format'
        if bpy.data.images.get(texture_name, None):
            return bpy.data.images.get(texture_name)
        else:
            image = bpy.data.images.new(texture_name, width=512, height=512, alpha=False)
            image_data = np.full((512 * 512, 4), fill_color, np.float32).flatten()
            if bpy.app.version > (2, 83, 0):
                image.pixels.foreach_set(image_data)
            else:
                image.pixels[:] = image_data
            return image

    def load_texture(self, texture_name, texture_path) -> Optional[bpy.types.Image]:
        pass

    @staticmethod
    def make_texture(texture_name, texture_dimm, texture_data, raw_texture=False):
        image = bpy.data.images.new(texture_name, width=texture_dimm[0], height=texture_dimm[1], alpha=True)
        image.alpha_mode = 'CHANNEL_PACKED'
        image.file_format = 'TARGA'
        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(texture_data.flatten().tolist())
        else:
            image.pixels[:] = texture_data.flatten().tolist()
        image.pack()
        if raw_texture:
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
        return image

    @staticmethod
    def split_to_channels(image):
        if bpy.app.version > (2, 83, 0):
            buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
            image.pixels.foreach_get(buffer)
        else:
            buffer = np.array(image.pixels[:])
        return buffer[0::4], buffer[1::4], buffer[2::4], buffer[3::4],

    def load_texture_or_default(self, file: str, default_color: tuple = (1.0, 1.0, 1.0, 1.0)):
        texture_name = Path(file).stem
        texture = self.load_texture(texture_name, file)
        return texture or self.get_missing_texture(f'missing_{texture_name}', default_color)

    @staticmethod
    def clamp_value(value, min_value=0.0, max_value=1.0):
        return min(max_value, max(value, min_value))

    @staticmethod
    def new_texture_name_with_suffix(old_name, suffix, ext):
        old_name = Path(old_name)
        return f'{old_name.with_name(old_name.stem)}_{suffix}.{ext}'

    def clean_nodes(self):
        for node in self.bpy_material.node_tree.nodes:
            self.bpy_material.node_tree.nodes.remove(node)

    def create_node(self, node_type: str, name: str = None):
        node = self.bpy_material.node_tree.nodes.new(node_type)
        if name:
            node.name = name
            node.label = name
        return node

    def get_node(self, name):
        return self.bpy_material.node_tree.nodes.get(name, None)

    def connect_nodes(self, output_socket, input_socket):
        self.bpy_material.node_tree.links.new(output_socket, input_socket)

    def insert_node(self, output_socket, middle_input_socket, middle_output_socket):
        receivers = []
        for link in output_socket.links:
            receivers.append(link.to_socket)
            self.bpy_material.node_tree.links.remove(link)
        self.connect_nodes(output_socket, middle_input_socket)
        for receiver in receivers:
            self.connect_nodes(middle_output_socket, receiver)

    def create_nodes(self, material_name: str):
        self.logger.info(f'Creating material {repr(material_name)}')
        self.bpy_material = bpy.data.materials.get(material_name, False) or bpy.data.materials.new(material_name)

        if self.bpy_material is None:
            self.logger.error('Failed to get or create material')
            return 'UNKNOWN'

        if self.bpy_material.get('source_loaded'):
            return 'LOADED'

        self.bpy_material.use_nodes = True
        self.clean_nodes()
        self.bpy_material.blend_method = 'HASHED'
        self.bpy_material.shadow_method = 'HASHED'
        self.bpy_material.use_screen_refraction = True
        self.bpy_material.refraction_depth = 0.01
        self.bpy_material['source_loaded'] = True

    def align_nodes(self):
        nodes_iterate(self.bpy_material.node_tree)
        self.bpy_material.node_tree.nodes.update()
