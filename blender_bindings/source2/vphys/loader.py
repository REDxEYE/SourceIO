import bpy

from ....library.source2.resource_types import ValveCompiledPhysics
from ....library.utils.math_utilities import SOURCE2_HAMMER_UNIT_TO_METERS


class ValveCompiledPhysicsLoader(ValveCompiledPhysics):
    def __init__(self, path_or_file, scale=SOURCE2_HAMMER_UNIT_TO_METERS):
        super().__init__(path_or_file)
        self.scale = scale

    def build_mesh(self):
        meshes = []
        for sphere in self.spheres:
            print(sphere)
        for capsule in self.capsules:
            print(capsule)
        for mesh in self.meshes:
            print(mesh)
        for name, polygons, vertices in self.hulls:
            mesh_data = bpy.data.meshes.new(name=f'{name}_mesh')
            mesh_obj = bpy.data.objects.new(name=name, object_data=mesh_data)

            mesh_data.from_pydata(vertices * self.scale, [], polygons)
            mesh_data.update()
            meshes.append(mesh_obj)
        return meshes
