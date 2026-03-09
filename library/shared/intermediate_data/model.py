from dataclasses import dataclass, replace

from SourceIO.library.shared.intermediate_data import Bone, Attachment
from SourceIO.library.shared.intermediate_data.common import Matrix4x4
from SourceIO.library.shared.intermediate_data.mesh import Mesh
from SourceIO.library.shared.intermediate_data.physics import PhysicsMesh


@dataclass(slots=True, frozen=True)
class BodyPart:
    name: str
    lods: list[tuple[int, list[Mesh]]]  # (lod_level, meshes)


@dataclass(slots=True, frozen=True)
class BodyGroup:
    name: str
    parts: list[BodyPart | None]


@dataclass(slots=True, frozen=True)
class Material:
    name: str
    fullpath: str


@dataclass(slots=True, frozen=True)
class Model:
    name: str
    bodygroups: list[BodyGroup]
    bones: list[Bone]
    attachments: list[Attachment]
    materials: list[Material]

    physics: list[PhysicsMesh]

    transform: Matrix4x4

    @classmethod
    def without_bodygroups(cls, name: str, meshes: list[Mesh], bones: list[Bone],
                           attachments: list[Attachment], materials: list[Material],
                           transform: Matrix4x4, physics: list[PhysicsMesh] | None = None):
        part = BodyPart(name="default", lods=[(0, meshes)])
        return cls(
            name=name,
            bodygroups=[BodyGroup(name="default", parts=[part])],
            bones=bones,
            attachments=attachments,
            materials=materials,
            transform=transform,
            physics=physics or [],
        )

    def with_physics(self, physics_meshes: list[PhysicsMesh]):
        return replace(self, physics=physics_meshes)
