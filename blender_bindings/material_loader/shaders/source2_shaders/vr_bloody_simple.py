from .vr_complex import VrComplex


class VRBloodySimple(VrComplex):
    SHADER: str = 'vr_bloody_simple.vfx'

    @property
    def metalness(self):
        return self._material_resource.get_int_property('F_METALNESS_TEXTURE', 1)