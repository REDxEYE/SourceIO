from .heroes_armor import HeroesArmor


class HeroesPBS(HeroesArmor):
    """``heroes_armor`` plus a ``$pbrmap`` texture slot."""
    SHADER: str = 'heroes_pbs'
    EXTRA_TEXTURE = ('$pbrmap', 'PBR')

    @property
    def pbrmap(self):
        return self.extra_texture
