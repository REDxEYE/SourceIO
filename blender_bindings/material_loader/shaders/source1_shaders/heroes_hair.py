from .heroes_armor import HeroesArmor


class HeroesHair(HeroesArmor):
    """``heroes_armor`` plus a ``$hairmask`` texture slot."""
    SHADER: str = 'heroes_hair'
    EXTRA_TEXTURE = ('$hairmask', 'hairmask')

    @property
    def hairmask(self):
        return self.extra_texture
