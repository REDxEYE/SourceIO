from .heroes_armor import HeroesArmor


class HeroesFaceskin(HeroesArmor):
    """Functionally identical to ``heroes_armor``; only the shader name differs."""
    SHADER: str = 'heroes_faceskin'
