from .base_element import BaseElement
from .game_model import GameModel


class AnimationSet(BaseElement):

    @property
    def controls(self):
        return self._element['controls']

    @property
    def preset_groups(self):
        return self._element['presetGroups']

    @property
    def phoneme_map(self):
        return self._element['phonememap']

    @property
    def operators(self):
        return self._element['operators']

    @property
    def root_control_group(self):
        return self._element['rootControlGroup']

    @property
    def game_model(self):
        return GameModel(self._element['gameModel']) if 'gameModel' in self._element else None
