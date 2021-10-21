from ....shared.content_providers.content_manager import ContentManager
from .base_element import BaseElement
from .dag import Dag
from .transform import Transform


class GameModel(BaseElement):

    @property
    def transform(self):
        return Transform(self._element['transform'])

    @property
    def children(self):
        return [Dag(elem) for elem in self._element['children']]

    @property
    def model_name(self):
        return self._element['modelName']

    @property
    def model_file(self):
        return ContentManager().find_file(self.model_name)

    @property
    def bones(self):
        return [Transform(elem) for elem in self._element['bones']]

    @property
    def flex_names(self):
        return [elem for elem in self._element['flexnames']]

    @property
    def flex_weights(self):
        return [elem for elem in self._element['flexWeights']]
