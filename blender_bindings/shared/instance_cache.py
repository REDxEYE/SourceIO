import json
from typing import Optional

import bpy

from ...library.utils.singleton import SingletonMeta

SERIALIZED_DATA_NAME = "_INSTANCE_CACHE.json"


class InstanceCache(metaclass=SingletonMeta):
    def __init__(self):
        self._collection_names = {}

    def add_collection(self, model_path: str, collection: bpy.types.Collection):
        self._collection_names[model_path] = collection.name

    def get_collection(self, model_path: str) -> Optional[str]:
        if model_path in self._collection_names:
            return self._collection_names[model_path]

    def load_from_blender_file(self):
        file = bpy.data.texts.get(SERIALIZED_DATA_NAME, None)
        if file is None:
            return
        try:
            data = file.as_string()
            self._collection_names.update(json.loads(data))
        except Exception as e:
            print(f"Failed to load cache, probably malformed: {e}")

    def save_to_blender_file(self):
        file = bpy.data.texts.get(SERIALIZED_DATA_NAME, None)
        if file is None:
            file = bpy.data.texts.new(SERIALIZED_DATA_NAME)
        file.clear()
        file.write(json.dumps(self._collection_names, ensure_ascii=False, indent=False))

    def wipe(self):
        self._collection_names.clear()
