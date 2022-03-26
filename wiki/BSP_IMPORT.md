# BSP import options:

* World scale -> Scale factor. Use value 1 for exporting back to source engine, leave as is for more or less human scale.
* Import materials -> Load materials with textures.
* Import cubemaps -> Place EEVEE cubemaps on place of source engine cubemaps.
* Use BlenderVertexLitGeneric shader -> Use BVLG shaders instead of blender shaders.


# BSP import FAQ:

* Q: All props are missing
* A: Entities/Static props are loaded as Empty blender objects with custom data. To load them as models, select handful of them and click "Load entities" in SourceIO panel (Make sure at least one prop object is **active**)

* Q: No materials are loaded
* A: Make sure SourceIO is able to detect game. Make sure you're loading map from game's maps folder. Loading map outside of game's folder may cause bugs with

* Q: Some materials are pink
* A: Most certain type of material is not supported or `Patch` shader failed to find it's target
