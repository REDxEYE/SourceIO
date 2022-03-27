# MDL import options:

* Write QC -> Generate QC file. Saved into text block in **.blend** file.
* ~~Load animations~~ -> **Unimplemented!**
* ~~Unique material names~~ -> **Unimplemented!**
* Create drivers for flexes -> Create HMW like drivers for model. Controls are located in SourceIO panel on right side of 3d viewport.
* Group meshes by bodygroup -> Create collections for each bodygroup. 
* Import materials -> Load materials with textures. 
* Use BlenderVertexLitGeneric shader -> Use BVLG shaders instead of blender shaders.
* World scale -> Scale factor. Use value 1 for exporting back to source engine, leave as is for more or less human scale.


# BSP import FAQ:

* Q: No materials are loaded
* A: Make sure SourceIO is able to detect game. Make sure you're loading model from game's models folder. Loading models outside of game's folder may cause bugs with

* Q: Some materials are pink
* A: Most certain type of material is not supported or `Patch` shader failed to find it's target

* Q: I cannot control flexes
* A: If you checked `Create drivers for flexes` option flexes should be controlled via special block in SourceIO panel
