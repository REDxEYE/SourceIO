# SourceIO
Blender SourceIO is an addon for importing\exporting source engine textures/models/maps
Our Discord server https://discord.gg/SF82W6aZ67

Current TODO list -> [TODO.md](TODO.md)


# Used libraries
* [datamodel.py](https://github.com/Artfunkel/BlenderSourceTools/blob/master/io_scene_valvesource/datamodel.py) by [Artfunkel](https://github.com/Artfunkel)
* [BlenderVertexLitGeneric](https://github.com/syborg64/BlenderVertexLitGeneric) Shader nodegroup by [syborg64](https://github.com/syborg64)
* [equilib](https://github.com/haruishi43/equilib) Cubemap to equirectangular converter by [haruishi43](https://github.com/haruishi43/equilib)
# Supported formats (v3.9.3):

## Source 1
| File Type | Contents                          | Import             | Export            |
| ------    | ------                            | ------             | ------            |
| .MDL      | Model                             | :heavy_check_mark: | :x:               |
| .BSP      | Map Files (Compiled)              | :heavy_check_mark: | Not Planned       |
| .VMF      | Map Files (Hammer Format)         | Not Planned        | Not Planned       |
| .VTF      | Textures                          | :heavy_check_mark: | :heavy_check_mark:|
| .VMT      | Materials                         | :heavy_check_mark: | :x:               |

## Source 2
| File Type | Contents                          | Import | Export |
| ------    | ------                            | ------ | ------ |
| .VMDL     | Model                             | :heavy_check_mark: | Not Planned      |
| .VWRLD    | Map Files (Compiled)              | :heavy_check_mark: | Not Planned      |
| .VMAP     | Map Files (Hammer Format)         | Not Planned        | Not Planned      |
| .VTEX     | Textures                          | :heavy_check_mark: | :x:              |
| .VMAT     | Materials                         | :heavy_check_mark: | :x:              |
