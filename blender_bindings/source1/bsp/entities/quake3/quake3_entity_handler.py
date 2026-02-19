from enum import auto, IntEnum
from functools import partial
from typing import NamedTuple

import bpy
import numpy as np

from SourceIO.blender_bindings.source1.bsp.entities.base_entity_classes import Base, parse_float_vector
from SourceIO.blender_bindings.source1.bsp.entities.abstract_entity_handlers import AbstractEntityHandler
from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_material, add_material
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source1.bsp.bsp_file import IBSPFile
from SourceIO.library.source1.bsp.datatypes.model import QuakeBspModel
from SourceIO.library.source1.bsp.lumps import Quake3BrushLump
from SourceIO.library.source1.bsp.lumps.brush_lump import Quake3BrushSidesLump
from SourceIO.library.source1.bsp.lumps.face_lump import Quake3FaceLump
from SourceIO.library.source1.bsp.lumps.model_lump import Quake3ModelLump
from SourceIO.library.source1.bsp.lumps.plane_lump import Quake3PlaneLump
from SourceIO.library.source1.bsp.lumps.surf_edge_lump import Quake3IndicesLump
from SourceIO.library.source1.bsp.lumps.texture_lump import Quake3TextureInfoLump
from SourceIO.library.source1.bsp.lumps.vertex_lump import Quake3VertexLump
from SourceIO.library.utils import SOURCE1_HAMMER_UNIT_TO_METERS, path_stem


def _bernstein_quadratic(t: np.ndarray) -> np.ndarray:
    """Return Bernstein basis B0..B2 for quadratic Bezier at samples t (shape: [N,3])."""
    t = np.asarray(t, dtype=np.float32)
    return np.stack(((1 - t) ** 2, 2 * t * (1 - t), t ** 2), axis=-1)


def _eval_patch_attrs(control_3x3: np.ndarray, tess: int) -> np.ndarray:
    """Evaluate a 3x3 quadratic Bezier patch over a (tess+1)x(tess+1) grid for all fields present in the dtype."""
    Bu = _bernstein_quadratic(np.linspace(0.0, 1.0, tess + 1, dtype=np.float32))
    Bv = _bernstein_quadratic(np.linspace(0.0, 1.0, tess + 1, dtype=np.float32))
    U = V = tess + 1
    out = np.zeros(U * V, dtype=control_3x3.dtype)

    def eval_field(arr: np.ndarray) -> np.ndarray:
        x = np.tensordot(Bu, arr, axes=(1, 0))
        x = np.tensordot(x, Bv.T, axes=(1, 0))
        x = np.transpose(x, (0, 2, 1)).reshape(U * V, -1)
        return x

    names = control_3x3.dtype.names

    out["pos"] = eval_field(control_3x3["pos"].astype(np.float32))

    if "st" in names:
        out["st"] = eval_field(control_3x3["st"].astype(np.float32))
    if "lightmap" in names:
        out["lightmap"] = eval_field(control_3x3["lightmap"].astype(np.float32))
    if "normal" in names:
        n = eval_field(control_3x3["normal"].astype(np.float32))
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        nn = np.where(nn > 0, nn, 1.0)
        out["normal"] = (n / nn).astype(np.float32)
    if "color" in names:
        color_ = control_3x3["color"]
        vertex_count = color_.shape[0], color_.shape[1]
        layer_count = color_.shape[2]
        comp_count = color_.shape[3] if len(color_.shape)==4 else 1
        c = eval_field(color_.reshape(*vertex_count, layer_count * comp_count).astype(np.float32))
        c = np.clip(np.rint(c), 0, 255).astype(np.uint8)
        c = c.reshape(U * V, layer_count, comp_count)
        out["color"] = c.reshape(out["color"].shape)

    return out


def _grid_indices(w: int, h: int) -> np.ndarray:
    """Return triangle indices for a regular grid of size w x h."""
    tris = []
    for y in range(h - 1):
        r0 = y * w
        r1 = (y + 1) * w
        for x in range(w - 1):
            a = r0 + x
            b = r0 + x + 1
            c = r1 + x
            d = r1 + x + 1
            tris.append([a, c, b])
            tris.append([b, c, d])
    return np.asarray(tris, dtype=np.int32)


def _tessellate_face_patch(vertices_slice: np.ndarray, patch_size_xy: tuple[int, int], tess: int):
    """Tessellate a Quake3 patch face given its control points and size returns (new_vertices, local_indices)."""
    pw, ph = int(patch_size_xy[0]), int(patch_size_xy[1])
    grid = vertices_slice.reshape(ph, pw)
    sx = (pw - 1) // 2
    sy = (ph - 1) // 2
    w = h = tess + 1
    all_verts = []
    all_tris = []
    base = 0
    cell_tris = _grid_indices(w, h)

    for jy in range(sy):
        for ix in range(sx):
            cp = grid[jy * 2:jy * 2 + 3, ix * 2:ix * 2 + 3]
            verts = _eval_patch_attrs(cp, tess)
            all_verts.append(verts)
            all_tris.append(cell_tris + base)
            base += verts.shape[0]

    if not all_verts:
        return np.empty((0,), dtype=vertices_slice.dtype), np.empty((0, 3), dtype=np.int32)

    new_vertices = np.concatenate(all_verts, axis=0)
    local_indices = np.concatenate(all_tris, axis=0)
    return new_vertices, local_indices


def _triangulate_convex(poly: np.ndarray) -> np.ndarray:
    """Return fan triangles for convex polygon poly (Nx3)."""
    if len(poly) < 3:
        return np.empty((0, 3), dtype=np.int32)
    return np.stack([[0, i, i + 1] for i in range(1, len(poly) - 1)], axis=0).astype(np.int32)


def _inside_sense_for_planes(planes, inside_pt: np.ndarray, eps: float = 1e-5) -> list[bool]:
    """Return a bool list where True means use 'dot(n,x) <= d' as inside for that plane; False means '>='."""
    inside_pt = np.asarray(inside_pt, dtype=np.float32)
    senses = []
    for pl in planes:
        n = np.asarray(pl.normal, dtype=np.float32)
        d = float(pl.dist)
        s = float(n @ inside_pt - d)
        senses.append(s <= eps)
    return senses


def _clip_poly_by_plane_signed(poly: np.ndarray, n: np.ndarray, d: float, use_le: bool,
                               eps: float = 1e-6) -> np.ndarray:
    """Clip convex polygon (Nx3) against half-space defined by use_le ? dot(n,x) <= d : dot(n,x) >= d."""
    if poly.shape[0] == 0:
        return poly
    n = np.asarray(n, dtype=np.float32)
    d = float(d)
    s = poly @ n - d
    if not use_le:
        s = -s
    out = []
    N = poly.shape[0]
    for i in range(N):
        a = poly[i]
        b = poly[(i + 1) % N]
        sa = s[i]
        sb = s[(i + 1) % N]
        ina = sa <= eps
        inb = sb <= eps
        if ina and inb:
            out.append(b)
        elif ina and not inb:
            denom = sa - sb
            t = 0.0 if abs(denom) < 1e-12 else sa / (sa - sb)
            out.append(a + t * (b - a))
        elif (not ina) and inb:
            denom = sa - sb
            t = 0.0 if abs(denom) < 1e-12 else sa / (sa - sb)
            p = a + t * (b - a)
            out.append(p)
            out.append(b)
    if not out:
        return np.empty((0, 3), dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)
    keep = [0]
    for i in range(1, out.shape[0]):
        if np.linalg.norm(out[i] - out[keep[-1]]) > eps:
            keep.append(i)
    if len(keep) >= 2 and np.linalg.norm(out[keep[0]] - out[keep[-1]]) <= eps:
        keep = keep[:-1]
    return out[keep]


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors spanning the plane with normal n."""
    n = np.asarray(n, dtype=np.float32)
    n = n / max(np.linalg.norm(n), 1e-12)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(n[2]) < 0.99 else np.array([0.0, 1.0, 0.0],
                                                                                       dtype=np.float32)
    t = np.cross(up, n)
    t /= max(np.linalg.norm(t), 1e-12)
    b = np.cross(n, t)
    return t.astype(np.float32), b.astype(np.float32)


def _polygonize_brush_side(planes: list, senses: list[bool], side_index: int, extent: float,
                           eps: float = 1e-6) -> np.ndarray:
    """Build polygon on planes[side_index] clipped by other planes' inside half-spaces per 'senses'."""
    n0 = np.asarray(planes[side_index].normal, dtype=np.float32)
    d0 = float(planes[side_index].dist)
    c = n0 * d0
    t, b = _plane_basis(n0)
    e = float(extent)
    poly = np.stack([c - t * e - b * e, c + t * e - b * e, c + t * e + b * e, c - t * e + b * e], axis=0).astype(
        np.float32)
    for j, pj in enumerate(planes):
        if j == side_index:
            continue
        poly = _clip_poly_by_plane_signed(poly, pj.normal, pj.dist, senses[j], eps)
        if poly.shape[0] < 3:
            return np.empty((0, 3), dtype=np.float32)
    return poly


class ItemType(IntEnum):
    BAD = auto()
    WEAPON = auto()  # EFX: rotate + upscale + minlight
    AMMO = auto()  # EFX: rotate
    ARMOR = auto()  # EFX: rotate + minlight
    HEALTH = auto()  # EFX: static external sphere + rotating internal
    POWERUP = auto()  # instant on, timer based. EFX: rotate + external ring that rotates
    HOLDABLE = auto()  # single use, holdable item. EFX: rotate + bob
    PERSISTANT_POWERUP = auto()
    TEAM = auto()


class WeaponType(IntEnum):
    NONE = auto()
    GAUNTLET = auto()
    MACHINEGUN = auto()
    SHOTGUN = auto()
    GRENADE_LAUNCHER = auto()
    ROCKET_LAUNCHER = auto()
    LIGHTNING = auto()
    RAILGUN = auto()
    PLASMAGUN = auto()
    BFG = auto()
    GRAPPLING_HOOK = auto()
    NAILGUN = auto()
    PROX_LAUNCHER = auto()
    CHAINGUN = auto()


class PowerUpType(IntEnum):
    NONE = auto()
    QUAD = auto()
    BATTLESUIT = auto()
    HASTE = auto()
    INVIS = auto()
    REGEN = auto()
    FLIGHT = auto()
    REDFLAG = auto()
    BLUEFLAG = auto()
    NEUTRALFLAG = auto()
    SCOUT = auto()
    GUARD = auto()
    DOUBLER = auto()
    AMMOREGEN = auto()
    INVULNERABILITY = auto()


class HoldableType(IntEnum):
    NONE = auto()
    TELEPORTER = auto()
    MEDKIT = auto()
    KAMIKAZE = auto()
    PORTAL = auto()
    INVULNERABILITY = auto()


class Item(NamedTuple):
    classname: str  # entity classname
    pickup_sound: str  # sound to play when item is picked up
    models: tuple  # (world model, icon model, view model, pickup model)
    icon: str  # icon to use in status bar
    pickup_name: str  # name to display in pickup message
    quantity: int  # for ammo how much, or duration of powerup
    item_type: ItemType  # ItemType enum
    tag: int  # for weapons, WeaponType enum; for powerups, PowerUpType enum
    precaches: str  # string of all the media this item will use
    sounds: str  # string of all the sounds this item will use


class QuakeEntityHandler(AbstractEntityHandler):
    entity_lookup_table = {
        "worldspawn": Base,
        "light": Base,
        "misc_model": Base,
    }

    def __init__(self, bsp_file: IBSPFile, content_manager: ContentManager,
                 parent_collection, world_scale: float = SOURCE1_HAMMER_UNIT_TO_METERS,
                 light_scale: float = 0.5):
        super().__init__(bsp_file, content_manager, parent_collection, world_scale, light_scale)
        items = [Item(classname='item_armor_shard', pickup_sound='sound/misc/ar1_pkup.wav',
                      models=('models/powerups/armor/shard.md3', 'models/powerups/armor/shard_sphere.md3', 0, 0),
                      icon='icons/iconr_shard', pickup_name='Armor Shard', quantity=5, item_type=ItemType.ARMOR, tag=0,
                      precaches='', sounds=''),
                 Item(classname='item_armor_combat', pickup_sound='sound/misc/ar2_pkup.wav',
                      models=('models/powerups/armor/armor_yel.md3', 0, 0, 0), icon='icons/iconr_yellow',
                      pickup_name='Armor', quantity=50, item_type=ItemType.ARMOR, tag=0, precaches='', sounds=''),
                 Item(classname='item_armor_body', pickup_sound='sound/misc/ar2_pkup.wav',
                      models=('models/powerups/armor/armor_red.md3', 0, 0, 0), icon='icons/iconr_red',
                      pickup_name='Heavy Armor', quantity=100, item_type=ItemType.ARMOR, tag=0, precaches='',
                      sounds=''),
                 Item(classname='item_health_small', pickup_sound='sound/items/s_health.wav',
                      models=('models/powerups/health/small_cross.md3', 'models/powerups/health/small_sphere.md3', 0,
                              0), icon='icons/iconh_green', pickup_name='5 Health', quantity=5,
                      item_type=ItemType.HEALTH, tag=0, precaches='', sounds=''),
                 Item(classname='item_health', pickup_sound='sound/items/n_health.wav',
                      models=('models/powerups/health/medium_cross.md3', 'models/powerups/health/medium_sphere.md3', 0,
                              0), icon='icons/iconh_yellow', pickup_name='25 Health', quantity=25,
                      item_type=ItemType.HEALTH, tag=0, precaches='', sounds=''),
                 Item(classname='item_health_large', pickup_sound='sound/items/l_health.wav',
                      models=('models/powerups/health/large_cross.md3', 'models/powerups/health/large_sphere.md3', 0,
                              0), icon='icons/iconh_red', pickup_name='50 Health', quantity=50,
                      item_type=ItemType.HEALTH, tag=0, precaches='', sounds=''),
                 Item(classname='item_health_mega', pickup_sound='sound/items/m_health.wav',
                      models=('models/powerups/health/mega_cross.md3', 'models/powerups/health/mega_sphere.md3', 0, 0),
                      icon='icons/iconh_mega', pickup_name='Mega Health', quantity=100, item_type=ItemType.HEALTH,
                      tag=0, precaches='', sounds=''),
                 Item(classname='weapon_gauntlet', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/gauntlet/gauntlet.md3', 0, 0, 0), icon='icons/iconw_gauntlet',
                      pickup_name='Gauntlet', quantity=0, item_type=ItemType.WEAPON, tag=WeaponType.GAUNTLET,
                      precaches='', sounds=''),
                 Item(classname='weapon_shotgun', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/shotgun/shotgun.md3', 0, 0, 0), icon='icons/iconw_shotgun',
                      pickup_name='Shotgun', quantity=10, item_type=ItemType.WEAPON, tag=WeaponType.SHOTGUN,
                      precaches='', sounds=''),
                 Item(classname='weapon_machinegun', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/machinegun/machinegun.md3', 0, 0, 0), icon='icons/iconw_machinegun',
                      pickup_name='Machinegun', quantity=40, item_type=ItemType.WEAPON, tag=WeaponType.MACHINEGUN,
                      precaches='', sounds=''),
                 Item(classname='weapon_grenadelauncher', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/grenadel/grenadel.md3', 0, 0, 0), icon='icons/iconw_grenade',
                      pickup_name='Grenade Launcher', quantity=10, item_type=ItemType.WEAPON,
                      tag=WeaponType.GRENADE_LAUNCHER, precaches='',
                      sounds='sound/weapons/grenade/hgrenb1a.wav sound/weapons/grenade/hgrenb2a.wav'),
                 Item(classname='weapon_rocketlauncher', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/rocketl/rocketl.md3', 0, 0, 0), icon='icons/iconw_rocket',
                      pickup_name='Rocket Launcher', quantity=10, item_type=ItemType.WEAPON,
                      tag=WeaponType.ROCKET_LAUNCHER, precaches='', sounds=''),
                 Item(classname='weapon_lightning', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/lightning/lightning.md3', 0, 0, 0), icon='icons/iconw_lightning',
                      pickup_name='Lightning Gun', quantity=100, item_type=ItemType.WEAPON, tag=WeaponType.LIGHTNING,
                      precaches='', sounds=''),
                 Item(classname='weapon_railgun', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/railgun/railgun.md3', 0, 0, 0), icon='icons/iconw_railgun',
                      pickup_name='Railgun', quantity=10, item_type=ItemType.WEAPON, tag=WeaponType.RAILGUN,
                      precaches='', sounds=''),
                 Item(classname='weapon_plasmagun', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/plasma/plasma.md3', 0, 0, 0), icon='icons/iconw_plasma',
                      pickup_name='Plasma Gun', quantity=50, item_type=ItemType.WEAPON, tag=WeaponType.PLASMAGUN,
                      precaches='', sounds=''),
                 Item(classname='weapon_bfg', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/bfg/bfg.md3', 0, 0, 0), icon='icons/iconw_bfg', pickup_name='BFG10K',
                      quantity=20, item_type=ItemType.WEAPON, tag=WeaponType.BFG, precaches='', sounds=''),
                 Item(classname='weapon_grapplinghook', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons2/grapple/grapple.md3', 0, 0, 0), icon='icons/iconw_grapple',
                      pickup_name='Grappling Hook', quantity=0, item_type=ItemType.WEAPON,
                      tag=WeaponType.GRAPPLING_HOOK, precaches='', sounds=''),
                 Item(classname='ammo_shells', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/shotgunam.md3', 0, 0, 0), icon='icons/icona_shotgun',
                      pickup_name='Shells', quantity=10, item_type=ItemType.AMMO, tag=WeaponType.SHOTGUN, precaches='',
                      sounds=''),
                 Item(classname='ammo_bullets', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/machinegunam.md3', 0, 0, 0), icon='icons/icona_machinegun',
                      pickup_name='Bullets', quantity=50, item_type=ItemType.AMMO, tag=WeaponType.MACHINEGUN,
                      precaches='', sounds=''),
                 Item(classname='ammo_grenades', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/grenadeam.md3', 0, 0, 0), icon='icons/icona_grenade',
                      pickup_name='Grenades', quantity=5, item_type=ItemType.AMMO, tag=WeaponType.GRENADE_LAUNCHER,
                      precaches='', sounds=''),
                 Item(classname='ammo_cells', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/plasmaam.md3', 0, 0, 0), icon='icons/icona_plasma',
                      pickup_name='Cells', quantity=30, item_type=ItemType.AMMO, tag=WeaponType.PLASMAGUN, precaches='',
                      sounds=''),
                 Item(classname='ammo_lightning', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/lightningam.md3', 0, 0, 0), icon='icons/icona_lightning',
                      pickup_name='Lightning', quantity=60, item_type=ItemType.AMMO, tag=WeaponType.LIGHTNING,
                      precaches='', sounds=''),
                 Item(classname='ammo_rockets', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/rocketam.md3', 0, 0, 0), icon='icons/icona_rocket',
                      pickup_name='Rockets', quantity=5, item_type=ItemType.AMMO, tag=WeaponType.ROCKET_LAUNCHER,
                      precaches='', sounds=''),
                 Item(classname='ammo_slugs', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/railgunam.md3', 0, 0, 0), icon='icons/icona_railgun',
                      pickup_name='Slugs', quantity=10, item_type=ItemType.AMMO, tag=WeaponType.RAILGUN, precaches='',
                      sounds=''),
                 Item(classname='ammo_bfg', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/bfgam.md3', 0, 0, 0), icon='icons/icona_bfg',
                      pickup_name='Bfg Ammo', quantity=15, item_type=ItemType.AMMO, tag=WeaponType.BFG, precaches='',
                      sounds=''),
                 Item(classname='holdable_teleporter', pickup_sound='sound/items/holdable.wav',
                      models=('models/powerups/holdable/teleporter.md3', 0, 0, 0), icon='icons/teleporter',
                      pickup_name='Personal Teleporter', quantity=60, item_type=ItemType.HOLDABLE,
                      tag=HoldableType.TELEPORTER, precaches='', sounds=''),
                 Item(classname='holdable_medkit', pickup_sound='sound/items/holdable.wav',
                      models=('models/powerups/holdable/medkit.md3', 'models/powerups/holdable/medkit_sphere.md3', 0,
                              0), icon='icons/medkit', pickup_name='Medkit', quantity=60, item_type=ItemType.HOLDABLE,
                      tag=HoldableType.MEDKIT, precaches='', sounds='sound/items/use_medkit.wav'),
                 Item(classname='item_quad', pickup_sound='sound/items/quaddamage.wav',
                      models=('models/powerups/instant/quad.md3', 'models/powerups/instant/quad_ring.md3', 0, 0),
                      icon='icons/quad', pickup_name='Quad Damage', quantity=30, item_type=ItemType.POWERUP,
                      tag=PowerUpType.QUAD, precaches='', sounds='sound/items/damage2.wav sound/items/damage3.wav'),
                 Item(classname='item_enviro', pickup_sound='sound/items/protect.wav',
                      models=('models/powerups/instant/enviro.md3', 'models/powerups/instant/enviro_ring.md3', 0, 0),
                      icon='icons/envirosuit', pickup_name='Battle Suit', quantity=30, item_type=ItemType.POWERUP,
                      tag=PowerUpType.BATTLESUIT, precaches='',
                      sounds='sound/items/airout.wav sound/items/protect3.wav'),
                 Item(classname='item_haste', pickup_sound='sound/items/haste.wav',
                      models=('models/powerups/instant/haste.md3', 'models/powerups/instant/haste_ring.md3', 0, 0),
                      icon='icons/haste', pickup_name='Speed', quantity=30, item_type=ItemType.POWERUP,
                      tag=PowerUpType.HASTE, precaches='', sounds=''),
                 Item(classname='item_invis', pickup_sound='sound/items/invisibility.wav',
                      models=('models/powerups/instant/invis.md3', 'models/powerups/instant/invis_ring.md3', 0, 0),
                      icon='icons/invis', pickup_name='Invisibility', quantity=30, item_type=ItemType.POWERUP,
                      tag=PowerUpType.INVIS, precaches='', sounds=''),
                 Item(classname='item_regen', pickup_sound='sound/items/regeneration.wav',
                      models=('models/powerups/instant/regen.md3', 'models/powerups/instant/regen_ring.md3', 0, 0),
                      icon='icons/regen', pickup_name='Regeneration', quantity=30, item_type=ItemType.POWERUP,
                      tag=PowerUpType.REGEN, precaches='', sounds='sound/items/regen.wav'),
                 Item(classname='item_flight', pickup_sound='sound/items/flight.wav',
                      models=('models/powerups/instant/flight.md3', 'models/powerups/instant/flight_ring.md3', 0, 0),
                      icon='icons/flight', pickup_name='Flight', quantity=60, item_type=ItemType.POWERUP,
                      tag=PowerUpType.FLIGHT, precaches='', sounds='sound/items/flight.wav'),
                 Item(classname='team_CTF_redflag', pickup_sound='', models=('models/flags/r_flag.md3', 0, 0, 0),
                      icon='icons/iconf_red1', pickup_name='Red Flag', quantity=0, item_type=ItemType.TEAM,
                      tag=PowerUpType.REDFLAG, precaches='', sounds=''),
                 Item(classname='team_CTF_blueflag', pickup_sound='', models=('models/flags/b_flag.md3', 0, 0, 0),
                      icon='icons/iconf_blu1', pickup_name='Blue Flag', quantity=0, item_type=ItemType.TEAM,
                      tag=PowerUpType.BLUEFLAG, precaches='', sounds=''),
                 Item(classname='holdable_kamikaze', pickup_sound='sound/items/holdable.wav',
                      models=('models/powerups/kamikazi.md3', 0, 0, 0), icon='icons/kamikaze', pickup_name='Kamikaze',
                      quantity=60, item_type=ItemType.HOLDABLE, tag=HoldableType.KAMIKAZE, precaches='',
                      sounds='sound/items/kamikazerespawn.wav'),
                 Item(classname='holdable_portal', pickup_sound='sound/items/holdable.wav',
                      models=('models/powerups/holdable/porter.md3', 0, 0, 0), icon='icons/portal',
                      pickup_name='Portal', quantity=60, item_type=ItemType.HOLDABLE, tag=HoldableType.PORTAL,
                      precaches='', sounds=''),
                 Item(classname='holdable_invulnerability', pickup_sound='sound/items/holdable.wav',
                      models=('models/powerups/holdable/invulnerability.md3', 0, 0, 0), icon='icons/invulnerability',
                      pickup_name='Invulnerability', quantity=60, item_type=ItemType.HOLDABLE,
                      tag=HoldableType.INVULNERABILITY, precaches='', sounds=''),
                 Item(classname='ammo_nails', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/nailgunam.md3', 0, 0, 0), icon='icons/icona_nailgun',
                      pickup_name='Nails', quantity=20, item_type=ItemType.AMMO, tag=WeaponType.NAILGUN, precaches='',
                      sounds=''),
                 Item(classname='ammo_mines', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/proxmineam.md3', 0, 0, 0), icon='icons/icona_proxlauncher',
                      pickup_name='Proximity Mines', quantity=10, item_type=ItemType.AMMO, tag=WeaponType.PROX_LAUNCHER,
                      precaches='', sounds=''),
                 Item(classname='ammo_belt', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/ammo/chaingunam.md3', 0, 0, 0), icon='icons/icona_chaingun',
                      pickup_name='Chaingun Belt', quantity=100, item_type=ItemType.AMMO, tag=WeaponType.CHAINGUN,
                      precaches='', sounds=''),
                 Item(classname='item_scout', pickup_sound='sound/items/scout.wav',
                      models=('models/powerups/scout.md3', 0, 0, 0), icon='icons/scout', pickup_name='Scout',
                      quantity=30, item_type=ItemType.PERSISTANT_POWERUP, tag=PowerUpType.SCOUT, precaches='',
                      sounds=''),
                 Item(classname='item_guard', pickup_sound='sound/items/guard.wav',
                      models=('models/powerups/guard.md3', 0, 0, 0), icon='icons/guard', pickup_name='Guard',
                      quantity=30, item_type=ItemType.PERSISTANT_POWERUP, tag=PowerUpType.GUARD, precaches='',
                      sounds=''),
                 Item(classname='item_doubler', pickup_sound='sound/items/doubler.wav',
                      models=('models/powerups/doubler.md3', 0, 0, 0), icon='icons/doubler', pickup_name='Doubler',
                      quantity=30, item_type=ItemType.PERSISTANT_POWERUP, tag=PowerUpType.DOUBLER, precaches='',
                      sounds=''),
                 Item(classname='item_ammoregen', pickup_sound='sound/items/ammoregen.wav',
                      models=('models/powerups/ammo.md3', 0, 0, 0), icon='icons/ammo_regen', pickup_name='Ammo Regen',
                      quantity=30, item_type=ItemType.PERSISTANT_POWERUP, tag=PowerUpType.AMMOREGEN, precaches='',
                      sounds=''),
                 Item(classname='team_CTF_neutralflag', pickup_sound='', models=('models/flags/n_flag.md3', 0, 0, 0),
                      icon='icons/iconf_neutral1', pickup_name='Neutral Flag', quantity=0, item_type=ItemType.TEAM,
                      tag=PowerUpType.NEUTRALFLAG, precaches='', sounds=''),
                 Item(classname='item_redcube', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/orb/r_orb.md3', 0, 0, 0), icon='icons/iconh_rorb',
                      pickup_name='Red Cube', quantity=0, item_type=ItemType.TEAM, tag=0, precaches='', sounds=''),
                 Item(classname='item_bluecube', pickup_sound='sound/misc/am_pkup.wav',
                      models=('models/powerups/orb/b_orb.md3', 0, 0, 0), icon='icons/iconh_borb',
                      pickup_name='Blue Cube', quantity=0, item_type=ItemType.TEAM, tag=0, precaches='', sounds=''),
                 Item(classname='weapon_nailgun', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons/nailgun/nailgun.md3', 0, 0, 0), icon='icons/iconw_nailgun',
                      pickup_name='Nailgun', quantity=10, item_type=ItemType.WEAPON, tag=WeaponType.NAILGUN,
                      precaches='', sounds=''),
                 Item(classname='weapon_prox_launcher', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons/proxmine/proxmine.md3', 0, 0, 0), icon='icons/iconw_proxlauncher',
                      pickup_name='Prox Launcher', quantity=5, item_type=ItemType.WEAPON, tag=WeaponType.PROX_LAUNCHER,
                      precaches='',
                      sounds='sound/weapons/proxmine/wstbtick.wav sound/weapons/proxmine/wstbactv.wav sound/weapons/proxmine/wstbimpl.wav sound/weapons/proxmine/wstbimpm.wav sound/weapons/proxmine/wstbimpd.wav sound/weapons/proxmine/wstbactv.wav'),
                 Item(classname='weapon_chaingun', pickup_sound='sound/misc/w_pkup.wav',
                      models=('models/weapons/vulcan/vulcan.md3', 0, 0, 0), icon='icons/iconw_chaingun',
                      pickup_name='Chaingun', quantity=80, item_type=ItemType.WEAPON, tag=WeaponType.CHAINGUN,
                      precaches='', sounds='sound/weapons/vulcan/wvulwind.wav')]
        for item in items:
            self.entity_lookup_table[item.classname] = Base
            setattr(self, "handle_" + item.classname,
                    partial(self._handle_model_entity, item.classname, "item", item.models[0]))

        point_entities: list[tuple[str, str]] = [
            ("info_player_start", "info"),
            ("info_player_deathmatch", "info"),
            ("info_player_intermission", "info"),
            ("info_null", "info"),
            ("info_notnull", "info"),
            ("info_camp", "info"),
            ("target_give", "target"),
            ("target_remove_powerups", "target"),
            ("target_delay", "target"),
            ("target_speaker", "target"),
            ("target_print", "target"),
            ("target_laser", "target"),
            ("target_score", "target"),
            ("target_teleporter", "target"),
            ("target_relay", "target"),
            ("target_kill", "target"),
            ("target_position", "target"),
            ("target_location", "target"),
            ("target_push", "target"),
            ("team_CTF_redplayer", "team"),
            ("team_CTF_blueplayer", "team"),
            ("team_CTF_redspawn", "team"),
            ("team_CTF_bluespawn", "team"),
            ("team_redobelisk", "team"),
            ("team_blueobelisk", "team"),
            ("team_neutralobelisk", "team"),
            ("item_botroam", "item"),
            ("func_timer", "func"),

            ("misc_teleporter_dest", "misc"),
            ("misc_portal_surface", "misc"),
            ("misc_portal_camera", "misc"),

        ]
        for classname, category in point_entities:
            self.entity_lookup_table[classname] = Base
            setattr(self, "handle_" + classname, partial(self._handle_point_entity, classname, category))

        brush_entities: list[tuple[str, str]] = [
            ("func_plat", "func"),
            ("func_button", "func"),
            ("func_door", "func"),
            ("func_static", "func"),
            ("func_rotating", "func"),
            ("func_bobbing", "func"),
            ("func_pendulum", "func"),
            ("func_train", "func"),

            ("trigger_once", "triggers"),
            ("trigger_multiple", "triggers"),
            ("trigger_push", "triggers"),
            ("trigger_teleport", "triggers"),
            ("trigger_hurt", "triggers"),
        ]
        for classname, category in brush_entities:
            self.entity_lookup_table[classname] = Base
            setattr(self, "handle_" + classname, partial(self._handle_brush_entity, classname, category))

    @staticmethod
    def _get_entity_name(entity: Base) -> str:
        raw_data = entity._raw_data
        name = raw_data.get("targetname", "")
        return name or f'{entity.class_name}_{entity.hammer_id}'

    def _handle_model_entity(self, classname, category, model_path, entity: Base, entity_raw: dict):
        if "model" not in entity_raw:
            entity_raw["model"] = model_path
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._set_single_angle(obj, float(entity_raw.get("angle", "0")))
        self._put_into_collection(classname, obj, category)

    def _handle_point_entity(self, classname, category, entity: Base, entity_raw: dict):
        assert "model" not in entity_raw, "Point entity has model key"
        obj = bpy.data.objects.new(self._get_entity_name(entity), None)
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_single_angle(obj, float(entity_raw.get("angle", "0")))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection(classname, obj, category)

    def _handle_brush_entity(self, classname, category, entity: Base, entity_raw: dict):
        assert "model" in entity_raw, "Brush entity missing model key"
        assert entity_raw["model"].startswith('*'), "Brush entity model key not a brush model"

        obj = self._load_brush_model(int(entity_raw["model"][1:]), self._get_entity_name(entity))
        if not obj:
            return
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        # self._set_single_angle(obj, float(entity_raw.get("angle", "0")))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection(classname, obj, category)

    def handle_worldspawn(self, entity: Base, entity_raw: dict):
        world = self._load_brush_model(0, 'world_geometry')
        if not world:
            return
        self._world_geometry_name = world.name
        self._set_entity_data(world, {'entity': entity_raw})
        self.parent_collection.objects.link(world)

    def handle_light(self, entity, entity_raw: dict):
        def to_linear(value):
            if value <= 0.0404482362771082:
                return value / 12.92
            else:
                return pow(((value + 0.055) / 1.055), 2.4)

        def srgb_to_linear(color):
            return (to_linear(color[0]),
                    to_linear(color[1]),
                    to_linear(color[2]))

        light_data = bpy.data.lights.new(name=self._get_entity_name(entity), type='POINT')
        light_data.energy = float(entity_raw.get("light", "100")) * self.light_scale
        light_data.color = srgb_to_linear(parse_float_vector(entity_raw.get("_color", "1 1 1")))
        obj = bpy.data.objects.new(self._get_entity_name(entity), light_data)
        self._set_location(obj, parse_float_vector(entity_raw.get("origin", "0 0 0")))
        self._set_single_angle(obj, float(entity_raw.get("angle", "0")))
        self._set_entity_data(obj, {'entity': entity_raw})
        self._put_into_collection('light', obj, 'light')

    def handle_misc_model(self, entity, entity_raw: dict):
        obj = self._handle_entity_with_model(entity, entity_raw)
        self._put_into_collection('misc_model', obj, 'misc')

    def _load_brush_model(self, model_id, model_name):
        bsp = self._bsp
        assert isinstance(bsp, IBSPFile)
        models_lump: Quake3ModelLump | None = bsp.get_lump("LUMP_MODELS")

        model = models_lump.models[model_id]

        if model.face_count:
            return self._handle_face_brush_model(bsp, model, model_name)
        elif model.brush_count:
            return self._handle_brsuh_model(bsp, model, model_name)
        return None

    def _handle_face_brush_model(self, bsp: IBSPFile, model: QuakeBspModel, model_name) -> bpy.types.Object:
        tex_info_lump: Quake3TextureInfoLump | None = bsp.get_lump("LUMP_TEXINFO")
        faces_lump: Quake3FaceLump | None = bsp.get_lump("LUMP_FACES")
        vertices_lump: Quake3VertexLump | None = bsp.get_lump("LUMP_DRAWVERTS")
        indices_lump: Quake3IndicesLump | None = bsp.get_lump("LUMP_DRAWINDEXES")

        unique_materials = []
        material_ids = []
        vertices = []
        indices = []

        faces = faces_lump.faces[model.face_offset:model.face_offset + model.face_count]
        for face in faces:
            if face.texture_id not in unique_materials:
                unique_materials.append(face.texture_id)
            vertex_offset = face.vertex_offset
            vertex_count = face.vertex_count
            mesh_vert_offset = face.index_offset
            mesh_vert_count = face.indices_count
            face_vertices = vertices_lump.vertices[vertex_offset:vertex_offset + vertex_count]
            if face.surface_type == 1:
                face_indices = indices_lump.indices[mesh_vert_offset:mesh_vert_offset + mesh_vert_count].reshape(-1,
                                                                                                                 3)
                indices.extend((face_indices + len(vertices)).tolist())
                vertices.extend(face_vertices)
                material_ids.extend([unique_materials.index(face.texture_id)] * face_indices.shape[0])
            elif face.surface_type == 2:
                tess = 5
                patch_verts, patch_tris = _tessellate_face_patch(face_vertices, (face.size[0], face.size[1]), tess)
                indices.extend((patch_tris + len(vertices)).tolist())
                vertices.extend(patch_verts)
                material_ids.extend([unique_materials.index(face.texture_id)] * patch_tris.shape[0])
            elif face.surface_type == 3:
                face_indices = indices_lump.indices[mesh_vert_offset:mesh_vert_offset + mesh_vert_count].reshape(-1,
                                                                                                                 3)
                indices.extend((face_indices + len(vertices)).tolist())
                vertices.extend(face_vertices)
                material_ids.extend([unique_materials.index(face.texture_id)] * face_indices.shape[0])
            else:
                # self.logger.warn(f"Unsupported face type {face.type} in model")
                continue

        # indices = np.asarray(indices).reshape((-1, 3))
        vertices = np.asarray(vertices)
        if len(indices) == 0:
            return None
        mesh_data = bpy.data.meshes.new(f"{model_name}_MESH")
        mesh_obj = bpy.data.objects.new(model_name, mesh_data)

        mesh_data.from_pydata(vertices["pos"] * self.scale, [], indices)

        for mat_id in unique_materials:
            mat_path = tex_info_lump.texture_info[mat_id].name
            material = get_or_create_material(path_stem(mat_path), mat_path)
            add_material(material, mesh_obj)
        mesh_data.polygons.foreach_set('material_index', material_ids)

        mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons), np.uint32))
        mesh_data.normals_split_custom_set_from_vertices(vertices['normal'] * -1)

        vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
        mesh_data.loops.foreach_get('vertex_index', vertex_indices)

        vc = mesh_data.vertex_colors.new()
        colors = vertices["color"][vertex_indices].astype(np.float32) / 255
        vc.data.foreach_set('color', colors.flatten())

        uv_data = mesh_data.uv_layers.new()
        uvs = vertices['st']
        uvs[:, 1] = 1 - uvs[:, 1]
        uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

        return mesh_obj

    def _handle_brsuh_model(self, bsp: IBSPFile, model: QuakeBspModel, model_name):
        tex_info_lump: Quake3TextureInfoLump | None = bsp.get_lump("LUMP_TEXINFO")
        brushes_lump: Quake3BrushLump = bsp.get_lump("LUMP_BRUSHES")
        brushsides_lump: Quake3BrushSidesLump = bsp.get_lump("LUMP_BRUSHSIDES")
        planes_lump: Quake3PlaneLump = bsp.get_lump("LUMP_PLANES")
        if not (brushes_lump and brushsides_lump and planes_lump):
            self.logger.warn("Missing brush/brushside/plane lumps")
            return None

        brushes = brushes_lump.brushes[model.brush_offset:model.brush_offset + model.brush_count]
        planes_all = planes_lump.planes

        coords = []
        tris = []
        face_mats = []
        uniq_mats = []

        center = (np.array(model.mins, dtype=np.float32) + np.array(model.maxs, dtype=np.float32)) * 0.5
        extent = max(8192.0, float(np.linalg.norm(np.array(model.maxs) - np.array(model.mins)) * 8.0))

        for brush in brushes:
            sides = brushsides_lump.brush_sides[brush.side_offset:brush.side_offset + brush.side_count]
            planes = [planes_all[s.plane_id] for s in sides]
            senses = _inside_sense_for_planes(planes, center)

            for si, side in enumerate(sides):
                shader_id = int(side.texture_id)
                if shader_id not in uniq_mats:
                    uniq_mats.append(shader_id)
                poly = _polygonize_brush_side(planes, senses, si, extent)
                if poly.shape[0] < 3:
                    continue
                base = len(coords)
                coords.extend(poly.tolist())
                tris.extend(
                    (np.stack([[0, i, i + 1] for i in range(1, poly.shape[0] - 1)], axis=0) + base).tolist())
                face_mats.extend([uniq_mats.index(shader_id)] * (poly.shape[0] - 2))

        if not tris:
            return None

        mesh_data = bpy.data.meshes.new(f"{model_name}_MESH")
        mesh_obj = bpy.data.objects.new(model_name, mesh_data)

        scaled = (np.asarray(coords, dtype=np.float32) * self.scale).tolist()
        mesh_data.from_pydata(scaled, [], tris)

        for mat_id in uniq_mats:
            mat_path = tex_info_lump.texture_info[mat_id].name
            material = get_or_create_material(path_stem(mat_path), mat_path)
            add_material(material, mesh_obj)

        for i, p in enumerate(mesh_data.polygons):
            p.material_index = face_mats[i]
        return mesh_obj
