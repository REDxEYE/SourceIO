"""Tests for .ani file loading and animation decompression."""
import os
import pytest
import numpy as np

os.environ['NO_BPY'] = '1'

from pathlib import Path
from SourceIO.library.utils import FileBuffer
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
from SourceIO.library.models.mdl.structs.ani_file import AniFile, read_anim_block_table
from SourceIO.library.models.mdl.load_animations import (
    load_animations_from_mdl, load_all_animations, AnimationData,
)
from SourceIO.library.shared.content_manager import ContentManager

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples"


@pytest.fixture
def dog_animations_mdl():
    path = SAMPLES_DIR / "dog_animations.mdl"
    if not path.exists():
        pytest.skip("dog_animations.mdl not found")
    return path


@pytest.fixture
def dog_gestures_mdl():
    path = SAMPLES_DIR / "dog_gestures.mdl"
    if not path.exists():
        pytest.skip("dog_gestures.mdl not found")
    return path


@pytest.fixture
def dog_animations_ani():
    path = SAMPLES_DIR / "dog_animations.ani"
    if not path.exists():
        pytest.skip("dog_animations.ani not found")
    return path


@pytest.fixture
def dog_mdl():
    path = SAMPLES_DIR / "dog.mdl"
    if not path.exists():
        pytest.skip("dog.mdl not found")
    return path


class TestAniFileReader:
    def test_opens_ani_file(self, dog_animations_ani):
        with FileBuffer(dog_animations_ani) as buf:
            ani = AniFile.from_buffer(buf)
        assert ani.version == 48

    def test_invalid_magic_raises(self):
        from SourceIO.library.utils import MemoryBuffer
        fake_data = b"XXXX\x30\x00\x00\x00" + b"\x00" * 100
        buf = MemoryBuffer(fake_data)
        with pytest.raises(ValueError, match="Not a valid ANI file"):
            AniFile.from_buffer(buf)


class TestBlockTable:
    def test_reads_block_table(self, dog_animations_mdl):
        with FileBuffer(dog_animations_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            blocks = read_anim_block_table(buf, mdl.header.anim_block_offset, mdl.header.anim_block_count)

        assert len(blocks) == 51
        assert blocks[0].data_offset == 0
        assert blocks[0].data_size == 0
        assert blocks[1].data_offset > 0
        assert blocks[1].data_size > 0

    def test_block_offsets_within_ani_size(self, dog_animations_mdl, dog_animations_ani):
        ani_size = dog_animations_ani.stat().st_size
        with FileBuffer(dog_animations_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            blocks = read_anim_block_table(buf, mdl.header.anim_block_offset, mdl.header.anim_block_count)

        for i, block in enumerate(blocks):
            if block.data_size > 0:
                assert block.data_offset < ani_size, \
                    f"Block[{i}] offset beyond ANI file"


class TestAnimationLoading:
    def test_loads_all_from_dog_animations(self, dog_animations_mdl):
        cm = ContentManager()
        model_path = TinyPath(str(dog_animations_mdl))
        cm.scan_for_content(model_path)

        with FileBuffer(dog_animations_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            anims = load_animations_from_mdl(mdl, buf, cm, model_path)

        cm.clean()
        assert len(anims) == 115

    def test_loads_all_from_dog_gestures(self, dog_gestures_mdl):
        cm = ContentManager()
        model_path = TinyPath(str(dog_gestures_mdl))
        cm.scan_for_content(model_path)

        with FileBuffer(dog_gestures_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            anims = load_animations_from_mdl(mdl, buf, cm, model_path)

        cm.clean()
        assert len(anims) == 83

    def test_animation_has_valid_frames(self, dog_animations_mdl):
        cm = ContentManager()
        model_path = TinyPath(str(dog_animations_mdl))
        cm.scan_for_content(model_path)

        with FileBuffer(dog_animations_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            anims = load_animations_from_mdl(mdl, buf, cm, model_path)

        cm.clean()

        for anim in anims[:10]:
            assert anim.frames.shape == (anim.frame_count, 51)
            assert anim.frames.dtype == np.dtype([("pos", np.float32, (3,)), ("rot", np.float32, (4,))])
            assert not np.all(anim.frames["rot"] == 0), f"{anim.name} has all-zero rotations"

    def test_animation_metadata(self, dog_animations_mdl):
        cm = ContentManager()
        model_path = TinyPath(str(dog_animations_mdl))
        cm.scan_for_content(model_path)

        with FileBuffer(dog_animations_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            anims = load_animations_from_mdl(mdl, buf, cm, model_path)

        cm.clean()

        ref = next(a for a in anims if a.name == "@reference")
        assert ref.frame_count == 1
        assert ref.fps == 30.0
        assert not ref.is_looping

    def test_multi_block_animation(self, dog_animations_mdl):
        cm = ContentManager()
        model_path = TinyPath(str(dog_animations_mdl))
        cm.scan_for_content(model_path)

        with FileBuffer(dog_animations_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            anims = load_animations_from_mdl(mdl, buf, cm, model_path)

        cm.clean()

        action = next(a for a in anims if a.name == "@actioninout")
        assert action.frame_count == 171
        assert action.frames.shape == (171, 51)

    def test_inline_animations_still_work(self, dog_mdl):
        cm = ContentManager()
        model_path = TinyPath(str(dog_mdl))
        cm.scan_for_content(model_path)

        with FileBuffer(dog_mdl) as buf:
            mdl = MdlV49.from_buffer(buf)
            anims = load_animations_from_mdl(mdl, buf, cm, model_path)

        cm.clean()
        assert len(anims) >= 1
        ragdoll = next((a for a in anims if a.name == "@ragdoll"), None)
        assert ragdoll is not None
        assert ragdoll.frame_count == 2
