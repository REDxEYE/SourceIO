from SourceIO.library.utils.tiny_path import TinyPath
from .cs2 import CS2Detector
from .gmod import GModDetector
from .goldsrc import GoldSrcDetector
from .hla import HLADetector
from .idtech3 import IDTech3Detector
from .robot_repair import RobotRepairDetector
from .sbox import SBoxDetector
from .sfm import SFMDetector
from .source1 import Source1Detector
from .source2 import Source2Detector
from .sourcemod import SourceMod
from .titanfall1 import TitanfallDetector
from .vindictus import VindictusDetector
from ..provider import ContentProvider


def detect_game(path: TinyPath) -> list[ContentProvider]:
    detector_addons = [GoldSrcDetector(), IDTech3Detector(),
                       SFMDetector(), GModDetector(), SourceMod(),
                       VindictusDetector(), TitanfallDetector(),
                       SBoxDetector(), CS2Detector(), HLADetector(), RobotRepairDetector(),
                       Source1Detector(), Source2Detector()]

    for detector in detector_addons:
        results = detector.scan(path)
        if results:
            return results
