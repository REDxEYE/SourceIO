from SourceIO.library.shared.content_manager.detectors.cs2 import CS2Detector
from SourceIO.library.shared.content_manager.detectors.csgo import CSGODetector
from SourceIO.library.shared.content_manager.detectors.deadlock import DeadlockDetector
from SourceIO.library.shared.content_manager.detectors.gmod import GModDetector
from SourceIO.library.shared.content_manager.detectors.goldsrc import GoldSrcDetector
from SourceIO.library.shared.content_manager.detectors.hla import HLADetector
from SourceIO.library.shared.content_manager.detectors.idtech3 import IDTech3Detector
from SourceIO.library.shared.content_manager.detectors.infra import InfraDetector
from SourceIO.library.shared.content_manager.detectors.portal2 import Portal2Detector
from SourceIO.library.shared.content_manager.detectors.portal2_revolution import Portal2RevolutionDetector
from SourceIO.library.shared.content_manager.detectors.robot_repair import RobotRepairDetector
from SourceIO.library.shared.content_manager.detectors.sbox import SBoxDetector
from SourceIO.library.shared.content_manager.detectors.sfm import SFMDetector
from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.detectors.sourcemod import SourceMod
from SourceIO.library.shared.content_manager.detectors.titanfall1 import TitanfallDetector
from SourceIO.library.shared.content_manager.detectors.vindictus import VindictusDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils.tiny_path import TinyPath


def detect_game(path: TinyPath) -> list[ContentProvider]:
    detector_addons = [GoldSrcDetector(),
                       SFMDetector(), GModDetector(), InfraDetector(), Portal2Detector(),Portal2RevolutionDetector(), SourceMod(), CSGODetector(),
                       # VindictusDetector(), TitanfallDetector(),
                       SBoxDetector(), CS2Detector(), HLADetector(), RobotRepairDetector(),
                       DeadlockDetector(), Source1Detector(), Source2Detector(), IDTech3Detector()]

    for detector in detector_addons:
        results = detector.scan(path)
        if results:
            return results
