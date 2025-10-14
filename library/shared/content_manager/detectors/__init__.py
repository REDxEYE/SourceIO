from SourceIO.library.shared.content_manager.detectors.content_detector import ContentDetector
from SourceIO.library.shared.content_manager.detectors.cs2 import CS2Detector
from SourceIO.library.shared.content_manager.detectors.csgo import CSGODetector
from SourceIO.library.shared.content_manager.detectors.deadlock import DeadlockDetector
from SourceIO.library.shared.content_manager.detectors.gmod import GModDetector
from SourceIO.library.shared.content_manager.detectors.goldsrc import GoldSrcDetector
from SourceIO.library.shared.content_manager.detectors.hla import HLADetector
from SourceIO.library.shared.content_manager.detectors.quake3 import QuakeIDTech3Detector
from SourceIO.library.shared.content_manager.detectors.infra import InfraDetector
from SourceIO.library.shared.content_manager.detectors.left4dead import Left4DeadDetector
from SourceIO.library.shared.content_manager.detectors.portal2 import Portal2Detector
from SourceIO.library.shared.content_manager.detectors.portal2_revolution import Portal2RevolutionDetector
from SourceIO.library.shared.content_manager.detectors.robot_repair import RobotRepairDetector
from SourceIO.library.shared.content_manager.detectors.sbox import SBoxDetector
from SourceIO.library.shared.content_manager.detectors.sfm import SFMDetector
from SourceIO.library.shared.content_manager.detectors.source1 import Source1Detector
from SourceIO.library.shared.content_manager.detectors.source2 import Source2Detector
from SourceIO.library.shared.content_manager.detectors.sourcemod import SourceMod
from SourceIO.library.shared.content_manager.detectors.swjk2 import StarWarsJediKnights2Detector
from SourceIO.library.shared.content_manager.detectors.titanfall1 import TitanfallDetector
from SourceIO.library.shared.content_manager.detectors.vindictus import VindictusDetector
from SourceIO.library.shared.content_manager.detectors.vampire import VampireDetector
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('game_detector')


def detect_game(path: TinyPath) -> set[ContentProvider]:
    detector_addons: list[ContentDetector] = [
        GoldSrcDetector(),
        SFMDetector(), GModDetector(), InfraDetector(), Left4DeadDetector(),
        Portal2Detector(),
        Portal2RevolutionDetector(), CSGODetector(), SourceMod(), Source1Detector(),
        # VindictusDetector(), TitanfallDetector(),
        SBoxDetector(), CS2Detector(), HLADetector(),
        RobotRepairDetector(), DeadlockDetector(), Source2Detector(),
        StarWarsJediKnights2Detector(), QuakeIDTech3Detector(),
        VampireDetector()
    ]
    content_providers = set()
    for detector in detector_addons:
        results, root_path = detector.scan(path)
        if results:
            logger.info(f"Detected {detector.game()} game: {root_path}")
            content_providers.update(results)
    return content_providers or None
