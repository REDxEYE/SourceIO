from ....shared.content_providers.content_manager import ContentManager
from ....utils import datamodel
from .session import Session


def open_session(path):
    ContentManager().scan_for_content(path)
    dmx_session = datamodel.load(path)
    return Session(dmx_session.root)
