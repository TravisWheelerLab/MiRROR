import uuid
import itertools as it
import pprint

from mirror.annotation import AnnotationParams, AnnotationResult, annotate
from mirror.util import load_config
from mirror.session import setup

from .shared import TEST_PEAKS, ANNO_CFG

def test_annotate():
    cfg = load_config("/home/user/Projects/MiRROR/params", config_name="setup_from_session")
    cfg.session.name = str(uuid.uuid4()).split('-')[0]
    session = setup(cfg)
    for i, peaks in enumerate(TEST_PEAKS):
        print(i)
        pprint.pprint(peaks)
        res = annotate(peaks, session.anno_params, session.pair_targets, session.boundary_targets, session.reverse_boundary_targets,)
        pprint.pprint(res)
