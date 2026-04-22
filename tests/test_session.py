import pathlib, os, uuid
from mirror.session import setup
from mirror.util import load_config

def test_session_setup():
    cfg = load_config("/home/user/Projects/MiRROR/params", config_name="setup_from_transcriptome")
    cfg.session.name = str(uuid.uuid4()).split('-')[0]
    session = setup(cfg)
    assert set(os.listdir(session.session_dir)) == set(['config.yaml','test.reverse.fasta','forward.sufr','reverse.sufr'])

def test_session_increment():
    session_name = str(uuid.uuid4()).split('-')[0]
    cfg = load_config("/home/user/Projects/MiRROR/params", config_name="setup_from_transcriptome")
    cfg.session.name = session_name
    
    session = setup(cfg)
    assert set(os.listdir(session.session_dir)) == set(['config.yaml','test.reverse.fasta','forward.sufr','reverse.sufr'])
    # first session should contain all the files it had to create, plus the config.
    
    new_cfg_dir = str(pathlib.Path(cfg.session.dir).absolute() / cfg.session.name)
    cfg = load_config(new_cfg_dir, config_name="config.yaml")
    session = setup(cfg)
    assert os.listdir(session.session_dir) == ['config.yaml']
    # second session is already be pointed to the first session's files, so this dir should only contain the config.
    assert session.session_dir.stem.endswith("_1")
    # second session should have the _1 suffix.

    new_cfg_dir = str(pathlib.Path(cfg.session.dir).absolute() / cfg.session.name)
    cfg = load_config(new_cfg_dir, config_name="config.yaml")
    session = setup(cfg)
    assert os.listdir(session.session_dir) == ['config.yaml']
    # likewise with the third.
    assert session.session_dir.stem.endswith("_2")
    # third session should have the _2 suffix.
