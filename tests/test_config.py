from irvision.utils.config import PROJECT_ROOT, load_config


def test_config_loads_with_required_sections():
    cfg = load_config()
    for section in ("project", "paths", "dataset", "alignment", "preprocessing", "patches", "baseline", "evaluation", "model", "training", "inference", "semantic", "optional"):
        assert section in cfg, f"missing config section: {section}"
    assert cfg["patches"]["size"] == 256
    assert set(cfg["dataset"]["bands"]) == {"red", "green", "blue", "ir", "qa"}


def test_paths_are_absolute_inside_repo():
    cfg = load_config()
    for path in cfg["paths"].values():
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents
