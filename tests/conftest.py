import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--gpu", 
        action="store_true",
        default=False,
        help="Run CUDA GPU test"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as using GPUs")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_gpu = pytest.mark.skip(reason="GPU tests skipped")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

