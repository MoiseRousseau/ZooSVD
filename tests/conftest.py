import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--gpu", 
        action="store_true",
        default=False,
        help="Run GPU test"
    )
    parser.addoption(
        "--real",
        action="store_true",
        default=False,
        help="Run tests on real matrix"
    )
    parser.addoption(
        "--complex",
        action="store_true",
        default=False,
        help="Run tests on complex matrix"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: mark test as using GPUs")
    config.addinivalue_line("markers", "real: mark test as real")
    config.addinivalue_line("markers", "complex: mark test as complex")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip = pytest.mark.skip(reason="GPU tests skipped")
        for item in items:
            if "gpu" in item.keywords and not config.getoption("--gpu"):
                item.add_marker(skip)
    if not config.getoption("--complex"):
        skip = pytest.mark.skip(reason="Complex SVD skipped")
        for item in items:
            if "complex" in item.keywords and not config.getoption("--complex"):
                item.add_marker(skip)
    if not config.getoption("--real"):
        skip = pytest.mark.skip(reason="Real SVD skipped")
        for item in items:
            if "real" in item.keywords and not config.getoption("--real"):
                item.add_marker(skip)
