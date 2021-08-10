import pytest


@pytest.mark.xfail
def test_srr_file() -> None:
    """
    Performs a smoke test that srr with a file actually produces a roughly correct reward signal.
    """
    raise NotImplementedError()


@pytest.mark.xfail
def test_srr_webcam() -> None:
    """
    Performs a smoke test that srr with a (mock) webcam actually produces a roughly correct reward signal.
    """
    raise NotImplementedError()
