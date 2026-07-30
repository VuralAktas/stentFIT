"""
Make ``src_oop/stentfit`` the importable ``stentfit`` for the OOP tests.

Both ``src/stentfit`` and ``src_oop/stentfit`` use the import name ``stentfit``,
so only one can be importable at a time. Putting ``src_oop`` at the front of
``sys.path`` shadows the editable install of the old package (whose ``.pth``
file appends ``src/`` at the end), and :func:`_assert_new_package` fails loudly
rather than silently testing the old code if that ever stops being true.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_OOP = REPO_ROOT / "src_oop"

# Ahead of the editable install of the old package, and ahead of anything else.
sys.path.insert(0, str(SRC_OOP))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _assert_new_package() -> None:
    """
    Fail the session if ``stentfit`` resolves to the old procedural package.

    :raises RuntimeError: If the imported ``stentfit`` does not live under
        ``src_oop/``.
    """
    import stentfit

    where = Path(stentfit.__file__).resolve()
    if SRC_OOP not in where.parents:
        raise RuntimeError(
            f"imported stentfit is {where}, not the src_oop package. "
            f"Uninstall the old package (pip uninstall stentfit) or run pytest "
            f"with PYTHONPATH={SRC_OOP}.")


_assert_new_package()


def pytest_report_header(config) -> str:
    """:returns: The resolved ``stentfit`` location, shown in the pytest header."""
    import stentfit

    return f"stentfit under test: {Path(stentfit.__file__).resolve()}"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """:returns: The repository root, for resolving input data paths."""
    return REPO_ROOT
