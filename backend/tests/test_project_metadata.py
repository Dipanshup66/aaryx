"""The version and the changelog must agree, so a release can never be half-updated."""

import re
import tomllib
from pathlib import Path

import app
from app.core.config import Settings
from app.main import create_app

BACKEND = Path(__file__).resolve().parents[1]
CHANGELOG = BACKEND.parent / "CHANGELOG.md"


def _pyproject_version() -> str:
    data = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_version_is_plain_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", app.__version__)


def test_version_is_the_same_everywhere() -> None:
    assert app.__version__ == _pyproject_version()
    assert create_app(Settings(_env_file=None)).version == app.__version__


def test_changelog_has_an_entry_for_the_current_version() -> None:
    text = CHANGELOG.read_text(encoding="utf-8")

    assert re.search(
        rf"^## \[{re.escape(app.__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", text, re.MULTILINE
    )


def test_changelog_keeps_an_unreleased_section_first() -> None:
    headings = re.findall(r"^## \[(.+?)\]", CHANGELOG.read_text(encoding="utf-8"), re.MULTILINE)

    assert headings[0] == "Unreleased"
    assert len(headings) == len(set(headings)), "duplicate version heading"


def test_changelog_documents_the_versioning_rule() -> None:
    text = CHANGELOG.read_text(encoding="utf-8")

    assert "## Versioning rule" in text and "1.0.0" in text
