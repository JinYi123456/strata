"""The deployment contract, read out of ``deploy/deploy.sh`` and ``deploy/deploy.py``.

Both storage writers in this project - :class:`service.store.S3Store` for the
HTTP service, and ``deploy/handler.py`` for the Lambda - have to write items into
a table that a shell script creates. When the two drifted apart, the managed
path failed on its first write in production and no test noticed, because the
table only exists in an AWS account.

So the script is the source of truth here, and these helpers parse it. Changing
``deploy.sh`` without changing the code that writes to it fails the suite, which
is the only way this contract can be checked without an account.

``deploy.py`` is parsed for the same reason. It resolves wheels for a different
machine than the one that ran the measurements, so the two dependency sets are
allowed to differ in *flavour* - Lambda has no X11, so the package ships the
headless OpenCV - but not in *version*, because a version drift would silently
invalidate every number the report quotes from the local run.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SH = ROOT / "deploy" / "deploy.sh"
DEPLOY_PY = ROOT / "deploy" / "deploy.py"


def _create_table_block() -> str:
    text = DEPLOY_SH.read_text(encoding="utf-8")
    assert "aws dynamodb create-table" in text, "deploy.sh no longer creates a table"
    return text.split("aws dynamodb create-table", 1)[1].split("--billing-mode")[0]


def declared_key_schema() -> list[tuple[str, str]]:
    """``[(attribute, role)]`` in declaration order, e.g. ``[("pk","HASH"),("sk","RANGE")]``."""
    return re.findall(
        r"AttributeName=([A-Za-z0-9_]+),KeyType=(HASH|RANGE)", _create_table_block()
    )


def declared_attribute_types() -> dict[str, str]:
    return dict(
        re.findall(
            r"AttributeName=([A-Za-z0-9_]+),AttributeType=(S|N)", _create_table_block()
        )
    )


def granted_actions(service: str) -> set[str]:
    """The DynamoDB/S3 actions the role policy in ``deploy.sh`` allows."""
    text = DEPLOY_SH.read_text(encoding="utf-8").replace('\\"', '"')
    allowed: set[str] = set()
    for group in re.findall(r'"Action":\[([^\]]*)\]', text):
        allowed.update(re.findall(rf'"{service}:([A-Za-z]+)"', group))
    for single in re.findall(rf'"Action":"{service}:([A-Za-z]+)"', text):
        allowed.add(single)
    return allowed


def key_attributes() -> list[str]:
    return [attr for attr, _role in declared_key_schema()]


def deployed_requirements() -> dict[str, str]:
    """The exact pins ``deploy/deploy.py`` resolves into the Lambda package."""
    text = DEPLOY_PY.read_text(encoding="utf-8")
    assert "REQUIREMENTS = [" in text, "deploy.py no longer declares its dependency set"
    block = text.split("REQUIREMENTS = [", 1)[1].split("]", 1)[0]
    return dict(re.findall(r'"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.]*)"', block))
