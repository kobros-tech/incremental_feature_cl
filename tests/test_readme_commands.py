# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT
"""Every ``python -m incremental_feature_cl...`` command in the README must still parse."""

import argparse
import importlib
import re
import shlex
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


class _Parsed(Exception):
    """Raised right after argparse accepted the arguments (nothing is run)."""


def _commands() -> list[str]:
    text = README.read_text(encoding="utf-8")
    out = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        for line in re.sub(r"\\\n\s*", " ", block).splitlines():
            line = line.split("  #")[0].strip()
            if line.startswith("python -m incremental_feature_cl") and "<" not in line:
                out.append(line)
    return out


COMMANDS = _commands()


def test_readme_documents_commands():
    assert len(COMMANDS) >= 15


@pytest.mark.parametrize("command", COMMANDS)
def test_documented_command_is_accepted_by_its_parser(command, monkeypatch):
    _, _, module, *args = shlex.split(command)
    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, *a, **k):
        original(self, *a, **k)  # exits with status 2 on an unknown flag / bad value
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    mod = importlib.import_module(module)  # the module must exist
    try:
        mod.main(args)
    except _Parsed:
        return
    except TypeError:  # a module whose main() takes no arguments (e.g. math_demo)
        assert not args
        return
    pytest.fail(f"{module} never parsed its arguments")
