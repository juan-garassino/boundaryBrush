from __future__ import annotations

import argparse

import pytest

from boundarybrush import cli


def _subparser_actions(parser: argparse.ArgumentParser):
    return [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_every_registered_subcommand_sets_func():
    for action in _subparser_actions(cli.build_parser()):
        for name, sub in action.choices.items():
            assert sub.get_default("func") is not None, f"subcommand {name!r} has no func"


def test_subcommand_without_func_is_a_usage_error_not_attributeerror(monkeypatch, capsys):
    real = cli.build_parser

    def build_parser_with_orphan():
        parser = real()
        _subparser_actions(parser)[0].add_parser("orphan")
        return parser

    monkeypatch.setattr(cli, "build_parser", build_parser_with_orphan)
    with pytest.raises(SystemExit) as exc:
        cli.main(["orphan"])
    assert exc.value.code == 2
    assert "orphan" in capsys.readouterr().err
