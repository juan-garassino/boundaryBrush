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


def test_subcommand_dispatches_to_func_and_returns_its_code(monkeypatch):
    real = cli.build_parser
    seen = []

    def build_parser_with_cmd():
        parser = real()
        sub = _subparser_actions(parser)[0].add_parser("hello")
        sub.set_defaults(func=lambda args: seen.append(args.cmd) or 7)
        return parser

    monkeypatch.setattr(cli, "build_parser", build_parser_with_cmd)
    assert cli.main(["-v", "hello"]) == 7
    assert seen == ["hello"]


def test_python_dash_m_entrypoint(monkeypatch, capsys):
    import runpy
    import sys

    from boundarybrush import __version__

    monkeypatch.setattr(sys, "argv", ["boundarybrush", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("boundarybrush", run_name="__main__")
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
