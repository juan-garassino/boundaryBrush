# GUARDIAN

Automated security / bug-fix ledger for boundaryBrush. Branch: `claude/guardian`.

## Checks

| Check | Command |
|---|---|
| Dependency audit (PyPI-resolvable deps) | `uv export --frozen --no-hashes --all-extras --no-emit-project > req.txt && pip-audit -r req.txt --no-deps --disable-pip` |
| Dependency audit (torch/torchvision, `+cpu` local tag stripped) | `grep ^torch req.txt \| sed 's/+cpu//' > torch.txt && pip-audit -r torch.txt --no-deps --disable-pip` |
| osv-scanner | `osv-scanner scan --lockfile uv.lock` (not usable: no extractor for uv.lock in this version; pip-audit used instead) |
| SAST | `bandit -r src` |
| Secrets (history + tree) | `gitleaks detect --source . --log-opts="--all" --redact` and `gitleaks dir . --redact` |
| Risky patterns | `grep -rnE "eval\(\|exec\(\|pickle\|torch\.load\|joblib\|yaml\.load\|shell=True\|verify=False\|0\.0\.0\.0" src tests` |
| GitHub Actions | manual review of `.github/workflows/ci.yml` (SHA pinning, `permissions:`, `${{ github.event.* }}` in `run:`) |
| Dockerfiles | none in repo |
| Tests | `uv run pytest -q` |
| Lint | `uv run ruff check .` and `uv run ruff format --check .` |

Environment note: `download-r2.pytorch.org` was unreachable through the egress proxy, so the venv
was built with `uv sync --frozen --extra dev --no-install-package torch --no-install-package torchvision`.
The current test suite does not import torch.

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | critical | dep vuln: torch 2.2.2 CVE-2025-32434 (`torch.load(weights_only=True)` RCE, fix 2.6.0) | pyproject.toml / uv.lock | open (wontfix for now, see Backlog) | 2026-10-15 |
| G-002 | critical | dep vuln: transformers 4.57.6 CVE-2026-4372 (RCE, fix 5.3.0) | pyproject.toml (`sam` extra) | open (blocked by `<5` pin) | 2026-10-15 |
| G-003 | high | dep vulns: transformers 4.57.6 CVE-2025-14929, CVE-2026-1839, CVE-2026-5241, CVE-2026-9856, CVE-2026-80047 (deserialization / remote-code / path-traversal file write) | pyproject.toml (`sam` extra) | open (blocked by `<5` pin) | 2026-10-15 |
| G-004 | high | dep vulns: torch 2.2.2 CVE-2024-48063 (disputed RemoteModule RCE), CVE-2026-24747 (fix 2.10.0) | pyproject.toml / uv.lock | open (blocked by torch pin) | 2026-10-15 |
| G-005 | medium | dep vulns: torch 2.2.2 CVE-2025-2148, -2998, -2999, -3000, -3001, CVE-2025-55558 (local memory corruption) | pyproject.toml / uv.lock | open (blocked by torch pin) | 2026-10-15 |
| G-006 | low | dep vulns: torch 2.2.2, 11 local DoS / incorrect-result CVEs (CVE-2025-2149, -2953, -3730, -46148, -55551, -55552, -55553, -55554, -55557, -55560, CVE-2026-4538) | pyproject.toml / uv.lock | open (blocked by torch pin) | 2026-10-15 |
| G-007 | low | CI: third-party actions pinned to mutable tags | .github/workflows/ci.yml | open (fix blocked: push token lacks `workflow` scope, see Backlog) | 2026-10-15 |
| G-008 | low | CI: no `permissions:` block (default token scope) | .github/workflows/ci.yml | open (fix blocked: push token lacks `workflow` scope, see Backlog) | 2026-10-15 |
| G-009 | low | bug: `BOUNDARYBRUSH_NUM_THREADS` / `BOUNDARYBRUSH_IMAGE_SIZE` accepted 0 / negatives and a non-integer gave a bare `int()` error not naming the variable | src/boundarybrush/config.py | fixed (tests `test_bad_int_env_names_the_variable`, `test_non_positive_int_env_rejected`) | 2026-10-15 |

No secrets found (gitleaks history + dir clean). bandit: 0 issues (all severities). No risky patterns in `src/`.

## Metrics

| Metric | Before | After |
|---|---|---|
| Open vulns crit/high/med/low | 2 / 7 / 6 / 13 | 2 / 7 / 6 / 13 |
| Open bugs | 1 | 0 |
| Failing tests | 0 | 0 |
| Test count | 2 | 5 |
| Ruff lint errors | 0 | 0 |
| Ruff format diffs | 0 | 0 |
| Coverage | not measured (pytest-cov not a dev dep) | not measured |

## Backlog

- G-007/G-008 (apply manually; the guardian push token cannot modify workflows): in
  `.github/workflows/ci.yml` add a top-level `permissions:\n  contents: read` block and replace
  `actions/checkout@v4` with `actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0` and
  `astral-sh/setup-uv@v3` with `astral-sh/setup-uv@caf0cab7a618c569241d31dcd442f54681755d39 # v3.2.4`
  (SHAs resolved via `git ls-remote` on 2026-10-15).

- G-001..G-006: torch is pinned to 2.2.2 deliberately (last macOS x86_64 wheel) and transformers to `<5`
  (transformers 5 needs torch>=2.5). A bump cannot happen without dropping Intel-mac support. Until then,
  when checkpoint loading lands: only load weights you produced or downloaded over a verified release URL,
  prefer `safetensors`, and never `torch.load` user-supplied files. Pin a `revision=` when calling
  `from_pretrained` for `sam_model_id`, and never pass `trust_remote_code=True`.
- `release_url` / `sam_model_id` are env-overridable; when the download code lands, verify a checksum of
  release assets before loading them.
- `cli.main` calls `args.func(args)`; every future subparser must `set_defaults(func=...)` or `main` will
  raise `AttributeError`.
- Consider `persist-credentials: false` on `actions/checkout` and adding `pytest-cov` to the dev extra.
- osv-scanner in this environment cannot parse `uv.lock`; use pip-audit on `uv export` output.

## Learnings

- `pip-audit` skips `torch==2.2.2+cpu` ("not found on PyPI"); strip the `+cpu` local tag to audit it.
- The pytorch CPU wheel host (`download-r2.pytorch.org`) may be blocked by the sandbox proxy; the config
  tests run without torch via `--no-install-package torch --no-install-package torchvision`.

## Ideas

1. Add a `safe_load_checkpoint()` helper (weights_only + safetensors preference + sha256 check) before any
   model loading code is written, so the torch.load CVE class is designed out.
2. Add a CI job running `pip-audit` on `uv export` output (non-blocking, report-only given the pins).
3. Add a CLI test asserting every registered subcommand sets `func`.

## Log

- 2026-10-15 run #1 — first scan. 26 dependency CVEs (torch 2.2.2, transformers 4.57.6) recorded as open,
  blocked by deliberate platform pins. CI hardening (SHA pins, `permissions:`) prepared but not pushable
  (token lacks `workflow` scope) -> backlog. Fixed config int-env validation (3 new tests). Tests 2 -> 5 passing, lint clean, no secrets.
