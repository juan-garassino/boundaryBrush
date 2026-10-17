# GUARDIAN

Automated security / bug-fix ledger for boundaryBrush. Branch: `claude/guardian`.

## Checks

| Check | Command |
|---|---|
| Dependency audit (PyPI-resolvable deps) | `uv export --frozen --no-hashes --all-extras --no-emit-project > req.txt && pip-audit -r req.txt --no-deps --disable-pip` |
| Dependency audit (torch/torchvision, `+cpu` local tag stripped) | `grep ^torch req.txt \| sed 's/+cpu//' > torch.txt && pip-audit -r torch.txt --no-deps --disable-pip` |
| osv-scanner | `osv-scanner scan --lockfile uv.lock` (not usable: no extractor for uv.lock in this version; pip-audit used instead) |
| SAST | `bandit -r src` |
| Secrets (history + tree) | (gitleaks binary from the v8.21.2 GitHub release tarball if not on PATH) `gitleaks detect --source . --log-opts="--all" --redact` and `gitleaks dir . --redact` |
| Risky patterns | `grep -rnE "eval\(\|exec\(\|pickle\|torch\.load\|joblib\|yaml\.load\|shell=True\|verify=False\|0\.0\.0\.0" src tests` |
| GitHub Actions | manual review of `.github/workflows/ci.yml` (SHA pinning, `permissions:`, `${{ github.event.* }}` in `run:`) |
| Dockerfiles | none in repo |
| Tests | `uv run --no-sync pytest -q` (`--no-sync`: plain `uv run` re-tries the blocked torch download) |
| Lint | `uv run --no-sync ruff check .` and `uv run --no-sync ruff format --check .` |
| Types | `pip install mypy && mypy src --strict --ignore-missing-imports` (mypy is not a dev dep; run from a separate install) |
| Coverage | `pip install coverage pytest && PYTHONPATH=src python3 -m coverage run --data-file=<scratch>/.cov --source=src -m pytest -q -p no:cacheprovider && python3 -m coverage report --data-file=<scratch>/.cov` |
| Unmerged feature branches (read-only) | `git grep -nE "torch\.load\|pickle\|from_pretrained\|urlopen\|extractall\|subprocess" origin/feat/v1 -- src` (review notes go to Backlog; never fix on another branch) |

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
| G-010 | low | bug: `cli.main` called `args.func` unconditionally; a subcommand registered without `set_defaults(func=...)` crashed with a bare `AttributeError` | src/boundarybrush/cli.py | fixed (tests `test_subcommand_without_func_is_a_usage_error_not_attributeerror`, `test_every_registered_subcommand_sets_func`, `test_no_subcommand_is_a_usage_error`) | 2026-10-17 |
| G-011 | low | hardening: `BOUNDARYBRUSH_RELEASE_URL` (checkpoint download base) accepted `http://`, `ftp://`, bare hosts -> tamperable weights once download code lands | src/boundarybrush/config.py | fixed; pass 2 re-allowed `file://` for offline mirrors (feat/v1 SHA-256-verifies every download) (tests `test_release_url_must_be_https`, `test_release_url_https_or_local_mirror_accepted`) | 2026-10-17 |
| G-012 | low | bug: `BOUNDARYBRUSH_DEVICE` accepted any string (`gpu`, `auto`, trailing spaces); the spec forbids auto and a typo only failed deep inside torch | src/boundarybrush/config.py | fixed (tests `test_device_typo_rejected_at_startup`, `test_device_accepted`) | 2026-10-17 |
| G-014 | low | bug: a blank `BOUNDARYBRUSH_DATA_DIR` / `RUNS_DIR` / `WEIGHTS_DIR` became `Path("")` == `.`, silently writing data, runs or weights into the current directory | src/boundarybrush/config.py | fixed (test `test_blank_path_env_rejected_instead_of_meaning_cwd`) | 2026-10-17 |
| G-013 | info | mypy `--strict`: 5 errors (untyped field helpers, `Any` returned from `cli.main`) | src/boundarybrush/config.py, cli.py | fixed (mypy --strict clean) | 2026-10-17 |
| G-009 | low | bug: `BOUNDARYBRUSH_NUM_THREADS` / `BOUNDARYBRUSH_IMAGE_SIZE` accepted 0 / negatives and a non-integer gave a bare `int()` error not naming the variable | src/boundarybrush/config.py | fixed (tests `test_bad_int_env_names_the_variable`, `test_non_positive_int_env_rejected`) | 2026-10-15 |

No secrets found (gitleaks history + dir clean). bandit: 0 issues (all severities). No risky patterns in `src/`.

## Metrics

| Metric | Run #1 before | Run #1 after | Run #2 before | Run #2 after |
|---|---|---|---|---|
| Open vulns crit/high/med/low | 2 / 7 / 6 / 13 | 2 / 7 / 6 / 13 | 2 / 7 / 6 / 14 (incl. G-011 found this run) | 2 / 7 / 6 / 13 |
| Open bugs | 1 | 0 | 3 (G-010; G-012 found in pass 2; G-014 in pass 3) | 0 |
| Failing tests | 0 | 0 | 0 | 0 |
| Test count | 2 | 5 | 5 | 34 (12 after pass 1, 28 after pass 2) |
| Ruff lint errors | 0 | 0 | 0 | 0 |
| Ruff format diffs | 0 | 0 | 0 | 0 |
| bandit issues | 0 | 0 | 0 | 0 |
| gitleaks findings | 0 | 0 | 0 | 0 |
| mypy --strict errors | not measured | not measured | 5 | 0 |
| Coverage (src, line) | not measured | not measured | 92% (after pass 1) | 100% |

## Backlog

- G-007/G-008 (apply manually; the guardian push token cannot modify workflows): in
  `.github/workflows/ci.yml` add a top-level `permissions: { contents: read }` block and replace
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
- Review notes for `origin/feat/v1` (read-only, apply when it merges into main):
  - `weights.fetch`: `urllib.request.urlopen(url)` has no `timeout=` (a stalled host hangs the CLI forever);
    add `timeout=60`.
  - `checkpoint.py` / `data.py` rely on `torch.load(..., weights_only=True)`, which torch 2.2.2 does not make
    safe (G-001, CVE-2025-32434); only load files that passed the SHA-256 manifest or were produced locally.
  - `slimsam_provider.from_pretrained(model_id)` has no `revision=` pin; `SamProcessor.from_pretrained` lacks
    `use_safetensors` (processor files are JSON, so low risk) — pin a commit SHA for `nielsr/slimsam-77-uniform`.
  - Merging feat/v1 into this branch will conflict in `config.py` (`device`/`release_url` validators) and
    `cli.py` (`func` guard); keep both validators.
- Consider `persist-credentials: false` on `actions/checkout` and adding `pytest-cov` to the dev extra.
- osv-scanner in this environment cannot parse `uv.lock`; use pip-audit on `uv export` output.

## Learnings

- `feat/v1` (owned by another workflow) already has `weights.py` (SHA-256 manifest) and `checkpoint.py`
  (`torch.load(weights_only=True)`); its tests call `weights.fetch` with `file://` URIs directly, so config
  must not forbid `file://` — check sibling branches before tightening a setting they consume.

- Dependency CVE set unchanged between run #1 and run #2 (20 torch + 6 transformers unique CVEs); pip-audit
  prints duplicate rows per advisory (one per fix-range), so count unique CVE aliases, not rows.
- `uv run` without `--no-sync` fails in this sandbox because it re-attempts the torch wheel download.

- `pip-audit` skips `torch==2.2.2+cpu` ("not found on PyPI"); strip the `+cpu` local tag to audit it.
- The pytorch CPU wheel host (`download-r2.pytorch.org`) may be blocked by the sandbox proxy; the config
  tests run without torch via `--no-install-package torch --no-install-package torchvision`.

## Ideas

1. Add a `safe_load_checkpoint()` helper (weights_only + safetensors preference + sha256 check) before any
   model loading code is written, so the torch.load CVE class is designed out (carried over from run #1).
2. Add a CI job running `pip-audit` on `uv export` output (non-blocking, report-only given the pins).
3. Add `mypy` (strict) and `pytest-cov` to the `dev` extra and the CI test job once the owner fixes CI permissions.
4. Validate `BOUNDARYBRUSH_SAM_MODEL_ID` as `owner/name` (no path separators beyond one `/`, no `..`) so it
   cannot point `from_pretrained` at an arbitrary local directory.

## Log

- 2026-10-15 run #1 — first scan. 26 dependency CVEs (torch 2.2.2, transformers 4.57.6) recorded as open,
  blocked by deliberate platform pins. CI hardening (SHA pins, `permissions:`) prepared but not pushable
  (token lacks `workflow` scope) -> backlog. Fixed config int-env validation (3 new tests). Tests 2 -> 5 passing, lint clean, no secrets.
- 2026-10-17 run #2 — no PR review comments; main unchanged since run #1. Rescan: same 26 dependency CVEs
  (pins unchanged), bandit 0, gitleaks history+tree clean, no risky patterns. Fixed G-010 (CLI subcommand
  without `func` -> clean usage error) and G-011 (release URL must be https). Tests 5 -> 12, lint clean.
  Installed `.git/hooks/pre-push` restricting pushes to `claude/guardian`. CI hardening still blocked (G-007/G-008).
- 2026-10-17 run #2 pass 2 — origin/claude/guardian unchanged. Fixed G-012 (device validation), relaxed G-011 to
  keep `file://` mirrors working (feat/v1 compatibility), mypy --strict 5 -> 0 (G-013), added CLI dispatch and
  `python -m` entrypoint tests: tests 12 -> 28, coverage 92% -> 100%. New checks: mypy, coverage, read-only
  risky-pattern review of feat/v1 (notes in Backlog).
- 2026-10-17 run #2 pass 3 — nothing moved upstream. Fixed G-014 (blank path env vars meant CWD); tests 28 -> 34,
  coverage stays 100%, mypy --strict clean. main holds only the ~130-line scaffold (config + CLI), now fully
  covered; nothing further is fixable on this branch until feat/v1 lands — remaining work is owner-side
  (CI permissions/pins, torch/transformers pins) or feat/v1 review notes in Backlog.
