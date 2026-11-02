---
name: kaggle-curobo-test
description: "Run cuRobo's GPU tests (pytest under curobo/tests) or benchmarks on a Kaggle GPU kernel (2x T4) from a session without a GPU. Use whenever cuRobo code needs to be verified on CUDA: before merging, after changing kernels, solvers, collision or anything a GPU-only test covers, or when asked to run the cuRobo test suite. Builds on the kaggle-gpu-test skill and adds a prebuilt warm environment (venv plus compiled Warp kernels) and a two-GPU split that cut the full suite from 45 to about 12 minutes."
---

# Testing cuRobo on Kaggle

Most cuRobo tests need CUDA. A session without a GPU runs them on Kaggle through the generic
`kaggle-gpu-test` skill (`scripts/kaggle_run.sh`), which clones the repository at a pushed commit,
runs a command on a Kaggle GPU kernel and brings back the log. This skill adds what makes that
fast for cuRobo:

- **Warm environment `curobo-warm`.** A Kaggle kernel named `kgt-env-curobo-warm` holds an
  `env.tar` with a Python 3.13 venv (torch, warp-lang and every other dependency) and the Warp
  kernel cache filled by a full test run. `--env curobo-warm` unpacks it, so the setup is only
  `pip install -e . --no-deps` and unchanged Warp modules are not compiled again. Compiling is
  what makes a cold run slow: the first mapper test alone spends 7 minutes compiling.
- **Two-GPU split.** `scripts/run_split_suite.sh` runs half of the suite on each T4.

Measured full-suite times (4435 tests, 2x T4):

| Setup | Time |
| --- | --- |
| One GPU, cold (`--setup "pip install -e '.[cu12,dev]'"`) | 45 min |
| Two GPUs, cold | 23 min |
| Two GPUs, `--env curobo-warm` | 10.5 min, plus about 1.5 min to restore and install |

## Before a run

1. **Kaggle credentials.** `kaggle_run.sh` reads `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and
   `KAGGLE_KEY`) from the environment. Never write a token into the repository, a file or a commit
   message. If none is set, ask the user to add it to the environment's secrets.
2. **Find the runner.** It ships with the `kaggle-gpu-test` skill:
   `KGT=$(find ~/.claude/skills -name kaggle_run.sh -path '*kaggle-gpu-test*' | head -1)`.
   Read that skill's SKILL.md once for its options and exit codes.
3. **Commit and push.** Kaggle clones the public GitHub repository, not this working tree. Pass
   the **full** commit SHA as `--ref` (`git rev-parse HEAD`); a short SHA fails to clone.
4. **Run from the repository root**, and do not commit `.kaggle-gpu-test/` (add it to
   `.git/info/exclude`).

## Run tests

Targeted tests, while developing (the usual loop):

```bash
bash "$KGT" run --env curobo-warm --repo https://github.com/kthfan/curobo --ref "$(git rev-parse HEAD)" \
  --slug curobo-check --setup "pip install -q -e . --no-deps" \
  --cmd "python -m pytest -n 0 -p no:cacheprovider -q -rfE curobo/tests/_src/collision" --timeout 60m
```

The full suite, before merging into `dev/main`:

```bash
bash "$KGT" run --env curobo-warm --repo https://github.com/kthfan/curobo --ref "$(git rev-parse HEAD)" \
  --slug curobo-full --setup "pip install -q -e . --no-deps" \
  --cmd "bash .claude/skills/kaggle-curobo-test/scripts/run_split_suite.sh" --timeout 90m
```

- Always pass `-n 0`. `pyproject.toml` sets `-n 4` for xdist, but the workers would share one GPU.
- For long runs add `--detach`, then `bash "$KGT" wait --slug S` and `bash "$KGT" fetch --slug S`
  (`wait` takes no `--poll`). Run the wait in the background; it can take an hour.
- Read `SUITE_A`/`SUITE_B` and the `FAILED` lines from the summary; the full logs are in
  `.kaggle-gpu-test/out/<slug>-*/test_output.log`. Add `--durations=0 --durations-min=1.0` to the
  pytest arguments to find slow tests.
- Each slug is one Kaggle kernel. Use different slugs to run several jobs at the same time.
- A commit that predates the script (before it reached `dev/main`) has no
  `run_split_suite.sh`: pass the script itself, e.g. `--cmd "echo <base64> | base64 -d > /tmp/s.sh
  && bash /tmp/s.sh"`.

Without the warm environment (for example in a Kaggle account that does not have it), use the
image's own Python, which already has torch: `--setup "pip install -q -e '.[cu12,dev]'"` and
no `--env`.

## Rebuild the warm environment

Rebuild when dependencies change (`pyproject.toml`, a new torch or warp-lang), when `--env
curobo-warm` reports "not found", or when many Warp modules changed so that runs compile again.
It takes about 35 minutes:

```bash
bash "$KGT" run --detach --python 3.13 --repo https://github.com/kthfan/curobo \
  --ref "$(git rev-parse HEAD)" --slug kgt-env-curobo-warm \
  --setup "pip install -q -e '.[cu12-torch,dev]'" \
  --cmd "bash .claude/skills/kaggle-curobo-test/scripts/build_warm_env.sh" --timeout 2h
bash "$KGT" wait --slug kgt-env-curobo-warm
```

- The slug must be `kgt-env-<name>`: `--env <name>` mounts that kernel's output.
- Do **not** `fetch` this kernel: its output is an 8.5 GB `env.tar`. Read only the log with
  `kaggle kernels output <user>/kgt-env-curobo-warm -p <dir> --file-pattern 'result\.json|test_output\.log'`
  and check `SUITE_A`, `SUITE_B` and the `ENV_TAR` size.
- The environment belongs to the Kaggle account that built it and is private. Every session that
  uses the same credentials can use it; another account has to build its own.

## Pitfalls

- Shell pipelines that end in `grep` exit 1 when nothing matches, which the runner reports as a
  failed run. End a `--cmd` with the command whose exit code should count.
- Tests for kernels that changed in the commit compile again on every run, even with the warm
  environment: only unchanged modules hit the cache.
- `curobo/tests/conftest.py` must return `None` from `pytest_ignore_collect` for paths it does not
  skip; returning `False` forces collection and silently disables `--ignore`, which the split
  relies on.
- Tests that are GPU-only run here; do not try to make them pass on the CPU.
