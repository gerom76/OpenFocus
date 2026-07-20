<!-- rtk-instructions v2 -->
# RTK

Full RTK reference lives in the global `~/.claude/CLAUDE.md` — don't duplicate it here.
Golden rule applies: prefix every shell command with `rtk`, including inside `&&` chains.

Python-specific filters relevant to this project:

```bash
rtk pytest              # test failures only
rtk ruff                # lint/format, compact
rtk mypy                # type errors grouped by file
rtk pip install         # compact install output (auto-detects uv)
rtk format              # black / ruff format check
```
<!-- /rtk-instructions -->
