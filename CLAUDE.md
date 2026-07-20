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

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
