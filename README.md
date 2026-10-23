# maestro-review

**AI code review for GitHub pull requests that runs as a CI check.**
`maestro-review` reads a PR's diff, asks Claude (via the
[Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/)) for a structured
review, and posts the findings as inline review comments. The check fails while
new findings are unaddressed, so a PR stays blocked until someone fixes,
disputes or tracks each one.

[![CI](https://github.com/thomas-sodon/maestro-review/actions/workflows/ci.yml/badge.svg)](https://github.com/thomas-sodon/maestro-review/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

<!-- demo screenshot goes here -->

It was extracted from [Maestro](https://github.com/thomas-sodon/maestro-agent),
an autonomous engineering agent, where it reviews every pull request the agent
opens. It has no dependency on Maestro and works on any GitHub repository.

## Quickstart (GitHub Action, 2 minutes)

1. Add an `ANTHROPIC_API_KEY` (or a `CLAUDE_CODE_OAUTH_TOKEN`) as a repository
   secret.
2. Commit this as `.github/workflows/ai-review.yml`:

```yaml
name: AI review
on:
  pull_request:

permissions:
  contents: read
  pull-requests: write
  issues: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      # Optional: fail if earlier findings are still unanswered. Run it before
      # the review so it sees the state as of this push.
      - uses: thomas-sodon/maestro-review@main
        with:
          mode: replies
      - uses: thomas-sodon/maestro-review@main
        with:
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          # or: claude-code-oauth-token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}
```

3. Open a pull request. Findings appear as inline review comments, and the job
   fails when something new was posted.

The example uses `@main` because no release is tagged yet; pin a full commit
SHA in your own workflows (see [Security model](#security-model)). All action
inputs are listed in [docs/REFERENCE.md](docs/REFERENCE.md#github-action-inputs).

## How it works

- **Fetch:** reads the PR diff and metadata through the `gh` CLI, truncated to a
  configurable size and file count.
- **Review:** sends the diff to Claude with read-only `Read`/`Glob`/`Grep`
  tools in the checked-out repository, so the model can look at surrounding
  code. Each finding has a severity: `bug`, `suggestion` or `nit`.
- **De-duplicate:** drops findings already tracked by an issue or matching an
  existing review thread, re-flags a bug whose thread was resolved while the
  problem persists, and escalates disputed findings to a GitHub issue instead
  of re-posting them.
- **Post and gate:** posts one review (inline comments, plus an *Additional
  findings* list for findings that cannot be anchored to a diff line) and exits non-zero when anything
  new was posted. A separate reply-coverage mode (`mode: replies`) makes no
  Claude call and fails while any finding is still unanswered.

## Security model

- **The model can only read.** The review query allows only the `Read`, `Glob`
  and `Grep` tools; no shell, edit or network tool is allowed, and the model
  never calls GitHub itself: fetching the PR, posting comments and opening issues are
  done by the tool's own Python code through the `gh` CLI.
- **Credentials stay in the environment.** The action passes your Anthropic
  credential and the GitHub token to the reviewer only as environment variables
  (`ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN`, `GH_TOKEN`). Empty
  credentials are unset rather than passed as blanks, and `mode: replies`
  needs no Anthropic credential at all.
- **Least-privilege GitHub token.** The job needs `contents: read`,
  `pull-requests: write` (to post reviews) and `issues: write` (only for the
  disputed-finding escalation). It uses the workflow's `github.token` by
  default.
- **Pin what holds your secret.** Because the action receives an Anthropic
  credential, pin it to a full commit SHA
  (`uses: thomas-sodon/maestro-review@<commit-sha>`) so a later change to
  `main` cannot run with your secret unreviewed.
- **What leaves your runner.** The PR title, description and diff (and any file
  the model reads) are sent to Anthropic for the review. Pull requests from
  forks do not receive repository secrets, so the review step cannot run there.

## Command line

Requires Python 3.11+ and an authenticated [GitHub CLI](https://cli.github.com/)
(`gh auth login`, or `GH_TOKEN` in the environment).

```bash
pip install "git+https://github.com/thomas-sodon/maestro-review.git"
maestro-review --repo owner/repo --pr 42 --dry-run   # print findings, post nothing
maestro-review --repo owner/repo --pr 42             # review and post
```

Options, exit codes, the reply-coverage rules (markers and verdict phrases)
and configuration via `maestro-review.toml` or environment variables are
documented in [docs/REFERENCE.md](docs/REFERENCE.md).

## Development

```bash
git clone https://github.com/thomas-sodon/maestro-review.git
cd maestro-review
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/
pytest
```

Tests mock the Claude SDK and the `gh` CLI; they make no network calls.

## License

[MIT](LICENSE)
