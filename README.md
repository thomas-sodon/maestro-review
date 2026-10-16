# maestro-review

An AI code reviewer for GitHub pull requests, built on the
[Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/).

`maestro-review` reads a PR's diff, asks Claude for a structured review, and
posts the findings back to the PR as inline review comments. It is designed to
run as a CI check: the command exits non-zero whenever it posts a new comment,
so a PR stays blocked until the findings are addressed.

It was extracted from [Maestro](https://github.com/thomas-sodon/maestro-agent),
an autonomous engineering agent, where it reviews every pull request the agent
opens. It has no dependency on Maestro and works on any GitHub repository.

## What it does

1. Fetches the PR diff and metadata through the `gh` CLI, truncating the diff to
   a configurable size and file count.
2. Sends the diff to Claude (read-only `Read`/`Glob`/`Grep` tools, run from the
   current directory, so checking out the PR first gives the model surrounding
   context). Findings come back with a severity: `bug`, `suggestion` or `nit`.
3. Before posting, it de-duplicates against the PR's state:
   - findings already tracked by an open GitHub issue are dropped;
   - findings matching an existing review thread are skipped, or the thread is
     reopened if it was resolved without an issue link but the problem is still there;
   - findings on a thread where someone replied *"Could not prove this defect"*
     are escalated to a GitHub issue (labels `ai-review` and the severity)
     instead of being re-posted.
4. Posts one review: an inline comment per finding, plus an *Additional findings*
   list in the review body for findings that cannot be anchored to a diff line.
   Each of those overflow findings carries a stable `of-<hash>` handle.

There is also a read-only **reply-coverage audit** (`--require-comment-replies`)
that makes no Claude call: it fails if any of the reviewer's own findings has
not been answered yet.

## Install

Requires Python 3.11+ and an authenticated [GitHub CLI](https://cli.github.com/)
(`gh auth login`, or `GH_TOKEN` in the environment).

```bash
pip install "git+https://github.com/thomas-sodon/maestro-review.git"
```

For development:

```bash
git clone https://github.com/thomas-sodon/maestro-review.git
cd maestro-review
pip install -e ".[dev]"
```

### Credentials

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` **or** `CLAUDE_CODE_OAUTH_TOKEN` | the review itself (not needed for `--require-comment-replies`) |
| `GH_TOKEN` / `GITHUB_TOKEN` (or a `gh auth login` session) | reading the PR, posting reviews, opening escalation issues |

The GitHub token needs `pull-requests: write` to post reviews and
`issues: write` for the disputed-finding escalation.

## CLI usage

```bash
maestro-review --repo owner/repo --pr 42                     # review and post
maestro-review --repo owner/repo --pr 42 --dry-run           # print findings, post nothing
maestro-review --repo owner/repo --pr 42 --max-diff-chars 100000 --max-files 75
maestro-review --repo owner/repo --pr 42 --require-comment-replies   # reply-coverage audit only
```

| Option | Default | Meaning |
|---|---|---|
| `--repo` | (required) | GitHub repository, `owner/repo` |
| `--pr` | (required) | Pull request number |
| `--max-diff-chars` | `80000` | Diff characters sent to the model before truncation |
| `--max-files` | `50` | Files included from the diff |
| `--gh-timeout` | `30` | Timeout in seconds for each `gh` call |
| `--dry-run` | off | Print the summary and findings to stdout instead of posting |
| `--require-comment-replies` | off | Check-only mode: skip the review, audit reply coverage |

`--require-comment-replies` takes precedence: when it is set, `--dry-run` has
no effect and no diff review is run.

## Exit codes

| Mode | Exit `0` | Exit `1` |
|---|---|---|
| Review (default) | Nothing new was posted: no findings, an empty diff, or every finding was already tracked by an issue, escalated to an issue, or matched an existing thread that is still open (or was resolved with an issue link) | At least one finding (any severity) was newly posted or a resolved thread was reopened; **or** the review failed (API/SDK error); **or** posting the review failed (findings are then printed to the log) |
| `--dry-run` | No finding would be posted or reopened after the same issue/thread filtering a real run applies | At least one finding would be posted or reopened (if the filtering itself errors, any raw finding counts); **or** the review failed |
| `--require-comment-replies` | Every reviewer finding has been answered | At least one unanswered finding; **or** the audit could not read the PR (a `gh`/GraphQL outage fails closed) |

Invalid or missing arguments exit `2` (standard `argparse` behaviour).

A finding counts as **answered** when:

- *inline*: its thread is resolved, outdated, or has any reply;
- *overflow*: a PR comment containing the `<!-- maestro:overflow-ack -->` marker
  lists its `of-<hash>` handle on a line together with one of these verdict
  phrases: `Could not prove this defect`, `Could not prove this bug`,
  `Fixed automatically`, `Fixed locally with implementation-plan context.`,
  `Tracked in #`, `Declined with implementation-plan context.`,
  `Answered with implementation-plan context.`

## Configuration

Command-line options always win. Defaults for `max_diff_chars`, `max_files`,
`gh_timeout` and `require_comment_replies` can also come from a config file
and/or environment variables.

**Config file.** The first of these files that exists and has a non-empty
`[review]` section is used:

1. the path in `MAESTRO_REVIEW_CONFIG`;
2. `./maestro-review.toml`;
3. `./maestro.toml` (compatible with a Maestro project config).

```toml
[review]
max_diff_chars = 80000
max_files = 50
gh_timeout = 30
require_comment_replies = 0   # 1 = run in reply-coverage audit mode by default
```

A file that fails to parse is ignored; a value that is not an integer falls
back to its default.

**Environment variables** override the config file:
`MAESTRO_REVIEW_MAX_DIFF_CHARS`, `MAESTRO_REVIEW_MAX_FILES`,
`MAESTRO_REVIEW_GH_TIMEOUT`, `MAESTRO_REVIEW_REQUIRE_COMMENT_REPLIES`
(integers; blank or non-integer values are ignored).

The model and effort are fixed in `maestro_review/sdk.py` (`REVIEW_MODEL`,
`REVIEW_EFFORT`).

## GitHub Action

The repository ships a composite action. Store an `ANTHROPIC_API_KEY` (or a
`CLAUDE_CODE_OAUTH_TOKEN`) as a repository secret, then:

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

| Input | Default | Meaning |
|---|---|---|
| `anthropic-api-key` | — | Anthropic API key |
| `claude-code-oauth-token` | — | Claude Code OAuth token (alternative to the key) |
| `github-token` | `github.token` | Token used by `gh` |
| `repo` | current repository | `owner/repo` |
| `pr-number` | triggering PR | Pull request number |
| `mode` | `review` | `review` or `replies` (`--require-comment-replies`; needs no Claude credential) |
| `dry-run` | `false` | `true` prints findings instead of posting |
| `max-diff-chars`, `max-files`, `gh-timeout` | tool defaults | Passed through to the CLI |
| `python-version` | `3.14` | Python used to run the reviewer |

The step's exit status is the CLI's exit code, so the job fails exactly when
the table under [Exit codes](#exit-codes) says it does. Pull requests from
forks do not receive repository secrets, so the review step cannot run there.

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/
pytest
```

Tests mock the Claude SDK and the `gh` CLI; they make no network calls.

## License

[MIT](LICENSE)
