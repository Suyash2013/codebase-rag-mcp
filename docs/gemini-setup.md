# Gemini Workflow Setup

The Gemini-powered triage and review workflows use the direct Gemini API (no GCP project or Workload Identity Federation required).

## Required Secret

| Secret | Description |
|--------|-------------|
| `GEMINI_API_KEY` | Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) |

This secret is already configured in the repository. If you're forking, add it via **Settings → Secrets and variables → Actions → New repository secret**.

## Required Variables

Set the following via the GitHub CLI (run once per repository):

```bash
gh variable set GOOGLE_GENAI_USE_VERTEXAI --body "false"
gh variable set GOOGLE_GENAI_USE_GCA --body "false"
gh variable set GEMINI_MODEL --body "gemini-3.1-pro-preview"
gh variable set GEMINI_CLI_VERSION --body "latest"
gh variable set GEMINI_DEBUG --body "false"
```

Or set them via **Settings → Secrets and variables → Actions → Variables**.

## What These Enable

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `gemini-triage.yml` | `issues.opened` | Auto-labels and summarises new issues |
| `gemini-scheduled-triage.yml` | Hourly cron | Picks up any unlabeled issues |
| `gemini-review.yml` | PR opened/updated | AI code review on pull requests |
| `gemini-dispatch.yml` | `@gemini` mention in issues/PRs | On-demand Gemini assistance |

## Notes

- `GOOGLE_GENAI_USE_VERTEXAI=false` and `GOOGLE_GENAI_USE_GCA=false` tell the Gemini CLI to authenticate via `GEMINI_API_KEY` instead of GCP service account credentials.
- No `APP_ID` or `APP_PRIVATE_KEY` is required — workflows fall back to `GITHUB_TOKEN` for repo operations.
- The `gemini-3.1-pro-preview` model is used by default. To change it, update the `GEMINI_MODEL` variable.
