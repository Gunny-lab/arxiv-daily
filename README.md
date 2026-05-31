# Daily arXiv Paper Screening Bot

Python MVP that fetches recent `hep-ph` and `hep-th` papers from the arXiv API, scores them against research-interest keywords, and writes Markdown candidate reports to `reports/`.

This version intentionally stays simple: local execution plus GitHub Actions automation, with reports committed back to the repository. By default, it does not call an LLM. It screens papers by title and abstract, then gives you candidate papers to read separately.

## 1. Installation

```bash
cd arxiv-daily
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd arxiv-daily
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Optional LLM Summary Settings

LLM summaries are disabled by default to avoid API cost. You do not need `OPENAI_API_KEY` for the default screening report.

If you later want automatic abstract summaries, set:

macOS/Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:ENABLE_LLM_SUMMARIES="1"
```

Optional controls:

```bash
export OPENAI_MODEL="gpt-4o-mini"
export MAX_HIGH_SUMMARIES="3"
export MAX_MEDIUM_SUMMARIES="0"
```

If `OPENAI_MODEL` is not set, the bot uses `gpt-4o-mini` by default.

## 3. Local Run

```bash
python arxiv_daily.py
```

The bot creates:

- `reports/arxiv_YYYY-MM-DD.md`
- `reports/latest.md`

## 4. GitHub Actions Secret

The default workflow can run without an OpenAI key because LLM summaries are disabled. If you enable LLM summaries in the workflow, add the key in your GitHub repository:

1. Go to **Settings > Secrets and variables > Actions**.
2. Click **New repository secret**.
3. Name it `OPENAI_API_KEY`.
4. Paste your OpenAI API key.

The workflow reads it as `secrets.OPENAI_API_KEY`.

## 5. Where Reports Appear

After local execution or GitHub Actions execution, reports are available in:

- `reports/arxiv_YYYY-MM-DD.md`
- `reports/latest.md`

GitHub Actions commits and pushes updated Markdown files back into the repository.

## 6. Edit Relevance Keywords

Edit `KEYWORD_GROUPS` in `arxiv_daily.py`.

Keywords are grouped as:

- `very_high`
- `high`
- `medium`

Title matches are weighted more heavily than abstract matches. Adjust thresholds near `HIGH_RELEVANCE_THRESHOLD` and `MEDIUM_RELEVANCE_THRESHOLD` if the report is too broad or too narrow.

For model-independent physics, edit or expand keywords such as:

- `model independent`
- `model-independent`
- `effective field theory`
- `operator basis`
- `basis independent`
- `general formalism`
- `sum rule`
- `null test`
- `observable`
- `kinematics`
- `symmetry`

## 7. Limitations

- The bot does not read full PDFs. Screening is based on arXiv titles and abstracts.
- LLM summaries are disabled by default. If enabled, summaries are based only on abstracts, not full papers.
- arXiv API `submittedDate` ordering may not exactly match the arXiv daily announcement page.
- Relevance is keyword-based, so related papers can be missed.

## Tests

```bash
pytest
```

## TODO

- Embedding-based recommendation
- PDF full-text parsing
- Email sending
- Notion integration
- Slack/Discord integration
- Web dashboard
