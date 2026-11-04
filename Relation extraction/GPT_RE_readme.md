# POLYIE: Three-shot GPT Relation Extraction Demo

This Python demo extracts **chemical–property–value** relationships from polymer-science text using the POLYIE annotation scheme and three fixed examples from the training set. It accepts pre-annotated entities (gold BIO labels by default), asks an OpenAI model to predict relations, and compares predictions with the held-out gold relations.

> **GitHub hosts the code; it does not execute the demo automatically.** Run the script locally or in a Python-enabled environment after cloning/downloading the repository.

## Requirements

- Python 3.9 or newer (recommended)
- An OpenAI API key and access to a supported model (live runs incur API charges)
- Files: `demo_GPT_RE_GitHub.py`, `train_split.txt`, and `test_split.txt` in the same folder. The data files must have the original POLYIE split and structure; the script verifies the three training examples using a SHA-256 hash.

## Installation

Clone the repository (replace the URL with your own), then install dependencies:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python -m pip install openai httpx
```

Set your API key in the environment **only for live runs**:

**Windows PowerShell**
```powershell
$env:OPENAI_API_KEY = "your-api-key"
```

**macOS / Linux**
```bash
export OPENAI_API_KEY="your-api-key"
```

Never commit an API key to GitHub.

## Run the demo

**1. Test run (no API request or charge):**

```bash
python demo_GPT_RE_GitHub.py
```

This checks the three demonstrations and saves the constructed prompt and metadata.

**2. Live relation extraction:**

```bash
python demo_GPT_RE_GitHub.py --execute --model YOUR_AVAILABLE_MODEL
```

Replace `YOUR_AVAILABLE_MODEL` with a model available to your OpenAI API account. The script's default is `gpt-6.1-sol`, which may not be available to every account.

**3. Optional: select another test segment:**

```bash
python demo_GPT_RE_GitHub.py --execute --model YOUR_AVAILABLE_MODEL --document 0 --segment 1
```

By default, the demo uses **gold-standard entities**. To use predicted NER labels instead, use the compatible `test_predictions.json` file and run with `--ner-source predicted --predictions test_predictions.json`.

## Output

Results are written as JSON to `gpt_re_results/`, for example `gpt_re_gold_doc0_seg0.json`. Live runs print predicted chemical–property–value relationships and exact-match **precision, recall and F1**, comparing the predictions against gold relations for the selected segment. A test run saves the prompt but does not produce predictions.

**Note:** A single-segment result is a demonstration, not a full-test-set performance estimate. The original three-shot demonstrations use training document/segment indices `(0, 2)`, `(1, 4)`, and `(50, 6)`.
