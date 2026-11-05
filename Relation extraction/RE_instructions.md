# POLYIE Relation Extraction — Demo Instructions

This README provides brief instructions for running four relation extraction (RE) demonstrations: **distance-based proximity (V3)**, **dependency parsing**, **PURE-SUM**, and **GPT**. The examples assume the corresponding Python demo scripts have been added to this repository.

> **Note:** The script names, command-line flags and dependency lists below are illustrative and have not been verified against the actual demo files. Check each script's  requirements before running.

## Prerequisites

Install Python 3.9+ and download or clone this repository. From a terminal in the repository directory, optionally create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
```

The demonstrations assume these data files are available:

| File | Purpose |
| --- | --- |
| `test_split.txt` | POLYIE held-out test dataset |
| `test_predictions.json` | Predicted BIO entity labels from the NER model |
| `best_model.pth` | Trained PURE-SUM checkpoint (PURE-SUM only) |

## 1. Distance-based proximity

Uses token proximity and heuristic rules to associate chemical names, property names and property values.

Install suggested dependencies:

```bash
pip install numpy pandas
```

Example command (on one line):

```bash
python demo_v3_proximity_re.py --test test_split.txt --predictions test_predictions.json --output-dir v3_demo_results
```

## 2. Dependency parsing

Uses grammatical dependency relationships to help link chemical names to properties and values.

Install suggested dependencies and the spaCy English model:

```bash
pip install "spacy>=3.7,<4" numpy pandas
python -m spacy download en_core_web_sm
```

Example command:

```bash
python demo_dependency_re.py --test test_split.txt --predictions test_predictions.json --output-dir dependency_results
```

## 3. PURE-SUM

Uses a supervised transformer-based RE model. A **compatible trained checkpoint** is required; installing dependencies alone is not sufficient.

Install suggested dependencies:

```bash
pip install torch transformers tqdm numpy pandas
```

Example command:

```bash
python demo_pure_sum_original_re.py --test test_split.txt --predictions test_predictions.json --checkpoint path/to/run_1/best_model.pth --output-dir pure_sum_demo_results
```

Replace the checkpoint path with the location of your trained PURE-SUM weights.

## Comparing outputs

Use the **same held-out test split and NER predictions** across methods for a meaningful comparison. Inspect the output files produced by each implementation for extracted relationships and, where supported, precision, recall and F1. Exact output filenames and supported options depend on the scripts.

These three methods do **not** require an OpenAI API key. The proximity and dependency approaches are rule-based; PURE-SUM performs inference with a previously trained model.

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
