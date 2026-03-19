# geminiocr

General-purpose batch OCR via Google Gemini API with context caching.

Send images to Gemini with your own system instruction and response schema. For a single request, it calls the API directly. For batches (2+ groups), it automatically creates a context cache so the system instruction is sent once and reused across all requests, saving up to 90% on input token costs.

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/mvujas/geminiocr.git

# Or clone and install in editable mode (for development)
git clone git@github.com:mvujas/geminiocr.git
cd geminiocr
pip install -e .
```

This installs the `geminiocr` package and its dependencies (`google-genai`, `python-dotenv`, `tqdm`).

## Setup

### 1. Get an API key

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Configure your environment

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your-actual-api-key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_CACHE_TTL=3600s
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | - | Your Gemini API key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Model to use |
| `GEMINI_CACHE_TTL` | No | `3600s` | Cache time-to-live for batch mode |

You can also pass these directly in code (see below).

## Usage

### Python library

#### Single image/group (no cache overhead)

```python
from geminiocr import Settings, OCRSession

settings = Settings(
    system_instruction="Extract item costs from this receipt. Return JSON.",
    response_schema={
        "type": "OBJECT",
        "properties": {
            "total_usd_cost": {"type": "NUMBER"},
            "time_of_the_day": {"type": "STRING"},
        },
        "required": ["total_usd_cost"],
    },
)
session = OCRSession(settings)

result = session.process_group("receipt_1", ["receipt.png"])
print(result)
# {"total_usd_cost": 23.50, "time_of_the_day": "11:42 AM"}
```

#### Batch processing (auto-caches, concurrent, progress bar)

```python
groups = {
    "receipt_1": ["images/receipt_1.jpg"],
    "receipt_2": ["images/receipt_2.jpg", "images/receipt_2_tip.jpg"],
    "receipt_3": ["images/receipt_3.jpg"],
    # ... hundreds more
}

results = session.process_batch(groups)

for group_id, result in results.items():
    if isinstance(result, Exception):
        print(f"FAILED {group_id}: {result}")
    else:
        print(f"{group_id}: ${result.get('total_usd_cost', '?')}")
```

When `process_batch` receives 2+ groups, it automatically:
- Creates a context cache with your system instruction (one API call)
- Processes groups concurrently (default 5 parallel requests)
- Shows a `tqdm` progress bar
- Returns partial results if interrupted (Ctrl+C, errors)
- Per-group failures are captured as `Exception` values, other groups continue

#### Overriding settings in code

```python
settings = Settings(
    api_key="your-key-here",          # overrides env var
    model="gemini-2.5-pro",           # overrides GEMINI_MODEL
    system_instruction="...",
    response_schema={...},
    concurrency=10,                   # max parallel requests
    max_retries=3,                    # retries per request
    cache_ttl="7200s",                # 2 hour cache
)
```

### CLI

```bash
# Basic usage - subdirectory layout (images/GroupA/*.jpg, images/GroupB/*.jpg)
python -m geminiocr ./images/ --instruction "Extract all text." -o results.json

# With a schema file and custom model
python -m geminiocr ./images/ \
    --instruction instructions.txt \
    --schema schema.json \
    --model gemini-2.5-pro \
    --concurrency 10 \
    -v

# Instruction can be inline text or a path to a .txt file
python -m geminiocr ./images/ --instruction "Extract item costs from receipts."
```

#### CLI arguments

| Argument | Description |
|---|---|
| `image_dir` | Directory with image groups (subdirs or flat `Name_1.jpg` layout) |
| `--instruction` | System instruction text or path to a `.txt` file |
| `--schema` | Path to a JSON file with the response schema |
| `--model` | Gemini model name (overrides env var) |
| `--output, -o` | Write results to this JSON file (default: stdout) |
| `--concurrency` | Max parallel API requests (default: 5) |
| `--verbose, -v` | Enable debug logging |

#### Image directory layouts

**Subdirectory per group:**
```
images/
  receipt_1/
    page1.jpg
    page2.jpg
  receipt_2/
    photo.png
```

**Flat with prefix:**
```
images/
  receipt_1_1.jpg
  receipt_1_2.jpg
  receipt_2_1.jpg
```

## How caching works

| Scenario | What happens | Cost impact |
|---|---|---|
| 1 group | Direct API call with inline system instruction | Normal cost |
| 2+ groups | Creates one cache, reuses across all requests | Up to 90% savings on input tokens |

The cache is created lazily on the first batch request and auto-refreshes before TTL expiry, so long-running batches don't fail mid-run.

## Examples

See the [examples/](examples/) directory for complete working examples.
