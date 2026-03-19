# Receipt OCR Example

Extracts item costs, cafe name, and time from a receipt image.

## Run with Python

```bash
python receipt_exctactor.py
```

## Run with CLI

```bash
python -m geminiocr images/ \
    --instruction instruction.txt \
    --schema schema.json \
    -o output.json
```

Both require `GEMINI_API_KEY` set in `.env` or environment. See the root [README](../../README.md) for setup.
