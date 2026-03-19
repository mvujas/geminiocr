"""Example: Using geminiocr for receipt/bill data extraction."""

import json
import os
from geminiocr import Settings, OCRSession

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load instruction and schema from files
with open(os.path.join(SCRIPT_DIR, "instruction.txt")) as f:
    system_instruction = f.read()

with open(os.path.join(SCRIPT_DIR, "schema.json")) as f:
    response_schema = json.load(f)

session = OCRSession(Settings(
    system_instruction=system_instruction,
    response_schema=response_schema,
))

# --- Single receipt ---

result = session.process_group("receipt_1", [os.path.join(SCRIPT_DIR, "images", "receipt.png")])
print(json.dumps(result, indent=2))

# --- Batch processing (uses cache automatically) ---
#
# receipts = {
#     "receipt_1": ["receipt_1.jpg"],
#     "receipt_2": ["receipt_2.jpg"],
# }
# results = session.process_batch(receipts)
#
# for receipt_id, result in results.items():
#     if isinstance(result, Exception):
#         print(f"FAILED {receipt_id}: {result}")
#     else:
#         print(f"{receipt_id}: ${result.get('total_usd_cost', '?')}")
