"""Example: Using geminiocr for receipt/bill data extraction."""

import json
import os
from geminiocr import Settings, OCRSession

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

session = OCRSession(Settings(
    system_instruction=(
        "Extract item costs and metadata from the provided receipt/bill images.\n"
        "Return ONLY valid JSON. Match item names to the schema fields.\n"
        "Use the per-unit cost for individual items, not the line total.\n"
        "If an item is not found on the receipt, return null for that field.\n"
        'For time_of_the_day, use the format "HH:MM" as printed on the receipt.'
    ),
    response_schema={
        "type": "OBJECT",
        "properties": {
            "cafe_name": {"type": "STRING"},
            "address": {"type": "STRING"},
            "blueberry_muffin_cost": {"type": "NUMBER"},
            "avocado_toast_cost": {"type": "NUMBER"},
            "lg_oat_latte_cost": {"type": "NUMBER"},
            "time_of_the_day": {"type": "STRING"},
            "total_usd_cost": {"type": "NUMBER"},
        },
        "required": ["total_usd_cost"],
    },
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
