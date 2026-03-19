import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from geminiocr import Settings, OCRSession

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR images via Google Gemini API"
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing image groups (subdirectories) or flat images",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="System instruction text, or path to a .txt file containing it",
    )
    parser.add_argument(
        "--schema",
        default=None,
        type=Path,
        help="Path to a JSON file defining the response schema",
    )
    parser.add_argument("--model", default=None, help="Gemini model name")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Max parallel API requests"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Build settings
    overrides: dict = {"concurrency": args.concurrency}
    if args.model:
        overrides["model"] = args.model

    if args.instruction:
        instruction_path = Path(args.instruction)
        if instruction_path.is_file():
            overrides["system_instruction"] = instruction_path.read_text().strip()
        else:
            overrides["system_instruction"] = args.instruction

    if args.schema:
        overrides["response_schema"] = json.loads(args.schema.read_text())

    settings = Settings(**overrides)
    session = OCRSession(settings)

    # Discover groups
    groups = discover_groups(args.image_dir)
    if not groups:
        logging.error("No images found in %s", args.image_dir)
        sys.exit(1)

    logging.info(
        "Found %d groups with %d total images",
        len(groups),
        sum(len(v) for v in groups.values()),
    )

    results = session.process_batch(groups)

    # Separate successes and errors
    output = {}
    errors = {}
    for gid, result in results.items():
        if isinstance(result, Exception):
            errors[gid] = str(result)
        else:
            output[gid] = result

    if args.output:
        args.output.write_text(json.dumps(output, indent=2))
        logging.info("Results written to %s", args.output)
    else:
        print(json.dumps(output, indent=2))

    logging.info(
        "Completed: %d/%d succeeded", len(output), len(output) + len(errors)
    )

    if errors:
        logging.warning(
            "Failed %d groups: %s", len(errors), json.dumps(errors, indent=2)
        )
        sys.exit(1)


def discover_groups(image_dir: Path) -> dict[str, list[Path]]:
    """Auto-discover image groups from directory structure.

    Supports two layouts:
      1. Subdirectory per group:  image_dir/GroupA/img1.jpg
      2. Flat with prefix:        image_dir/GroupA_1.jpg, GroupA_2.jpg
    """
    groups: dict[str, list[Path]] = defaultdict(list)

    subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
    if subdirs:
        for subdir in sorted(subdirs):
            images = sorted(
                f for f in subdir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
            )
            if images:
                groups[subdir.name] = images
    else:
        for f in sorted(image_dir.iterdir()):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                stem = f.stem
                parts = stem.rsplit("_", 1)
                group_id = (
                    parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
                )
                groups[group_id].append(f)

    return dict(groups)


if __name__ == "__main__":
    main()
