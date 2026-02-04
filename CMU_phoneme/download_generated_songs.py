"""
Download Generated Songs from SUNO API Results

This script downloads audio files from the JSON results produced by suno_generator.py
and organizes them for AudioJudge evaluation.

Usage:
    python download_generated_songs.py --input generated_songs --output ../AudioJudge/audio_mp3
"""

import json
import argparse
import requests
from pathlib import Path
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Download a file from URL to output path."""
    try:
        logger.info(f"Downloading: {output_path.name}")
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"  Saved: {output_path}")
        return True

    except requests.RequestException as e:
        logger.error(f"  Failed to download {url}: {e}")
        return False


def process_results_file(json_path: Path, output_base: Path) -> int:
    """Process a single results JSON file and download audio."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {json_path}: {e}")
        return 0

    song_key = data.get("song_key", "unknown")
    song_name = data.get("song_name", "Unknown")
    genre = data.get("genre", "Other")
    sample_num = data.get("sample_num", 1)
    api_response = data.get("api_response", [])

    if not api_response:
        logger.warning(f"No API response in {json_path}")
        return 0

    # Create output directory: output_base/genre/song_key/generated/
    output_dir = output_base / genre.lower() / song_key / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for i, clip in enumerate(api_response):
        audio_url = clip.get("audio_url")
        clip_id = clip.get("id", f"clip{i}")

        if not audio_url:
            logger.warning(f"  No audio_url for clip {clip_id}")
            continue

        # Filename: song_key_sample{N}_clip{M}.mp3
        filename = f"{song_key}_sample{sample_num}_clip{i}.mp3"
        output_path = output_dir / filename

        if output_path.exists():
            logger.info(f"  Skipping (exists): {filename}")
            continue

        if download_file(audio_url, output_path):
            downloaded += 1
            time.sleep(0.5)  # Brief delay between downloads

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download generated songs from SUNO API results"
    )
    parser.add_argument(
        "--input", "-i",
        default="generated_songs",
        help="Input directory containing JSON result files"
    )
    parser.add_argument(
        "--output", "-o",
        default="../AudioJudge/audio_mp3",
        help="Output directory for organized audio files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Find all JSON files (excluding summaries)
    json_files = [
        f for f in input_dir.glob("*.json")
        if "summary" not in f.name.lower()
    ]

    if not json_files:
        logger.warning(f"No JSON result files found in {input_dir}")
        return

    logger.info(f"Found {len(json_files)} result files")
    logger.info(f"Output directory: {output_dir}")

    if args.dry_run:
        logger.info("\n[DRY RUN] Files to process:")
        for jf in json_files:
            logger.info(f"  - {jf.name}")
        return

    # Process each file
    output_dir.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0

    for json_file in sorted(json_files):
        logger.info(f"\nProcessing: {json_file.name}")
        downloaded = process_results_file(json_file, output_dir)
        total_downloaded += downloaded

    logger.info(f"\n{'='*50}")
    logger.info(f"Download complete!")
    logger.info(f"Total files downloaded: {total_downloaded}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
