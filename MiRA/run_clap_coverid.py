#!/usr/bin/env python3
"""
Script to run CLAP and CoverID metrics on all songs in the audio directory.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Configuration
AUDIO_BASE = Path("/work/pi_ahoumansadr_umass_edu/jroh/MiRA/audio")
MIRA_BASE = Path("/work/pi_ahoumansadr_umass_edu/mira")
OUTPUT_DIR = Path("/work/pi_ahoumansadr_umass_edu/jroh/MiRA/results")
TEMP_DIR = Path("/work/pi_ahoumansadr_umass_edu/jroh/MiRA/temp_wav")

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 file to WAV using pydub (requires ffmpeg)."""
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio.export(str(wav_path), format="wav")
    return wav_path

def prepare_wav_files(song_dir):
    """Convert all MP3 files to WAV in a temp directory for a song."""
    song_name = song_dir.name
    temp_song_dir = TEMP_DIR / song_name
    temp_original = temp_song_dir / "original"
    temp_generated = temp_song_dir / "generated"

    # Create temp directories
    temp_original.mkdir(parents=True, exist_ok=True)
    temp_generated.mkdir(parents=True, exist_ok=True)

    # Process original files
    original_dir = song_dir / "original"
    if original_dir.exists():
        for f in original_dir.iterdir():
            if f.suffix.lower() == '.mp3':
                wav_name = f.stem + '.wav'
                wav_path = temp_original / wav_name
                if not wav_path.exists():
                    print(f"  Converting: {f.name} -> {wav_name}")
                    convert_mp3_to_wav(f, wav_path)
            elif f.suffix.lower() == '.wav':
                wav_path = temp_original / f.name
                if not wav_path.exists():
                    shutil.copy(f, wav_path)

    # Process generated files
    generated_dir = song_dir / "generated"
    if generated_dir.exists():
        for f in generated_dir.iterdir():
            if f.suffix.lower() == '.mp3':
                wav_name = f.stem + '.wav'
                wav_path = temp_generated / wav_name
                if not wav_path.exists():
                    print(f"  Converting: {f.name} -> {wav_name}")
                    convert_mp3_to_wav(f, wav_path)
            elif f.suffix.lower() == '.wav':
                wav_path = temp_generated / f.name
                if not wav_path.exists():
                    shutil.copy(f, wav_path)

    return temp_original, temp_generated

def run_mira_metric(original_dir, generated_dir, eval_name, metric, log_dir):
    """Run a specific mira metric."""
    # Add trailing slash - mira uses glob with folder + '*.wav'
    original_path = str(original_dir) + "/"
    generated_path = str(generated_dir) + "/"

    cmd = [
        "mira",
        original_path,
        generated_path,
        "--eval_name", eval_name,
        "-m", metric,
        "--log", str(log_dir)
    ]

    print(f"  Running {metric}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MIRA_BASE))

    if result.returncode != 0:
        print(f"  Error running {metric}:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return False

    return True

def main():
    print("=" * 60)
    print("MiRA Audio Evaluation - CLAP and CoverID")
    print("=" * 60)

    # Get all song directories
    songs = sorted([d for d in AUDIO_BASE.iterdir() if d.is_dir() and d.name != "misc"])

    print(f"\nFound {len(songs)} songs to process:")
    for s in songs:
        print(f"  - {s.name}")

    # Process each song
    all_results = []

    for i, song_dir in enumerate(songs, 1):
        song_name = song_dir.name
        # Create safe eval name (replace spaces and special chars)
        eval_name = song_name.replace(" ", "_").replace("'", "")

        print(f"\n[{i}/{len(songs)}] Processing: {song_name}")
        print("-" * 40)

        # Prepare WAV files
        print("  Preparing WAV files...")
        try:
            original_dir, generated_dir = prepare_wav_files(song_dir)
        except Exception as e:
            print(f"  Error preparing files: {e}")
            continue

        # Check if we have files to process
        orig_files = list(original_dir.glob("*.wav"))
        gen_files = list(generated_dir.glob("*.wav"))

        if not orig_files:
            print(f"  No original WAV files found, skipping...")
            continue
        if not gen_files:
            print(f"  No generated WAV files found, skipping...")
            continue

        print(f"  Original files: {len(orig_files)}, Generated files: {len(gen_files)}")

        # Run CLAP
        clap_success = run_mira_metric(original_dir, generated_dir, eval_name, "clap", OUTPUT_DIR)

        # Run CoverID
        coverid_success = run_mira_metric(original_dir, generated_dir, eval_name, "coverid", OUTPUT_DIR)

        all_results.append({
            "song": song_name,
            "clap": clap_success,
            "coverid": coverid_success
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in all_results:
        clap_status = "✓" if r["clap"] else "✗"
        coverid_status = "✓" if r["coverid"] else "✗"
        print(f"  {r['song']}: CLAP {clap_status}, CoverID {coverid_status}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nResult files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
