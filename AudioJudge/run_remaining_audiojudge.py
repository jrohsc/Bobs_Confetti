"""
Run AudioJudge on remaining song pairs for ablation study.

This script processes:
1. APT (phoneme attack variant)
2. Espresso (phoneme attack variant)
3. Christmas_Phoneme (Jingle Bell variants)

Run from the AudioJudge directory:
    cd /Users/jroh/Desktop/Research/Bobs_confetti/AudioJudge
    python run_remaining_audiojudge.py
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_audiojudge import judge_pair_and_save, batch_convert_mp3_tree

# Pairs to process: (original_wav, generated_wav, output_base_name)
SONG_PAIRS = [
    # APT
    (
        "audio_wav/APT/original/APT.wav",
        "audio_wav/APT/generated/APT (female punk rock)_1.wav",
        "audiojudge_results/APT/APT__APT_phoneme"
    ),
    # Espresso
    (
        "audio_wav/Espresso/original/Espresso_original.wav",
        "audio_wav/Espresso/generated/Espresso_generated.wav",
        "audiojudge_results/Espresso/Espresso__Espresso_phoneme"
    ),
    # Jingle Bells variants
    (
        "audio_wav/Christmas_Phoneme/original/Jingle Bells.wav",
        "audio_wav/Christmas_Phoneme/generated/Jingle Shells (SUNO, christmas style)_1.wav",
        "audiojudge_results/Christmas_Phoneme/Jingle_Bells__Jingle_Shells_christmas_1"
    ),
    (
        "audio_wav/Christmas_Phoneme/original/Jingle Bells.wav",
        "audio_wav/Christmas_Phoneme/generated/Giggle Shell (christmas style) 1.wav",
        "audiojudge_results/Christmas_Phoneme/Jingle_Bells__Giggle_Shell_1"
    ),
    # Jingle Bell Rock variants
    (
        "audio_wav/Christmas_Phoneme/original/Jingle Bell Rock.wav",
        "audio_wav/Christmas_Phoneme/generated/Giggle Shell sock (christmas style) 7.wav",
        "audiojudge_results/Christmas_Phoneme/Jingle_Bell_Rock__Giggle_Shell_sock_1"
    ),
]


def main():
    print("=" * 60)
    print("Running AudioJudge on Remaining Song Pairs")
    print("=" * 60)

    processed = 0
    skipped = 0
    errors = []

    for orig_path, gen_path, output_base in SONG_PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(orig_path)} vs {os.path.basename(gen_path)}")
        print(f"{'='*60}")

        # Check if files exist
        if not os.path.exists(orig_path):
            print(f"  ⚠️ Original not found: {orig_path}")
            errors.append((orig_path, "Original file not found"))
            continue

        if not os.path.exists(gen_path):
            print(f"  ⚠️ Generated not found: {gen_path}")
            errors.append((gen_path, "Generated file not found"))
            continue

        # Check if already processed
        if os.path.exists(output_base + ".json"):
            print(f"  ⏭️ Already processed, skipping")
            skipped += 1
            continue

        # Create output directory
        os.makedirs(os.path.dirname(output_base), exist_ok=True)

        try:
            judge_pair_and_save(orig_path, gen_path, output_base)
            processed += 1
            print(f"  ✅ Successfully processed")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors.append((orig_path, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for path, err in errors:
            print(f"  - {path}: {err}")

    print("\nNext steps:")
    print("1. Run: python ../CMU_phoneme/ablation_study.py")
    print("2. Check audiojudge_results/ for new results")


if __name__ == "__main__":
    main()
