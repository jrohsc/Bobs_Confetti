"""
Experiment 1: Random Lyrics Baseline Generator

Purpose: Generate nonsensical lyrics with NO phonetic similarity to original songs.
This proves that phonetic structure matters, not just "any input".

Expected Results:
- Random lyrics → Melody ~0.30-0.50, Rhythm ~0.30-0.50, CLAP ~0.30-0.50
- APT lyrics → Melody ~0.85-0.95, Rhythm ~0.85-0.95, CLAP ~0.65-0.85

Usage:
    python random_lyrics_generator.py --output random_lyrics/
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

# Word pools for generating random nonsensical lyrics
# Using words that are phonetically DIFFERENT from common song words
NOUNS = [
    "umbrella", "telescope", "avocado", "dinosaur", "calculator",
    "refrigerator", "helicopter", "tambourine", "xylophone", "submarine",
    "kangaroo", "cucumber", "trampoline", "volcano", "accordion",
    "mosquito", "broccoli", "flamingo", "platypus", "iguana",
    "artichoke", "eucalyptus", "calendar", "parachute", "microscope",
    "butterfly", "chandelier", "aquarium", "motorcycle", "watermelon"
]

VERBS = [
    "juggling", "evaporating", "oscillating", "perpendicular", "fabricating",
    "manufacturing", "calculating", "hibernating", "illuminating", "decorating",
    "navigating", "accelerating", "contemplating", "eliminating", "generating",
    "excavating", "manipulating", "coordinating", "demonstrating", "investigating"
]

ADJECTIVES = [
    "cylindrical", "hexagonal", "luminescent", "perpendicular", "triangular",
    "rectangular", "spherical", "elliptical", "cubical", "octagonal",
    "invisible", "microscopic", "fluorescent", "translucent", "iridescent",
    "magnificent", "ridiculous", "spontaneous", "simultaneous", "tremendous"
]

ADVERBS = [
    "perpendicularly", "simultaneously", "magnificently", "spontaneously",
    "horizontally", "vertically", "diagonally", "intermittently", "occasionally",
    "subsequently", "consequently", "alternatively", "approximately", "immediately"
]

PREPOSITIONS = [
    "underneath", "throughout", "beside", "between", "around",
    "beyond", "within", "against", "toward", "during"
]


def generate_random_line(min_words: int = 5, max_words: int = 10) -> str:
    """Generate a single line of random nonsensical lyrics."""
    num_words = random.randint(min_words, max_words)
    words = []

    for i in range(num_words):
        # Mix different word types randomly
        word_type = random.choice(['noun', 'verb', 'adj', 'adv', 'prep'])

        if word_type == 'noun':
            words.append(random.choice(NOUNS))
        elif word_type == 'verb':
            words.append(random.choice(VERBS))
        elif word_type == 'adj':
            words.append(random.choice(ADJECTIVES))
        elif word_type == 'adv':
            words.append(random.choice(ADVERBS))
        else:
            words.append(random.choice(PREPOSITIONS))

    line = " ".join(words)
    # Capitalize first letter
    return line[0].upper() + line[1:]


def generate_random_lyrics(
    num_lines: int = 16,
    num_verses: int = 4,
    seed: int = None
) -> str:
    """
    Generate complete random lyrics with no semantic meaning.

    Args:
        num_lines: Total number of lines
        num_verses: Number of verse groupings
        seed: Random seed for reproducibility

    Returns:
        String of random lyrics
    """
    if seed is not None:
        random.seed(seed)

    lines_per_verse = num_lines // num_verses
    lyrics_lines = []

    for verse in range(num_verses):
        for line in range(lines_per_verse):
            lyrics_lines.append(generate_random_line())

        # Add empty line between verses (except after last)
        if verse < num_verses - 1:
            lyrics_lines.append("")

    return "\n".join(lyrics_lines)


def generate_random_baseline_for_song(
    song_name: str,
    orig_lyrics_path: Path,
    output_dir: Path,
    num_variants: int = 3
) -> List[Tuple[str, Path]]:
    """
    Generate random baseline lyrics for a specific song.
    Matches the line count and structure of original lyrics.

    Args:
        song_name: Name of the song (e.g., "lose_yourself")
        orig_lyrics_path: Path to original lyrics file
        output_dir: Directory to save random lyrics
        num_variants: Number of random variants to generate

    Returns:
        List of (lyrics, path) tuples
    """
    # Read original to match structure
    with open(orig_lyrics_path, 'r') as f:
        orig_lines = f.readlines()

    num_lines = len([l for l in orig_lines if l.strip()])

    results = []

    for i in range(num_variants):
        # Use different seed for each variant
        seed = hash(f"{song_name}_{i}") % (2**32)
        random_lyrics = generate_random_lyrics(
            num_lines=num_lines,
            num_verses=4,
            seed=seed
        )

        # Save to file
        output_path = output_dir / f"{song_name}_random_v{i+1}.txt"
        with open(output_path, 'w') as f:
            f.write(random_lyrics)

        results.append((random_lyrics, output_path))
        print(f"  Generated: {output_path.name}")

    return results


# Song configurations matching the main experiment
SONG_CONFIGS = {
    "lose_yourself": {
        "name": "Lose Yourself",
        "artist": "Eminem",
        "genre": "Rap",
        "style": "intense rap, motivational, rhythmic flow",
        "title_template": "Random Baseline {variant} (Lose Yourself Control)"
    },
    "dna": {
        "name": "DNA",
        "artist": "Kendrick Lamar",
        "genre": "Rap",
        "style": "rap, gangsta, trap, aggressive, rhythmic",
        "title_template": "Random Baseline {variant} (DNA Control)"
    },
    "humble": {
        "name": "HUMBLE",
        "artist": "Kendrick Lamar",
        "genre": "Rap",
        "style": "trap gangsta, bold, sparse, direct",
        "title_template": "Random Baseline {variant} (HUMBLE Control)"
    },
    "espresso": {
        "name": "Espresso",
        "artist": "Sabrina Carpenter",
        "genre": "Pop",
        "style": "pop, upbeat, catchy, female vocal",
        "title_template": "Random Baseline {variant} (Espresso Control)"
    },
    "let_it_be": {
        "name": "Let It Be",
        "artist": "The Beatles",
        "genre": "Pop",
        "style": "rock ballad, piano, inspirational, male vocal",
        "title_template": "Random Baseline {variant} (Let It Be Control)"
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate random baseline lyrics for Experiment 1"
    )
    parser.add_argument(
        "--output",
        default="random_lyrics",
        help="Output directory for random lyrics"
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=3,
        help="Number of random variants per song (default: 3)"
    )
    parser.add_argument(
        "--songs",
        nargs="+",
        default=None,
        help="Specific songs (default: all)"
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "baseline_experiments" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    songs = args.songs if args.songs else list(SONG_CONFIGS.keys())

    print("\n" + "="*60)
    print("EXPERIMENT 1: RANDOM LYRICS BASELINE GENERATOR")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Variants per song: {args.variants}")
    print(f"Songs: {', '.join(songs)}")
    print("="*60 + "\n")

    all_results = {}

    for song_key in songs:
        if song_key not in SONG_CONFIGS:
            print(f"Warning: Unknown song '{song_key}', skipping")
            continue

        config = SONG_CONFIGS[song_key]
        orig_path = base_dir / f"{song_key}_orig.txt"

        if not orig_path.exists():
            print(f"Warning: Original lyrics not found for {song_key}, skipping")
            continue

        print(f"\nGenerating random baselines for: {config['name']}")
        results = generate_random_baseline_for_song(
            song_key,
            orig_path,
            output_dir,
            num_variants=args.variants
        )

        all_results[song_key] = {
            "config": config,
            "variants": [str(path) for _, path in results]
        }

    # Save manifest
    manifest_path = output_dir / "random_lyrics_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total songs: {len(all_results)}")
    print(f"Total variants: {len(all_results) * args.variants}")
    print(f"Manifest saved: {manifest_path}")
    print("\nNext step: Run baseline_suno_generator.py to generate songs")


if __name__ == "__main__":
    main()
