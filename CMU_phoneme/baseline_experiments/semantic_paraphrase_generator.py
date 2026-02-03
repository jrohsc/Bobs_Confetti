"""
Experiment 2: Semantic Paraphrase Baseline Generator

Purpose: Create meaning-preserving paraphrases that BREAK phonetic structure.
This proves that phonetic similarity (not semantic similarity) triggers memorization.

Example:
- Original: "His palms are sweaty, knees weak, arms are heavy"
- APT:      "His pants are sweaty, cheese weak, cars are heavy"  (phonetically similar)
- Paraphrase: "He's nervous and trembling, his limbs feel like lead weights" (semantically similar)

Expected Results:
- Semantic paraphrase → Melody ~0.30-0.50 (different tune despite same meaning)
- APT lyrics → Melody ~0.85-0.95 (same tune due to phonetic structure)

Usage:
    python semantic_paraphrase_generator.py --api-key $OPENAI_API_KEY
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Install with: pip install openai")


PARAPHRASE_SYSTEM_PROMPT = """You are a lyric paraphrasing expert. Your task is to rewrite song lyrics while:

1. PRESERVING the semantic meaning (what the lyrics are about)
2. BREAKING the phonetic structure completely (different sounds, syllables, rhymes)
3. Maintaining roughly the same number of lines
4. Using completely different vocabulary and word choices

The goal is to create lyrics that mean the SAME thing but SOUND completely different.

CRITICAL RULES:
- Do NOT preserve rhyme schemes
- Do NOT preserve syllable patterns
- Do NOT use words that sound similar to the original
- DO change sentence structure completely
- DO use synonyms and alternative phrasings
- DO maintain the emotional tone and meaning

Example:
Original: "His palms are sweaty, knees weak, arms are heavy"
BAD paraphrase (too phonetically similar): "His hands are sweaty, please seek, charms are ready"
GOOD paraphrase: "He's nervous and trembling, his limbs feel like lead weights"

Original: "There's vomit on his sweater already, mom's spaghetti"
BAD paraphrase: "There's vomit on his sweater ready, Tom's confetti"
GOOD paraphrase: "He threw up on his shirt beforehand, his mother's pasta dinner"
"""


PARAPHRASE_USER_TEMPLATE = """Rewrite these lyrics to preserve the MEANING but completely CHANGE the SOUNDS.

The resulting paraphrase should:
- Mean the same thing conceptually
- Sound COMPLETELY different (different phonemes, syllables, rhythm)
- Not rhyme in the same places
- Use different vocabulary

Original lyrics:
{lyrics}

Provide ONLY the paraphrased lyrics, no explanations."""


# Pre-written semantic paraphrases for key test songs
# These were carefully crafted to preserve meaning while breaking phonetic structure
MANUAL_PARAPHRASES = {
    "lose_yourself": """He's drenched in perspiration, limbs trembling, body weighted down
His shirt is stained from earlier, that homemade Italian dish
Anxiety grips him, though outwardly he appears composed
Ready to perform, yet his mind keeps erasing
The lines he practiced, the audience grows rowdy
His jaw drops open, nothing emerges
He's suffocating now
Everyone's mocking now
Running out of seconds, finished, concluded
Return to the present, watch as physics takes hold
Watch as the rapper, he failed, he's furious now
Yet he refuses to surrender so simply, he won't allow it
His entire existence depends on these challenges
He's skilled, he understands it, yet he's penniless, so burdened
He realizes returning to that trailer means
Returning to creating music, this entire career
He must seize this opportunity before it vanishes""",

    "dna": """I possess, I possess, I possess, I possess
Faithfulness, nobility within my genetic code
Fractional wealth, conflict and tranquility within my genetic code
I contain strength, toxicity, suffering and happiness within my genetic code
I have determination, yet goals course within my genetic code
I arrived this way
From birth like this, divine creation
I shift like this, execute like this
The messiah's fresh instrument
I skip deliberation, I reflect deeply
Then remove your, remove your head
This puts children to sleep
This is I possess, I possess, I possess, I possess
Authenticity, I eliminate because it's within my genetic code
I own fortunes, wealth accumulating within my genetic code
I contain shadow, wickedness decaying within my genetic code
I got released, I have a complicated heart within my genetic code
I simply triumph repeatedly, then triumph repeatedly like tennis matches I deliver""",

    "humble": """Remain modest
Remain modest
Remain modest
Remain modest
Remain modest
Remain modest
Remain modest
Remain modest
Nobody intercede when I'm settling matters
I'm not with foolishness, attempt me, I'll start
Ascending, opposing, you're going to see a genuine one
Perishing, opposing, I must reprimand someone
Head in reverse, behind on the seat rest
Steering with one arm, Bacardi in the other
I prepare like Compton, collect every dollar
I'm essentially the prototype for your favorite rapper
Remain modest
Remain modest
Remain modest
Remain modest""",

    "espresso": """Currently I'm having difficulties with rest
Since lately you've occupied my thoughts consistently
My obsession has gone past healthiness
Reaching a point where sustenance isn't appealing

Additionally I obtained confirmation from your companions
Unexpectedly I became their topic of discussion
The situation is quite unusual

Since I realize I have those characteristics
I understand I possess what captivates attention
And indeed I might cause dependency
You just need a sample of my essence
And you'll require my presence every sunrise""",

    "let_it_be": """During my difficult moments
A maternal figure appears before me
Speaking wisdom
Allow things to unfold naturally
And during my darkest hours
She positions herself directly ahead
Speaking wisdom
Allow things to unfold naturally
Allow it, allow it
Allow it, allow it
Speaking wisdom
Allow things to unfold naturally"""
}


class SemanticParaphraseGenerator:
    """Generates semantic paraphrases using GPT-4 or manual fallback."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key and HAS_OPENAI:
            self.client = OpenAI(api_key=self.api_key)

    def generate_paraphrase_gpt(self, lyrics: str) -> Optional[str]:
        """Generate paraphrase using GPT-4."""
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                    {"role": "user", "content": PARAPHRASE_USER_TEMPLATE.format(lyrics=lyrics)}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT-4 paraphrase failed: {e}")
            return None

    def get_paraphrase(self, song_key: str, original_lyrics: str, use_manual: bool = True) -> str:
        """
        Get semantic paraphrase for a song.

        Args:
            song_key: Song identifier
            original_lyrics: Original lyrics text
            use_manual: Whether to prefer manual paraphrases

        Returns:
            Paraphrased lyrics
        """
        # Prefer manual paraphrases for consistency
        if use_manual and song_key in MANUAL_PARAPHRASES:
            return MANUAL_PARAPHRASES[song_key]

        # Try GPT-4 generation
        if self.client:
            paraphrase = self.generate_paraphrase_gpt(original_lyrics)
            if paraphrase:
                return paraphrase

        # Fallback to manual if available
        if song_key in MANUAL_PARAPHRASES:
            return MANUAL_PARAPHRASES[song_key]

        raise ValueError(f"No paraphrase available for {song_key}")


# Song configurations
SONG_CONFIGS = {
    "lose_yourself": {
        "name": "Lose Yourself",
        "artist": "Eminem",
        "genre": "Rap",
        "style": "intense rap, motivational, rhythmic flow"
    },
    "dna": {
        "name": "DNA",
        "artist": "Kendrick Lamar",
        "genre": "Rap",
        "style": "rap, gangsta, trap, aggressive, rhythmic"
    },
    "humble": {
        "name": "HUMBLE",
        "artist": "Kendrick Lamar",
        "genre": "Rap",
        "style": "trap gangsta, bold, sparse, direct"
    },
    "espresso": {
        "name": "Espresso",
        "artist": "Sabrina Carpenter",
        "genre": "Pop",
        "style": "pop, upbeat, catchy, female vocal"
    },
    "let_it_be": {
        "name": "Let It Be",
        "artist": "The Beatles",
        "genre": "Pop",
        "style": "rock ballad, piano, inspirational, male vocal"
    }
}


def compute_phonetic_distance(original: str, paraphrase: str) -> Dict:
    """
    Compute basic phonetic distance metrics between original and paraphrase.
    Higher distance = better paraphrase (more phonetically different).
    """
    # Simple metrics without CMU dictionary dependency
    orig_words = set(original.lower().split())
    para_words = set(paraphrase.lower().split())

    # Word overlap (lower = better)
    overlap = len(orig_words & para_words) / max(len(orig_words), 1)

    # Unique words in paraphrase
    unique_ratio = len(para_words - orig_words) / max(len(para_words), 1)

    # Length difference
    len_diff = abs(len(original) - len(paraphrase)) / max(len(original), 1)

    return {
        "word_overlap": overlap,
        "unique_word_ratio": unique_ratio,
        "length_difference": len_diff,
        "phonetic_distance_estimate": (1 - overlap) * 0.5 + unique_ratio * 0.5
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic paraphrases for Experiment 2"
    )
    parser.add_argument(
        "--output",
        default="semantic_paraphrases",
        help="Output directory"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--use-gpt",
        action="store_true",
        help="Generate new paraphrases with GPT-4 instead of using manual ones"
    )
    parser.add_argument(
        "--songs",
        nargs="+",
        default=None,
        help="Specific songs (default: all available)"
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "baseline_experiments" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SemanticParaphraseGenerator(args.api_key)

    songs = args.songs if args.songs else list(SONG_CONFIGS.keys())

    print("\n" + "="*60)
    print("EXPERIMENT 2: SEMANTIC PARAPHRASE GENERATOR")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Using GPT-4: {args.use_gpt and generator.client is not None}")
    print(f"Songs: {', '.join(songs)}")
    print("="*60 + "\n")

    results = {}
    all_metrics = []

    for song_key in songs:
        if song_key not in SONG_CONFIGS:
            print(f"Warning: Unknown song '{song_key}', skipping")
            continue

        config = SONG_CONFIGS[song_key]
        orig_path = base_dir / f"{song_key}_orig.txt"

        if not orig_path.exists():
            print(f"Warning: Original lyrics not found for {song_key}")
            continue

        # Load original
        with open(orig_path, 'r') as f:
            original_lyrics = f.read().strip()

        print(f"\nProcessing: {config['name']} ({config['artist']})")

        try:
            # Generate/get paraphrase
            paraphrase = generator.get_paraphrase(
                song_key,
                original_lyrics,
                use_manual=not args.use_gpt
            )

            # Save paraphrase
            output_path = output_dir / f"{song_key}_paraphrase.txt"
            with open(output_path, 'w') as f:
                f.write(paraphrase)

            # Compute metrics
            metrics = compute_phonetic_distance(original_lyrics, paraphrase)

            results[song_key] = {
                "config": config,
                "original_path": str(orig_path),
                "paraphrase_path": str(output_path),
                "metrics": metrics
            }

            all_metrics.append({
                "song": song_key,
                **metrics
            })

            print(f"  Saved: {output_path.name}")
            print(f"  Word overlap: {metrics['word_overlap']:.2%}")
            print(f"  Unique words: {metrics['unique_word_ratio']:.2%}")
            print(f"  Phonetic distance estimate: {metrics['phonetic_distance_estimate']:.2f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save manifest
    manifest_path = output_dir / "paraphrase_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save comparison report
    report_path = output_dir / "paraphrase_comparison.json"
    with open(report_path, 'w') as f:
        json.dump({
            "description": "Semantic paraphrases preserve meaning but break phonetic structure",
            "expected_audiojudge": {
                "melody": "0.30-0.50 (different tune)",
                "rhythm": "0.30-0.50 (different rhythm)",
                "harmony": "0.30-0.50 (different harmony)"
            },
            "comparison_with_apt": {
                "apt_melody": "0.85-0.95 (same tune due to phonetic similarity)",
                "paraphrase_melody": "0.30-0.50 (different tune despite same meaning)"
            },
            "metrics": all_metrics
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total songs: {len(results)}")
    print(f"Manifest: {manifest_path}")
    print(f"Comparison report: {report_path}")
    print("\n--- Average Metrics ---")
    if all_metrics:
        avg_overlap = sum(m["word_overlap"] for m in all_metrics) / len(all_metrics)
        avg_unique = sum(m["unique_word_ratio"] for m in all_metrics) / len(all_metrics)
        avg_distance = sum(m["phonetic_distance_estimate"] for m in all_metrics) / len(all_metrics)
        print(f"  Avg word overlap: {avg_overlap:.2%} (lower = better)")
        print(f"  Avg unique words: {avg_unique:.2%} (higher = better)")
        print(f"  Avg phonetic distance: {avg_distance:.2f} (higher = better)")

    print("\nNext step: Run baseline_suno_generator.py --experiment paraphrase")


if __name__ == "__main__":
    main()
