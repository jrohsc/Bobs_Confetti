"""
Suno API Song Generator for APT Attack Evaluation
Generates k=3 independent samples per song for statistical rigor.

Usage:
    python suno_generator.py --api-url http://localhost:3000 --k 3

Requirements:
    - suno-api running (https://github.com/gcui-art/suno-api)
    - SUNO_COOKIE environment variable set in suno-api
"""

import os
import json
import time
import requests
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Song configurations matching the paper's Table 1
SONG_CONFIGS = {
    "dna": {
        "name": "DNA",
        "artist": "Kendrick Lamar",
        "genre": "Rap",
        "lyrics_file": "dna_apt.txt",
        "style": "rap, gangsta, trap, aggressive, rhythmic",
        "title": "BMA (DNA Parody)"
    },
    "lose_yourself": {
        "name": "Lose Yourself",
        "artist": "Eminem",
        "genre": "Rap",
        "lyrics_file": "lose_yourself_apt.txt",
        "style": "intense rap, motivational, rhythmic flow",
        "title": "Bob's Confetti (Lose Yourself Parody)"
    },
    "humble": {
        "name": "HUMBLE",
        "artist": "Kendrick Lamar",
        "genre": "Rap",
        "lyrics_file": "HUMBLE_apt.txt",
        "style": "trap gangsta, bold, sparse, direct",
        "title": "HUMBLE Parody"
    },
    "espresso": {
        "name": "Espresso",
        "artist": "Sabrina Carpenter",
        "genre": "Pop",
        "lyrics_file": "espresso_apt.txt",
        "style": "pop, upbeat, catchy, female vocal",
        "title": "Depresso (Espresso Parody)"
    },
    "let_it_be": {
        "name": "Let It Be",
        "artist": "The Beatles",
        "genre": "Pop",
        "lyrics_file": "let_it_be_apt.txt",
        "style": "rock ballad, piano, inspirational, male vocal",
        "title": "Get It Free (Let It Be Parody)"
    },
    "we_will_rock_you": {
        "name": "We Will Rock You",
        "artist": "Queen",
        "genre": "Pop",
        "lyrics_file": "we_will_rock_you_apt.txt",
        "style": "rock anthem, stomping beat, powerful",
        "title": "We Will Mock You"
    },
    "apt": {
        "name": "APT",
        "artist": "ROSÉ & Bruno Mars",
        "genre": "Korean",
        "lyrics_file": "apt_apt.txt",
        "style": "K-pop, catchy, dance, energetic",
        "title": "APT Parody"
    },
    "gangnam_style": {
        "name": "Gangnam Style",
        "artist": "PSY",
        "genre": "Korean",
        "lyrics_file": "gangnam_apt.txt",
        "style": "K-pop, electronic, dance, energetic male vocal",
        "title": "Gangnam Parody"
    },
    "gentleman": {
        "name": "GENTLEMAN",
        "artist": "PSY",
        "genre": "Korean",
        "lyrics_file": "gentleman_apt.txt",
        "style": "K-pop, electronic, dance, male vocal",
        "title": "Gentleman Parody"
    },
    "jingle_bell": {
        "name": "Jingle Bells",
        "artist": "Traditional",
        "genre": "Christmas",
        "lyrics_file": "jingle_bell_apt.txt",
        "style": "christmas, traditional, cheerful",
        "title": "Giggle Shells"
    },
    "jingle_bell_rock": {
        "name": "Jingle Bell Rock",
        "artist": "Bobby Helms",
        "genre": "Christmas",
        "lyrics_file": "rock_jingle_bell_apt.txt",
        "style": "christmas style, rock, upbeat",
        "title": "Giggle Shell Sock"
    }
}


class SunoGenerator:
    """Handler for Suno API song generation."""

    def __init__(self, api_url: str = "http://localhost:3000"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()

    def check_connection(self) -> bool:
        """Verify API is accessible."""
        try:
            resp = self.session.get(f"{self.api_url}/api/get_limit", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"API connected. Credits left: {data.get('credits_left', 'N/A')}")
                return True
        except Exception as e:
            logger.error(f"API connection failed: {e}")
        return False

    def get_quota(self) -> dict:
        """Get current quota information."""
        try:
            resp = self.session.get(f"{self.api_url}/api/get_limit", timeout=10)
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get quota: {e}")
            return {}

    def generate_song(
        self,
        lyrics: str,
        style: str,
        title: str,
        make_instrumental: bool = False,
        wait_audio: bool = True
    ) -> Optional[dict]:
        """
        Generate a song using custom_generate endpoint.

        Args:
            lyrics: The lyrics/prompt for the song
            style: Genre/style description
            title: Song title
            make_instrumental: Whether to create instrumental only
            wait_audio: Whether to wait for audio generation to complete

        Returns:
            Response data with song IDs and URLs
        """
        payload = {
            "prompt": lyrics,
            "tags": style,
            "title": title,
            "make_instrumental": make_instrumental,
            "wait_audio": wait_audio
        }

        try:
            logger.info(f"Generating: {title} (style: {style[:50]}...)")
            resp = self.session.post(
                f"{self.api_url}/api/custom_generate",
                json=payload,
                timeout=300  # 5 min timeout for generation
            )

            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"Generation initiated successfully")
                return data
            else:
                logger.error(f"Generation failed: {resp.status_code} - {resp.text}")
                return None

        except requests.Timeout:
            logger.error("Generation timed out")
            return None
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return None

    def get_song_info(self, song_ids: list) -> Optional[dict]:
        """Get information about generated songs."""
        try:
            ids_str = ",".join(song_ids)
            resp = self.session.get(
                f"{self.api_url}/api/get",
                params={"ids": ids_str},
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"Failed to get song info: {e}")
        return None

    def wait_for_completion(
        self,
        song_ids: list,
        max_wait: int = 600,
        poll_interval: int = 10
    ) -> list:
        """
        Poll until songs are complete or timeout.

        Args:
            song_ids: List of song IDs to wait for
            max_wait: Maximum wait time in seconds
            poll_interval: Time between polls in seconds

        Returns:
            List of completed song data
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            songs = self.get_song_info(song_ids)
            if songs:
                # Check if all songs are complete
                all_complete = all(
                    s.get('status') == 'complete' or s.get('audio_url')
                    for s in songs
                )
                if all_complete:
                    logger.info("All songs completed!")
                    return songs

                # Log progress
                for s in songs:
                    status = s.get('status', 'unknown')
                    logger.info(f"  Song {s.get('id', 'N/A')}: {status}")

            time.sleep(poll_interval)

        logger.warning("Timeout waiting for song completion")
        return self.get_song_info(song_ids) or []


def load_lyrics(lyrics_file: str, base_dir: str) -> str:
    """Load lyrics from file."""
    filepath = Path(base_dir) / lyrics_file
    if not filepath.exists():
        raise FileNotFoundError(f"Lyrics file not found: {filepath}")
    return filepath.read_text(encoding='utf-8')


def generate_k_samples(
    generator: SunoGenerator,
    song_key: str,
    config: dict,
    base_dir: str,
    output_dir: str,
    k: int = 3
) -> list:
    """
    Generate k independent samples for a single song.

    Args:
        generator: SunoGenerator instance
        song_key: Key for the song in SONG_CONFIGS
        config: Song configuration dict
        base_dir: Directory containing lyrics files
        output_dir: Directory to save results
        k: Number of samples to generate

    Returns:
        List of generation results
    """
    results = []

    # Load lyrics
    try:
        lyrics = load_lyrics(config['lyrics_file'], base_dir)
    except FileNotFoundError as e:
        logger.error(f"Skipping {song_key}: {e}")
        return results

    logger.info(f"\n{'='*60}")
    logger.info(f"Generating k={k} samples for: {config['name']} ({config['artist']})")
    logger.info(f"Genre: {config['genre']}, Style: {config['style']}")
    logger.info(f"{'='*60}")

    for i in range(k):
        sample_num = i + 1
        logger.info(f"\n--- Sample {sample_num}/{k} ---")

        # Generate song
        result = generator.generate_song(
            lyrics=lyrics,
            style=config['style'],
            title=f"{config['title']} (Sample {sample_num})",
            wait_audio=True
        )

        if result:
            # Add metadata
            sample_result = {
                "song_key": song_key,
                "song_name": config['name'],
                "artist": config['artist'],
                "genre": config['genre'],
                "sample_num": sample_num,
                "timestamp": datetime.now().isoformat(),
                "api_response": result
            }
            results.append(sample_result)

            # Save individual result
            result_file = Path(output_dir) / f"{song_key}_sample{sample_num}.json"
            with open(result_file, 'w') as f:
                json.dump(sample_result, f, indent=2)
            logger.info(f"Saved: {result_file}")

            # Brief delay between generations to avoid rate limiting
            if i < k - 1:
                logger.info("Waiting 5s before next generation...")
                time.sleep(5)
        else:
            logger.error(f"Failed to generate sample {sample_num} for {song_key}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate k=3 song samples using Suno API for APT attack evaluation"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:3000",
        help="Suno API base URL (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of samples per song (default: 3)"
    )
    parser.add_argument(
        "--songs",
        nargs="+",
        default=None,
        help="Specific songs to generate (default: all). Options: " +
             ", ".join(SONG_CONFIGS.keys())
    )
    parser.add_argument(
        "--output-dir",
        default="generated_songs",
        help="Output directory for results (default: generated_songs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without making API calls"
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Select songs to generate
    songs_to_generate = args.songs if args.songs else list(SONG_CONFIGS.keys())

    # Validate song keys
    invalid_songs = [s for s in songs_to_generate if s not in SONG_CONFIGS]
    if invalid_songs:
        logger.error(f"Invalid song keys: {invalid_songs}")
        logger.info(f"Valid options: {list(SONG_CONFIGS.keys())}")
        return

    # Show plan
    logger.info("\n" + "="*70)
    logger.info("SUNO API SONG GENERATION - APT Attack Evaluation")
    logger.info("="*70)
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Samples per song (k): {args.k}")
    logger.info(f"Total generations: {len(songs_to_generate) * args.k}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nSongs to generate:")
    for song_key in songs_to_generate:
        config = SONG_CONFIGS[song_key]
        logger.info(f"  - {config['name']} ({config['artist']}) [{config['genre']}]")

    if args.dry_run:
        logger.info("\n[DRY RUN] No API calls will be made.")
        return

    # Initialize generator
    generator = SunoGenerator(args.api_url)

    # Check API connection
    if not generator.check_connection():
        logger.error("Cannot connect to Suno API. Please ensure:")
        logger.error("  1. suno-api is running (npm run dev)")
        logger.error("  2. SUNO_COOKIE is set correctly")
        logger.error("  3. API URL is correct")
        return

    # Check quota
    quota = generator.get_quota()
    credits = quota.get('credits_left', 0)
    needed = len(songs_to_generate) * args.k * 10  # ~10 credits per song
    logger.info(f"\nCredits available: {credits}")
    logger.info(f"Estimated credits needed: {needed}")

    if credits < needed:
        logger.warning(f"Warning: May not have enough credits!")

    # Generate all songs
    all_results = []

    for song_key in songs_to_generate:
        config = SONG_CONFIGS[song_key]
        results = generate_k_samples(
            generator=generator,
            song_key=song_key,
            config=config,
            base_dir=str(base_dir),
            output_dir=str(output_dir),
            k=args.k
        )
        all_results.extend(results)

        # Brief pause between different songs
        if song_key != songs_to_generate[-1]:
            logger.info("\nPausing 10s before next song...")
            time.sleep(10)

    # Save summary
    summary_file = output_dir / f"generation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "api_url": args.api_url,
        "k": args.k,
        "songs_generated": len(songs_to_generate),
        "total_samples": len(all_results),
        "results": all_results
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total songs: {len(songs_to_generate)}")
    logger.info(f"Total samples: {len(all_results)}")
    logger.info(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
