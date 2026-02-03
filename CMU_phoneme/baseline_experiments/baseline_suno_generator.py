"""
Baseline Experiment SUNO Generator with Content Filter Logging

This script generates songs for all baseline conditions and logs content filter responses.
Critical for demonstrating that:
1. Original/verbatim lyrics (AVT) → BLOCKED by content filter
2. APT lyrics (phonetically similar) → PASSES filter, triggers memorization
3. Random lyrics → PASSES filter, NO memorization (control)
4. Semantic paraphrase → PASSES filter, NO memorization (control)

Usage:
    python baseline_suno_generator.py --api-url http://localhost:3000 --experiment all
"""

import os
import json
import time
import requests
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('baseline_generation.log')
    ]
)
logger = logging.getLogger(__name__)


class FilterStatus(Enum):
    """Content filter response status."""
    PASSED = "passed"
    BLOCKED = "blocked"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ContentFilterLog:
    """Logs and tracks content filter responses for analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logs: List[Dict[str, Any]] = []
        self.log_file = output_dir / "content_filter_log.json"
        self.summary_file = output_dir / "content_filter_summary.json"

    def log_attempt(
        self,
        experiment_type: str,
        song_name: str,
        lyrics_type: str,
        lyrics_content: str,
        api_response: Optional[Dict],
        status: FilterStatus,
        error_message: str = None,
        raw_response: str = None
    ):
        """Log a generation attempt with filter status."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": experiment_type,
            "song_name": song_name,
            "lyrics_type": lyrics_type,  # "original", "apt", "random", "paraphrase"
            "lyrics_preview": lyrics_content[:200] + "..." if len(lyrics_content) > 200 else lyrics_content,
            "lyrics_length": len(lyrics_content),
            "filter_status": status.value,
            "error_message": error_message,
            "raw_response": raw_response,
            "api_response": api_response
        }

        self.logs.append(entry)

        # Log to console with clear formatting
        status_emoji = {
            FilterStatus.PASSED: "✓",
            FilterStatus.BLOCKED: "✗",
            FilterStatus.ERROR: "!",
            FilterStatus.TIMEOUT: "⏱",
            FilterStatus.UNKNOWN: "?"
        }

        logger.info(
            f"[{status_emoji[status]} {status.value.upper()}] "
            f"{experiment_type}/{song_name} ({lyrics_type})"
        )
        if error_message:
            logger.info(f"  → Error: {error_message}")

        # Save after each entry for durability
        self._save()

    def _save(self):
        """Save logs to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def generate_summary(self) -> Dict:
        """Generate summary statistics of filter responses."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_attempts": len(self.logs),
            "by_status": {},
            "by_lyrics_type": {},
            "by_experiment": {},
            "blocked_details": []
        }

        for entry in self.logs:
            status = entry["filter_status"]
            lyrics_type = entry["lyrics_type"]
            exp_type = entry["experiment_type"]

            # Count by status
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

            # Count by lyrics type
            if lyrics_type not in summary["by_lyrics_type"]:
                summary["by_lyrics_type"][lyrics_type] = {"passed": 0, "blocked": 0, "error": 0}
            if status == "passed":
                summary["by_lyrics_type"][lyrics_type]["passed"] += 1
            elif status == "blocked":
                summary["by_lyrics_type"][lyrics_type]["blocked"] += 1
            else:
                summary["by_lyrics_type"][lyrics_type]["error"] += 1

            # Count by experiment
            if exp_type not in summary["by_experiment"]:
                summary["by_experiment"][exp_type] = {"passed": 0, "blocked": 0, "error": 0}
            if status == "passed":
                summary["by_experiment"][exp_type]["passed"] += 1
            elif status == "blocked":
                summary["by_experiment"][exp_type]["blocked"] += 1
            else:
                summary["by_experiment"][exp_type]["error"] += 1

            # Collect blocked details
            if status == "blocked":
                summary["blocked_details"].append({
                    "song": entry["song_name"],
                    "lyrics_type": lyrics_type,
                    "error": entry["error_message"],
                    "lyrics_preview": entry["lyrics_preview"]
                })

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.generate_summary()

        print("\n" + "="*70)
        print("CONTENT FILTER ANALYSIS SUMMARY")
        print("="*70)

        print(f"\nTotal Attempts: {summary['total_attempts']}")

        print("\n--- By Status ---")
        for status, count in summary["by_status"].items():
            pct = (count / summary["total_attempts"] * 100) if summary["total_attempts"] > 0 else 0
            print(f"  {status}: {count} ({pct:.1f}%)")

        print("\n--- By Lyrics Type ---")
        for ltype, counts in summary["by_lyrics_type"].items():
            total = counts["passed"] + counts["blocked"] + counts["error"]
            block_rate = (counts["blocked"] / total * 100) if total > 0 else 0
            print(f"  {ltype}:")
            print(f"    Passed: {counts['passed']}, Blocked: {counts['blocked']}, Error: {counts['error']}")
            print(f"    Block Rate: {block_rate:.1f}%")

        print("\n--- Blocked Entries (Evidence for Paper) ---")
        for detail in summary["blocked_details"]:
            print(f"  [{detail['song']}] {detail['lyrics_type']}")
            print(f"    Error: {detail['error']}")
            print(f"    Preview: {detail['lyrics_preview'][:100]}...")
            print()

        print("="*70)
        print(f"Full log saved to: {self.log_file}")
        print(f"Summary saved to: {self.summary_file}")
        print("="*70 + "\n")


class BaselineSunoGenerator:
    """SUNO API generator with content filter tracking."""

    # Known content filter error patterns
    FILTER_PATTERNS = [
        "copyright",
        "copyrighted",
        "content policy",
        "content filter",
        "blocked",
        "not allowed",
        "violates",
        "prohibited",
        "cannot generate",
        "inappropriate",
        "rejected"
    ]

    def __init__(self, api_url: str, filter_log: ContentFilterLog):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        self.filter_log = filter_log

    def check_connection(self) -> bool:
        """Verify API is accessible."""
        try:
            resp = self.session.get(f"{self.api_url}/api/get_limit", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"API connected. Credits: {data.get('credits_left', 'N/A')}")
                return True
        except Exception as e:
            logger.error(f"API connection failed: {e}")
        return False

    def _check_filter_response(self, response_text: str, status_code: int) -> tuple[FilterStatus, str]:
        """Analyze response to determine if content was filtered."""
        response_lower = response_text.lower()

        # Check for known filter patterns
        for pattern in self.FILTER_PATTERNS:
            if pattern in response_lower:
                return FilterStatus.BLOCKED, f"Content filter triggered: {pattern}"

        # Check HTTP status codes
        if status_code == 400:
            return FilterStatus.BLOCKED, f"Bad request (likely filtered): {response_text[:200]}"
        elif status_code == 403:
            return FilterStatus.BLOCKED, f"Forbidden (content policy): {response_text[:200]}"
        elif status_code == 429:
            return FilterStatus.ERROR, "Rate limited"
        elif status_code >= 500:
            return FilterStatus.ERROR, f"Server error: {status_code}"
        elif status_code == 200:
            return FilterStatus.PASSED, None

        return FilterStatus.UNKNOWN, f"Unknown response: {status_code} - {response_text[:200]}"

    def generate_song(
        self,
        lyrics: str,
        style: str,
        title: str,
        experiment_type: str,
        song_name: str,
        lyrics_type: str,
        wait_audio: bool = True
    ) -> Optional[Dict]:
        """
        Generate a song and log filter response.

        Args:
            lyrics: The lyrics/prompt
            style: Genre/style description
            title: Song title
            experiment_type: "random_baseline", "semantic_paraphrase", "avt", "apt"
            song_name: Original song name for reference
            lyrics_type: "original", "apt", "random", "paraphrase"
            wait_audio: Whether to wait for generation

        Returns:
            API response dict or None if failed/blocked
        """
        payload = {
            "prompt": lyrics,
            "tags": style,
            "title": title,
            "make_instrumental": False,
            "wait_audio": wait_audio
        }

        try:
            logger.info(f"Attempting: {title} ({lyrics_type})")

            resp = self.session.post(
                f"{self.api_url}/api/custom_generate",
                json=payload,
                timeout=300
            )

            # Analyze filter response
            status, error_msg = self._check_filter_response(resp.text, resp.status_code)

            # Parse response if successful
            api_response = None
            if status == FilterStatus.PASSED:
                try:
                    api_response = resp.json()
                except:
                    pass

            # Log the attempt
            self.filter_log.log_attempt(
                experiment_type=experiment_type,
                song_name=song_name,
                lyrics_type=lyrics_type,
                lyrics_content=lyrics,
                api_response=api_response,
                status=status,
                error_message=error_msg,
                raw_response=resp.text[:500] if resp.text else None
            )

            if status == FilterStatus.PASSED:
                return api_response
            else:
                return None

        except requests.Timeout:
            self.filter_log.log_attempt(
                experiment_type=experiment_type,
                song_name=song_name,
                lyrics_type=lyrics_type,
                lyrics_content=lyrics,
                api_response=None,
                status=FilterStatus.TIMEOUT,
                error_message="Request timed out after 300s"
            )
            return None

        except Exception as e:
            self.filter_log.log_attempt(
                experiment_type=experiment_type,
                song_name=song_name,
                lyrics_type=lyrics_type,
                lyrics_content=lyrics,
                api_response=None,
                status=FilterStatus.ERROR,
                error_message=str(e)
            )
            return None


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


def load_lyrics(filepath: Path) -> str:
    """Load lyrics from file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Lyrics file not found: {filepath}")
    return filepath.read_text(encoding='utf-8').strip()


def run_avt_experiment(generator: BaselineSunoGenerator, base_dir: Path, output_dir: Path):
    """
    Run AVT (Adversarial VerbaTim) experiment - original lyrics.
    Expected: BLOCKED by content filter.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: AVT (Adversarial VerbaTim) - Original Lyrics")
    print("Expected: BLOCKED by content filter")
    print("="*60)

    results = []

    for song_key, config in SONG_CONFIGS.items():
        orig_path = base_dir / f"{song_key}_orig.txt"
        if not orig_path.exists():
            logger.warning(f"Original lyrics not found: {orig_path}")
            continue

        lyrics = load_lyrics(orig_path)

        result = generator.generate_song(
            lyrics=lyrics,
            style=config["style"],
            title=f"{config['name']} - Verbatim Test",
            experiment_type="avt_baseline",
            song_name=song_key,
            lyrics_type="original"
        )

        results.append({
            "song": song_key,
            "status": "passed" if result else "blocked/failed",
            "response": result
        })

        # Delay between attempts
        time.sleep(5)

    # Save results
    results_file = output_dir / "avt_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_apt_experiment(generator: BaselineSunoGenerator, base_dir: Path, output_dir: Path):
    """
    Run APT (Adversarial PhoneTic) experiment - phonetically similar lyrics.
    Expected: PASSES filter, triggers memorization.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: APT (Adversarial PhoneTic) - Phonetic Lyrics")
    print("Expected: PASSES filter, triggers memorization")
    print("="*60)

    results = []

    for song_key, config in SONG_CONFIGS.items():
        apt_path = base_dir / f"{song_key}_apt.txt"
        if not apt_path.exists():
            logger.warning(f"APT lyrics not found: {apt_path}")
            continue

        lyrics = load_lyrics(apt_path)

        result = generator.generate_song(
            lyrics=lyrics,
            style=config["style"],
            title=f"{config['name']} - APT Phonetic",
            experiment_type="apt_attack",
            song_name=song_key,
            lyrics_type="apt"
        )

        results.append({
            "song": song_key,
            "status": "passed" if result else "blocked/failed",
            "response": result
        })

        time.sleep(5)

    results_file = output_dir / "apt_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_random_experiment(generator: BaselineSunoGenerator, base_dir: Path, output_dir: Path):
    """
    Run random lyrics baseline experiment.
    Expected: PASSES filter, NO memorization (control).
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Random Lyrics Baseline")
    print("Expected: PASSES filter, NO memorization")
    print("="*60)

    random_dir = base_dir / "baseline_experiments" / "random_lyrics"

    if not random_dir.exists():
        logger.error(f"Random lyrics directory not found: {random_dir}")
        logger.info("Run random_lyrics_generator.py first")
        return []

    results = []

    for song_key, config in SONG_CONFIGS.items():
        # Try each variant
        for variant in range(1, 4):  # v1, v2, v3
            random_path = random_dir / f"{song_key}_random_v{variant}.txt"
            if not random_path.exists():
                continue

            lyrics = load_lyrics(random_path)

            result = generator.generate_song(
                lyrics=lyrics,
                style=config["style"],
                title=f"Random Control v{variant} ({config['name']})",
                experiment_type="random_baseline",
                song_name=song_key,
                lyrics_type="random"
            )

            results.append({
                "song": song_key,
                "variant": variant,
                "status": "passed" if result else "blocked/failed",
                "response": result
            })

            time.sleep(5)

    results_file = output_dir / "random_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_paraphrase_experiment(generator: BaselineSunoGenerator, base_dir: Path, output_dir: Path):
    """
    Run semantic paraphrase baseline experiment.
    Expected: PASSES filter, NO memorization (control).
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Semantic Paraphrase Baseline")
    print("Expected: PASSES filter, NO memorization")
    print("="*60)

    paraphrase_dir = base_dir / "baseline_experiments" / "semantic_paraphrases"

    if not paraphrase_dir.exists():
        logger.error(f"Paraphrase directory not found: {paraphrase_dir}")
        logger.info("Run semantic_paraphrase_generator.py first")
        return []

    results = []

    for song_key, config in SONG_CONFIGS.items():
        paraphrase_path = paraphrase_dir / f"{song_key}_paraphrase.txt"
        if not paraphrase_path.exists():
            continue

        lyrics = load_lyrics(paraphrase_path)

        result = generator.generate_song(
            lyrics=lyrics,
            style=config["style"],
            title=f"Semantic Paraphrase ({config['name']})",
            experiment_type="semantic_paraphrase",
            song_name=song_key,
            lyrics_type="paraphrase"
        )

        results.append({
            "song": song_key,
            "status": "passed" if result else "blocked/failed",
            "response": result
        })

        time.sleep(5)

    results_file = output_dir / "paraphrase_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline experiments with content filter logging"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:3000",
        help="Suno API base URL"
    )
    parser.add_argument(
        "--experiment",
        choices=["avt", "apt", "random", "paraphrase", "all"],
        default="all",
        help="Which experiment to run"
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_results",
        help="Output directory"
    )

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "baseline_experiments" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize filter logger
    filter_log = ContentFilterLog(output_dir)

    # Initialize generator
    generator = BaselineSunoGenerator(args.api_url, filter_log)

    print("\n" + "="*70)
    print("BASELINE EXPERIMENTS WITH CONTENT FILTER LOGGING")
    print("="*70)
    print(f"API URL: {args.api_url}")
    print(f"Experiment: {args.experiment}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Check connection
    if not generator.check_connection():
        logger.error("Cannot connect to SUNO API")
        return

    # Run experiments
    if args.experiment in ["avt", "all"]:
        run_avt_experiment(generator, base_dir, output_dir)

    if args.experiment in ["apt", "all"]:
        run_apt_experiment(generator, base_dir, output_dir)

    if args.experiment in ["random", "all"]:
        run_random_experiment(generator, base_dir, output_dir)

    if args.experiment in ["paraphrase", "all"]:
        run_paraphrase_experiment(generator, base_dir, output_dir)

    # Print summary
    filter_log.print_summary()


if __name__ == "__main__":
    main()
