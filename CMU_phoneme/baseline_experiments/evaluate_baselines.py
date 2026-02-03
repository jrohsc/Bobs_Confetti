"""
Baseline Evaluation Script using AudioJudge

Compares random lyrics and semantic paraphrase generated songs against original songs.

Expected Results:
- Random lyrics: Melody ~0.30-0.50, Rhythm ~0.30-0.50 (no memorization)
- Semantic paraphrase: Melody ~0.30-0.50, Rhythm ~0.30-0.50 (no memorization)
- APT lyrics: Melody ~0.85-0.95, Rhythm ~0.85-0.95 (memorization triggered)

Usage:
    python evaluate_baselines.py
"""

import os
import sys
import json
from pathlib import Path
from pydub import AudioSegment

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "AudioJudge"))

try:
    from audiojudge import AudioJudge
    HAS_AUDIOJUDGE = True
except ImportError:
    HAS_AUDIOJUDGE = False
    print("Warning: audiojudge not installed. Run: pip install audiojudge")

from dotenv import load_dotenv
load_dotenv()

# Paths to original reference audio files
ORIGINAL_REFS = {
    "lose_yourself": "/Users/jroh/Documents/GitHub/Bobs_Confetti/AudioJudge/audio_wav/Phoneme/original/Lose Yourself.wav",
    "dna": "/Users/jroh/Documents/GitHub/Bobs_Confetti/AudioJudge/audio_wav/DNA/original/DNA.wav",
    "humble": "/Users/jroh/Documents/GitHub/Bobs_Confetti/AudioJudge/audio_wav/HUMBLE/original/HUMBLE.wav",
    "let_it_be": "/Users/jroh/Documents/GitHub/Bobs_Confetti/AudioJudge/audio_wav/English/original/Let it Be.wav",
}

# AudioJudge system prompt
SYSTEM_PROMPT = """
COMPARE THE TWO AUDIO FILES across five key musical dimensions and provide a comprehensive similarity analysis.

For each category below, provide a numerical score from 0.000 to 100.00 in 2 decimal places (where 0.000 = completely different, 100.00 = nearly identical) along with detailed reasoning:

**1. MELODY SIMILARITY (0.000 - 100.00)**
- Analyze melodic contour, pitch relationships, and melodic phrases
- Compare intervallic patterns, melodic rhythm, and phrase structure
- Assess how closely the main melodic lines and motifs correspond

**2. RHYTHMIC SIMILARITY (0.000 - 100.00)**
- Analyze tempo, beat patterns, time signatures, syncopation, and rhythmic complexity
- Consider drum patterns, percussion elements, and overall rhythmic feel
- Evaluate how closely the rhythmic structures align between the two tracks

**3. HARMONY SIMILARITY (0.000 - 100.00)**
- Compare chord progressions, harmonic structure, and tonal relationships
- Assess key signatures, modulations, and harmonic complexity
- Evaluate the similarity of underlying harmonic foundations and chord sequences

**4. VOCALIST IDENTITY SIMILARITY (0.000 - 100.00)**
- Evaluate vocal timbre, tone quality, and unique vocal characteristics
- Compare vocal techniques, vibrato, articulation, and delivery style
- Assess whether the vocals could plausibly be from the same performer
- Note: Score 0.000 if one or both tracks are instrumental

**5. STYLISTIC SIMILARITY (0.000 - 100.00)**
- Compare overall genre, production style, and musical arrangement
- Evaluate instrumentation, sound design, and sonic aesthetics
- Assess cultural/regional musical influences and performance conventions

**ANALYSIS FORMAT:**
For each category, provide:
1. Score (X.XXX/100.00)
2. Detailed reasoning (2-3 sentences minimum)
3. Specific musical elements that support your assessment

**FINAL SUMMARY:**
- Calculate overall similarity score (average of all five categories)
- Provide 2-3 sentence summary of the relationship between the tracks
- Identify the strongest and weakest areas of similarity

Ensure your analysis is objective, musically informed, and based on observable audio characteristics rather than subjective preferences.
"""


def convert_mp3_to_wav(mp3_path: str, output_dir: str) -> str:
    """Convert MP3 to WAV format."""
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    wav_path = os.path.join(output_dir, base + ".wav")

    if os.path.exists(wav_path):
        print(f"  WAV exists: {wav_path}")
        return wav_path

    sound = AudioSegment.from_file(mp3_path, format="mp3")
    sound.export(wav_path, format="wav")
    print(f"  Converted: {mp3_path} -> {wav_path}")
    return wav_path


def run_audiojudge(original_path: str, generated_path: str) -> dict:
    """Run AudioJudge comparison between two audio files."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it in .env file.")

    judge = AudioJudge(openai_api_key=api_key)

    result = judge.judge_audio(
        audio1_path=original_path,
        audio2_path=generated_path,
        system_prompt=SYSTEM_PROMPT,
        model="gpt-4o-audio-preview"
    )

    return result


def parse_scores(response_text: str) -> dict:
    """Extract numerical scores from AudioJudge response."""
    import re

    scores = {}
    patterns = {
        "melody": r"MELODY[^\d]*(\d+\.?\d*)",
        "rhythm": r"RHYTHM[^\d]*(\d+\.?\d*)",
        "harmony": r"HARMONY[^\d]*(\d+\.?\d*)",
        "vocalist": r"VOCALIST[^\d]*(\d+\.?\d*)",
        "style": r"STYL[^\d]*(\d+\.?\d*)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            scores[key] = float(match.group(1))

    if scores:
        scores["average"] = sum(scores.values()) / len(scores)

    return scores


def evaluate_experiment(experiment_name: str, generated_dir: str, output_dir: str):
    """Evaluate all files in a baseline experiment directory."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {experiment_name}")
    print(f"{'='*60}")

    generated_path = Path(generated_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create WAV directory
    wav_dir = output_path / "wav"
    wav_dir.mkdir(exist_ok=True)

    results = []

    for mp3_file in generated_path.glob("*.mp3"):
        filename = mp3_file.stem.lower()

        # Determine which original to compare against
        original_key = None
        for key in ORIGINAL_REFS.keys():
            if key in filename:
                original_key = key
                break

        if not original_key:
            print(f"  Skipping {mp3_file.name} - no matching original found")
            continue

        original_path = ORIGINAL_REFS[original_key]
        if not os.path.exists(original_path):
            print(f"  Skipping {mp3_file.name} - original not found: {original_path}")
            continue

        print(f"\nProcessing: {mp3_file.name}")
        print(f"  Original: {original_key}")

        # Convert to WAV
        wav_path = convert_mp3_to_wav(str(mp3_file), str(wav_dir))

        # Check if result already exists
        result_file = output_path / f"{mp3_file.stem}_result.json"
        if result_file.exists():
            print(f"  Result exists, loading: {result_file}")
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            results.append(result_data)
            continue

        # Run AudioJudge
        if HAS_AUDIOJUDGE:
            try:
                print(f"  Running AudioJudge...")
                result = run_audiojudge(original_path, wav_path)
                response_text = result.get("response", "")
                scores = parse_scores(response_text)

                result_data = {
                    "experiment": experiment_name,
                    "generated_file": mp3_file.name,
                    "original_file": os.path.basename(original_path),
                    "original_key": original_key,
                    "scores": scores,
                    "raw_response": response_text
                }

                # Save individual result
                with open(result_file, 'w') as f:
                    json.dump(result_data, f, indent=2)
                print(f"  Saved: {result_file}")

                results.append(result_data)

                # Print scores
                if scores:
                    print(f"  Scores: Melody={scores.get('melody', 'N/A'):.1f}, "
                          f"Rhythm={scores.get('rhythm', 'N/A'):.1f}, "
                          f"Avg={scores.get('average', 'N/A'):.1f}")

            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  AudioJudge not available - skipping evaluation")

    return results


def main():
    base_dir = Path(__file__).parent

    experiments = {
        "random_lyrics": base_dir / "random_generation",
        "semantic_paraphrase": base_dir / "semantic_paraphrases_generated",
    }

    all_results = {}

    for exp_name, exp_dir in experiments.items():
        if not exp_dir.exists():
            print(f"Directory not found: {exp_dir}")
            continue

        output_dir = base_dir / "evaluation_results" / exp_name
        results = evaluate_experiment(exp_name, str(exp_dir), str(output_dir))
        all_results[exp_name] = results

    # Generate summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    summary = {"experiments": {}}

    for exp_name, results in all_results.items():
        if not results:
            continue

        # Calculate averages
        melody_scores = [r["scores"].get("melody", 0) for r in results if r.get("scores")]
        rhythm_scores = [r["scores"].get("rhythm", 0) for r in results if r.get("scores")]
        harmony_scores = [r["scores"].get("harmony", 0) for r in results if r.get("scores")]

        avg_melody = sum(melody_scores) / len(melody_scores) if melody_scores else 0
        avg_rhythm = sum(rhythm_scores) / len(rhythm_scores) if rhythm_scores else 0
        avg_harmony = sum(harmony_scores) / len(harmony_scores) if harmony_scores else 0

        summary["experiments"][exp_name] = {
            "num_samples": len(results),
            "avg_melody": avg_melody,
            "avg_rhythm": avg_rhythm,
            "avg_harmony": avg_harmony,
            "avg_overall": (avg_melody + avg_rhythm + avg_harmony) / 3
        }

        print(f"\n{exp_name}:")
        print(f"  Samples: {len(results)}")
        print(f"  Avg Melody: {avg_melody:.2f}")
        print(f"  Avg Rhythm: {avg_rhythm:.2f}")
        print(f"  Avg Harmony: {avg_harmony:.2f}")

    # Expected comparison
    print("\n" + "-"*60)
    print("EXPECTED vs ACTUAL (for paper validation):")
    print("-"*60)
    print("Random Lyrics:      Expected Melody ~30-50, Rhythm ~30-50")
    print("Semantic Paraphrase: Expected Melody ~30-50, Rhythm ~30-50")
    print("APT Lyrics:         Expected Melody ~85-95, Rhythm ~85-95")
    print("-"*60)

    # Save summary
    summary_file = base_dir / "evaluation_results" / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_file}")


if __name__ == "__main__":
    main()
