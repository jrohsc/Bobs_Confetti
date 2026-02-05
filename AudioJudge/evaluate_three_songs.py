#!/usr/bin/env python3
"""
Evaluate missing AudioJudge results for three songs:
- cant_help_falling_in_love_with_you (5 files)
- Espresso (2 additional files)
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
import scipy.stats as stats

# Load environment
env_path = Path('/Users/jroh/Documents/GitHub/Bobs_Confetti/.env')
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

client = OpenAI()

AUDIO_BASE = Path('/Users/jroh/Documents/GitHub/Bobs_Confetti/audio')
OUTPUT_DIR = Path('/Users/jroh/Documents/GitHub/Bobs_Confetti/AudioJudge/all_songs_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_to_wav(input_path, output_path):
    """Convert audio file to WAV format."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format='wav')
        return True
    except Exception as e:
        print(f"  Error converting {input_path}: {e}")
        return False

def encode_audio_to_base64(audio_path):
    """Encode audio file to base64."""
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_audiojudge(original_path, generated_path):
    """Run AudioJudge comparison using OpenAI API."""

    # Convert to WAV if needed
    temp_dir = OUTPUT_DIR / 'temp_wav'
    temp_dir.mkdir(exist_ok=True)

    orig_wav = temp_dir / f"orig_{Path(original_path).stem}.wav"
    gen_wav = temp_dir / f"gen_{Path(generated_path).stem}.wav"

    # Convert original
    if str(original_path).endswith('.wav'):
        orig_wav = original_path
    elif not orig_wav.exists():
        if not convert_to_wav(original_path, orig_wav):
            return None

    # Convert generated
    if str(generated_path).endswith('.wav'):
        gen_wav = generated_path
    elif not gen_wav.exists():
        if not convert_to_wav(generated_path, gen_wav):
            return None

    # Encode audio
    orig_b64 = encode_audio_to_base64(orig_wav)
    gen_b64 = encode_audio_to_base64(gen_wav)

    prompt = """You are an expert music analyst. Compare the GENERATED audio to the ORIGINAL audio and rate their similarity on these criteria (0-100 scale):

1. MELODY SIMILARITY: How similar are the melodic lines, pitch contours, and intervals?
2. RHYTHMIC SIMILARITY: How similar are the tempo, beat patterns, and rhythmic structure?
3. HARMONY SIMILARITY: How similar are the chord progressions and harmonic content?
4. VOCALIST IDENTITY SIMILARITY: How similar do the vocals sound (timbre, style)?
5. STYLISTIC SIMILARITY: How similar is the overall genre, instrumentation, and production style?

IMPORTANT: You MUST end your response with EXACTLY this format (with actual numbers):
SCORES:
melody: [NUMBER]
rhythm: [NUMBER]
harmony: [NUMBER]
vocalist: [NUMBER]
style: [NUMBER]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the ORIGINAL song:"},
                        {"type": "input_audio", "input_audio": {"data": orig_b64, "format": "wav"}}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the GENERATED song:"},
                        {"type": "input_audio", "input_audio": {"data": gen_b64, "format": "wav"}}
                    ]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ],
            modalities=["text"]
        )

        response_text = response.choices[0].message.content

        # Parse scores
        scores = {}
        for line in response_text.split('\n'):
            line = line.strip().lower()
            for metric in ['melody', 'rhythm', 'harmony', 'vocalist', 'style']:
                if line.startswith(f'{metric}:'):
                    try:
                        value = line.split(':')[1].strip()
                        value = ''.join(c for c in value if c.isdigit() or c == '.')
                        scores[metric] = float(value)
                    except:
                        pass

        return scores if len(scores) >= 3 else None

    except Exception as e:
        print(f"  API Error: {e}")
        return None

def evaluate_song(song_name, original_path, generated_files):
    """Evaluate all generated files for a song."""
    print(f"\n{'='*60}")
    print(f"Processing: {song_name}")
    print(f"{'='*60}")
    print(f"Original: {original_path.name}")

    song_output_dir = OUTPUT_DIR / song_name
    song_output_dir.mkdir(exist_ok=True)

    results = []

    for gen_file in generated_files:
        print(f"\n  Evaluating: {gen_file.name}")

        # Check if already evaluated
        result_file = song_output_dir / f"{gen_file.stem}.json"
        if result_file.exists():
            with open(result_file) as f:
                cached = json.load(f)
                if 'scores' in cached:
                    scores = cached['scores']
                    print(f"    (cached) Melody: {scores.get('melody', 'N/A')}, Rhythm: {scores.get('rhythm', 'N/A')}, Harmony: {scores.get('harmony', 'N/A')}")
                    results.append({**scores, 'file': gen_file.name})
                    continue

        # Run evaluation
        scores = run_audiojudge(original_path, gen_file)

        if scores:
            print(f"    → Melody: {scores.get('melody', 'N/A')}, Rhythm: {scores.get('rhythm', 'N/A')}, Harmony: {scores.get('harmony', 'N/A')}")
            results.append({**scores, 'file': gen_file.name})

            # Save individual result
            with open(result_file, 'w') as f:
                json.dump({
                    'original': str(original_path),
                    'generated': str(gen_file),
                    'scores': scores
                }, f, indent=2)
        else:
            print(f"    → Failed to evaluate")

    return results

def calculate_statistics(scores_list, metrics=['melody', 'rhythm', 'harmony', 'vocalist', 'style']):
    """Calculate mean, std for each metric."""
    results = {}
    for metric in metrics:
        values = [s[metric] for s in scores_list if metric in s]
        if len(values) >= 1:
            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0
            results[metric] = {
                'mean': mean,
                'std': std,
                'n': len(values)
            }
    return results

def main():
    print("=" * 80)
    print("AUDIOJUDGE EVALUATION FOR THREE SONGS")
    print("=" * 80)

    all_results = {}

    # 1. cant_help_falling_in_love_with_you
    song_name = "cant_help_falling_in_love_with_you"
    original_path = AUDIO_BASE / song_name / "original" / "Can't Help Falling in Love.mp3"
    generated_dir = AUDIO_BASE / song_name / "generated"
    generated_files = list(generated_dir.glob('*.mp3'))

    if original_path.exists() and generated_files:
        results = evaluate_song(song_name, original_path, generated_files)
        all_results[song_name] = results

    # 2. Espresso (only new files)
    song_name = "Espresso"
    original_path = AUDIO_BASE / song_name / "original" / "Espresso_original.wav"
    generated_dir = AUDIO_BASE / song_name / "generated"
    # Get all files including the new ones
    generated_files = list(generated_dir.glob('*.mp3'))

    if original_path.exists() and generated_files:
        results = evaluate_song(song_name, original_path, generated_files)
        all_results[song_name] = results

    # 3. DOPE (load existing results)
    song_name = "DOPE"
    dope_results = []
    dope_dir = OUTPUT_DIR / "DOPE"
    if dope_dir.exists():
        for result_file in dope_dir.glob('*.json'):
            with open(result_file) as f:
                cached = json.load(f)
                if 'scores' in cached:
                    dope_results.append({**cached['scores'], 'file': result_file.stem + '.mp3'})
        all_results[song_name] = dope_results
        print(f"\n{'='*60}")
        print(f"Loaded existing results for DOPE: {len(dope_results)} evaluations")
        print(f"{'='*60}")

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)

    print(f"\n{'Song':<35} | {'N':>3} | {'Melody':>12} | {'Rhythm':>12} | {'Harmony':>12} | {'Identity':>12} | {'Style':>12}")
    print("-" * 120)

    for song_name, results in all_results.items():
        if not results:
            continue
        stats = calculate_statistics(results)
        n = stats.get('melody', {}).get('n', 0)

        melody_str = f"{stats['melody']['mean']:.1f} ± {stats['melody']['std']:.1f}" if 'melody' in stats else "N/A"
        rhythm_str = f"{stats['rhythm']['mean']:.1f} ± {stats['rhythm']['std']:.1f}" if 'rhythm' in stats else "N/A"
        harmony_str = f"{stats['harmony']['mean']:.1f} ± {stats['harmony']['std']:.1f}" if 'harmony' in stats else "N/A"
        vocalist_str = f"{stats['vocalist']['mean']:.1f} ± {stats['vocalist']['std']:.1f}" if 'vocalist' in stats else "N/A"
        style_str = f"{stats['style']['mean']:.1f} ± {stats['style']['std']:.1f}" if 'style' in stats else "N/A"

        print(f"{song_name:<35} | {n:>3} | {melody_str:>12} | {rhythm_str:>12} | {harmony_str:>12} | {vocalist_str:>12} | {style_str:>12}")

    # Save results to JSON
    summary = {
        'songs': {}
    }
    for song_name, results in all_results.items():
        if results:
            summary['songs'][song_name] = {
                'results': results,
                'statistics': calculate_statistics(results)
            }

    with open(OUTPUT_DIR / 'three_songs_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_DIR / 'three_songs_summary.json'}")

if __name__ == '__main__':
    main()
