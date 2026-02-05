#!/usr/bin/env python3
"""
Comprehensive AudioJudge evaluation for all songs with statistics.
Evaluates all songs in the audio folder and calculates mean ± std.
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

def calculate_statistics(scores_list, metrics=['melody', 'rhythm', 'harmony']):
    """Calculate mean, std, and 95% CI for each metric."""
    results = {}
    for metric in metrics:
        values = [s[metric] for s in scores_list if metric in s]
        if len(values) >= 2:
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
            results[metric] = {
                'mean': mean,
                'std': std,
                'ci_95_lower': ci[0],
                'ci_95_upper': ci[1],
                'n': n
            }
        elif len(values) == 1:
            results[metric] = {
                'mean': values[0],
                'std': 0,
                'ci_95_lower': values[0],
                'ci_95_upper': values[0],
                'n': 1
            }
    return results

def get_songs_to_evaluate():
    """Get all songs that have both original and generated files."""
    songs = {}

    for song_dir in AUDIO_BASE.iterdir():
        if not song_dir.is_dir():
            continue

        orig_dir = song_dir / 'original'
        gen_dir = song_dir / 'generated'

        if not orig_dir.exists() or not gen_dir.exists():
            continue

        # Get original file(s)
        originals = list(orig_dir.glob('*.[mw][pa][3v]'))
        if not originals:
            continue

        # Get generated files
        generated = list(gen_dir.glob('*.[mw][pa][3v]'))
        if not generated:
            continue

        songs[song_dir.name] = {
            'original': originals[0],  # Use first original
            'generated': generated
        }

    return songs

def main():
    print("=" * 80)
    print("COMPREHENSIVE AUDIOJUDGE EVALUATION")
    print("=" * 80)

    songs = get_songs_to_evaluate()
    print(f"\nFound {len(songs)} songs to evaluate:\n")

    for name, files in songs.items():
        print(f"  {name}: {len(files['generated'])} generated versions")

    all_results = {}
    all_stats = {}

    for song_name, files in songs.items():
        print(f"\n{'='*60}")
        print(f"Processing: {song_name}")
        print(f"{'='*60}")
        print(f"Original: {files['original'].name}")

        song_results = []
        song_output_dir = OUTPUT_DIR / song_name
        song_output_dir.mkdir(exist_ok=True)

        for gen_file in files['generated']:
            print(f"\n  Evaluating: {gen_file.name}")

            # Check if already evaluated
            result_file = song_output_dir / f"{gen_file.stem}.json"
            if result_file.exists():
                with open(result_file) as f:
                    cached = json.load(f)
                    if 'scores' in cached:
                        scores = cached['scores']
                        print(f"    (cached) Melody: {scores.get('melody', 'N/A')}, Rhythm: {scores.get('rhythm', 'N/A')}, Harmony: {scores.get('harmony', 'N/A')}")
                        song_results.append({**scores, 'file': gen_file.name})
                        continue

            # Run evaluation
            scores = run_audiojudge(files['original'], gen_file)

            if scores:
                print(f"    → Melody: {scores.get('melody', 'N/A')}, Rhythm: {scores.get('rhythm', 'N/A')}, Harmony: {scores.get('harmony', 'N/A')}")
                song_results.append({**scores, 'file': gen_file.name})

                # Save individual result
                with open(result_file, 'w') as f:
                    json.dump({
                        'original': str(files['original']),
                        'generated': str(gen_file),
                        'scores': scores
                    }, f, indent=2)
            else:
                print(f"    → Failed to evaluate")

        if song_results:
            all_results[song_name] = song_results
            all_stats[song_name] = calculate_statistics(song_results)

            # Print song statistics
            stats_info = all_stats[song_name]
            print(f"\n  Statistics for {song_name} (n={len(song_results)}):")
            for metric in ['melody', 'rhythm', 'harmony']:
                if metric in stats_info:
                    s = stats_info[metric]
                    print(f"    {metric.capitalize()}: {s['mean']:.1f} ± {s['std']:.1f}")

    # Save all results
    summary = {
        'all_results': all_results,
        'statistics': all_stats
    }

    with open(OUTPUT_DIR / 'all_songs_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    latex = """
\\begin{table}[t]
\\centering
\\small
\\caption{AudioJudge similarity scores (0-100) with standard deviation across multiple generated versions.}
\\label{tab:audiojudge-all-songs}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Song} & \\textbf{N} & \\textbf{Melody} & \\textbf{Rhythm} & \\textbf{Harmony} \\\\
\\midrule
"""

    for song_name in sorted(all_stats.keys()):
        stats_info = all_stats[song_name]
        n = stats_info.get('melody', {}).get('n', 0)

        melody = stats_info.get('melody', {})
        rhythm = stats_info.get('rhythm', {})
        harmony = stats_info.get('harmony', {})

        melody_str = f"{melody.get('mean', 0):.1f} ± {melody.get('std', 0):.1f}" if melody else "N/A"
        rhythm_str = f"{rhythm.get('mean', 0):.1f} ± {rhythm.get('std', 0):.1f}" if rhythm else "N/A"
        harmony_str = f"{harmony.get('mean', 0):.1f} ± {harmony.get('std', 0):.1f}" if harmony else "N/A"

        latex += f"{song_name} & {n} & {melody_str} & {rhythm_str} & {harmony_str} \\\\\n"
        print(f"{song_name:25} | n={n:2} | Melody: {melody_str:12} | Rhythm: {rhythm_str:12} | Harmony: {harmony_str:12}")

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(OUTPUT_DIR / 'all_songs_table.tex', 'w') as f:
        f.write(latex)

    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
