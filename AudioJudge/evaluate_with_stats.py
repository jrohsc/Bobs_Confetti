"""
AudioJudge evaluation with statistics (mean ± std, 95% CI)

Evaluates multiple generated versions against original and calculates:
- Mean scores across all dimensions
- Standard deviation
- 95% Confidence Interval

Usage:
    python evaluate_with_stats.py
"""

import os
import json
import re
import base64
import numpy as np
from scipy import stats
from pydub import AudioSegment
from openai import OpenAI
from pathlib import Path

# Load .env file manually
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

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


def convert_to_wav(input_path, output_dir):
    """Convert audio file to WAV format."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base + ".wav")

    if os.path.exists(output_path):
        return output_path

    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.wav':
        # Just copy
        import shutil
        shutil.copy(input_path, output_path)
    else:
        sound = AudioSegment.from_file(input_path)
        sound.export(output_path, format="wav")
    print(f"✅ Converted {input_path} → {output_path}")
    return output_path


def parse_scores_from_response(response_text):
    """Extract numerical scores from AudioJudge response."""
    scores = {}

    # Patterns to match scores
    patterns = {
        'melody': r'MELODY.*?(\d+\.?\d*)\s*/\s*100',
        'rhythm': r'RHYTHM.*?(\d+\.?\d*)\s*/\s*100',
        'harmony': r'HARMONY.*?(\d+\.?\d*)\s*/\s*100',
        'vocalist': r'VOCALIST.*?(\d+\.?\d*)\s*/\s*100',
        'style': r'STYLIST.*?(\d+\.?\d*)\s*/\s*100',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            scores[key] = float(match.group(1))

    # Alternative pattern: "1. MELODY SIMILARITY (85.000/100.00)"
    alt_patterns = {
        'melody': r'1\.\s*MELODY[^(]*\((\d+\.?\d*)',
        'rhythm': r'2\.\s*RHYTHM[^(]*\((\d+\.?\d*)',
        'harmony': r'3\.\s*HARMONY[^(]*\((\d+\.?\d*)',
        'vocalist': r'4\.\s*VOCALIST[^(]*\((\d+\.?\d*)',
        'style': r'5\.\s*STYLIST[^(]*\((\d+\.?\d*)',
    }

    for key, pattern in alt_patterns.items():
        if key not in scores:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))

    return scores


def encode_audio_to_base64(audio_path):
    """Encode audio file to base64."""
    with open(audio_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def run_audiojudge(original_path, generated_path, output_json_path):
    """Run AudioJudge comparison using OpenAI API directly."""

    # Check if already evaluated
    if os.path.exists(output_json_path):
        print(f"⏭️ Loading existing: {output_json_path}")
        with open(output_json_path, 'r') as f:
            data = json.load(f)
        scores = data.get('scores')
        if not scores:
            scores = parse_scores_from_response(data.get('result', {}).get('response', ''))
        return scores

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    client = OpenAI(api_key=api_key)

    print(f"🎵 Evaluating: {os.path.basename(generated_path)}")

    # Encode audio files
    audio1_b64 = encode_audio_to_base64(original_path)
    audio2_b64 = encode_audio_to_base64(generated_path)

    # Call OpenAI API with audio
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please compare these two audio files. The first is the ORIGINAL and the second is the GENERATED version."
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio1_b64,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio2_b64,
                            "format": "wav"
                        }
                    }
                ]
            }
        ],
        modalities=["text"]
    )

    response_text = response.choices[0].message.content
    scores = parse_scores_from_response(response_text)

    # Save results
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump({
            'original': original_path,
            'generated': generated_path,
            'scores': scores,
            'result': {'response': response_text}
        }, f, indent=2)

    # Save text version
    txt_path = output_json_path.replace('.json', '.txt')
    with open(txt_path, 'w') as f:
        f.write(response_text)

    print(f"  → Melody: {scores.get('melody', 'N/A')}, Rhythm: {scores.get('rhythm', 'N/A')}, Harmony: {scores.get('harmony', 'N/A')}")

    return scores


def calculate_statistics(scores_list, metrics=['melody', 'rhythm', 'harmony']):
    """Calculate mean, std, and 95% CI for each metric."""
    results = {}

    for metric in metrics:
        values = [s.get(metric) for s in scores_list if s.get(metric) is not None]
        if not values:
            continue

        values = np.array(values)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0

        # 95% CI
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
            ci_lower, ci_upper = ci
        else:
            ci_lower, ci_upper = mean, mean

        results[metric] = {
            'mean': mean,
            'std': std,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n': n,
            'values': values.tolist()
        }

    return results


def evaluate_song_category(category_name, audio_mp3_dir, wav_dir, results_dir):
    """Evaluate all generated versions for a song category."""

    generated_dir = os.path.join(audio_mp3_dir, category_name, 'generated')
    original_dir = os.path.join(audio_mp3_dir, category_name, 'original')

    if not os.path.exists(generated_dir) or not os.path.exists(original_dir):
        print(f"⚠️ Skipping {category_name}: directories not found")
        return None

    # Find original file
    original_files = [f for f in os.listdir(original_dir) if f.endswith(('.mp3', '.wav'))]
    if not original_files:
        print(f"⚠️ No original file found for {category_name}")
        return None

    # Use first original (or the one without 'copy' in name)
    original_file = original_files[0]
    for f in original_files:
        if 'original' in f.lower() or 'copy' not in f.lower():
            original_file = f
            break

    original_path = os.path.join(original_dir, original_file)

    # Convert original to WAV
    wav_original_dir = os.path.join(wav_dir, category_name, 'original')
    os.makedirs(wav_original_dir, exist_ok=True)
    original_wav = convert_to_wav(original_path, wav_original_dir)

    # Process all generated files
    generated_files = [f for f in os.listdir(generated_dir) if f.endswith(('.mp3', '.wav'))]

    all_scores = []

    for gen_file in generated_files:
        gen_path = os.path.join(generated_dir, gen_file)

        # Convert to WAV
        wav_gen_dir = os.path.join(wav_dir, category_name, 'generated')
        os.makedirs(wav_gen_dir, exist_ok=True)
        gen_wav = convert_to_wav(gen_path, wav_gen_dir)

        # Output path
        output_json = os.path.join(results_dir, category_name, f"{os.path.splitext(gen_file)[0]}.json")

        try:
            scores = run_audiojudge(original_wav, gen_wav, output_json)
            scores['file'] = gen_file
            all_scores.append(scores)
        except Exception as e:
            print(f"❌ Error evaluating {gen_file}: {e}")

    return all_scores


def select_best_versions(scores_list, n=3, metric='melody'):
    """Select top N versions based on a metric."""
    valid = [s for s in scores_list if s.get(metric) is not None]
    sorted_scores = sorted(valid, key=lambda x: x.get(metric, 0), reverse=True)
    return sorted_scores[:n]


def format_stats_table(category_stats):
    """Format statistics as a table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Category':<20} {'Metric':<10} {'Mean':>8} {'±Std':>8} {'95% CI':>20} {'N':>5}")
    lines.append("=" * 80)

    for category, stats in category_stats.items():
        for metric, data in stats.items():
            ci_str = f"[{data['ci_95_lower']:.1f}, {data['ci_95_upper']:.1f}]"
            lines.append(f"{category:<20} {metric:<10} {data['mean']:>8.1f} {data['std']:>8.1f} {ci_str:>20} {data['n']:>5}")

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    base_dir = Path(__file__).parent
    audio_mp3_dir = base_dir / 'audio_mp3'
    wav_dir = base_dir / 'audio_wav'
    results_dir = base_dir / 'audiojudge_results_stats'

    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    categories = ['Christmas', 'DNA', 'APT']
    all_results = {}
    category_stats = {}

    print("\n" + "=" * 80)
    print("AUDIOJUDGE EVALUATION WITH STATISTICS")
    print("=" * 80 + "\n")

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Processing: {category}")
        print(f"{'='*60}")

        scores = evaluate_song_category(category, audio_mp3_dir, wav_dir, results_dir)

        if scores:
            all_results[category] = scores

            # For APT, select best 3 versions
            if category == 'APT':
                best_3 = select_best_versions(scores, n=3, metric='melody')
                print(f"\n🏆 Best 3 APT versions (by melody):")
                for i, s in enumerate(best_3, 1):
                    print(f"  {i}. {s.get('file')}: Melody={s.get('melody')}, Rhythm={s.get('rhythm')}, Harmony={s.get('harmony')}")

                # Calculate stats on best 3
                stats_best3 = calculate_statistics(best_3)
                category_stats[f'{category}_best3'] = stats_best3

            # Calculate stats on all versions
            stats_all = calculate_statistics(scores)
            category_stats[category] = stats_all

    # Print summary table
    print("\n\n")
    print(format_stats_table(category_stats))

    # Save summary
    summary_path = results_dir / 'summary_stats.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'all_results': {k: [dict(s) for s in v] for k, v in all_results.items()},
            'statistics': {k: {m: {kk: vv for kk, vv in data.items() if kk != 'values'}
                              for m, data in v.items()}
                          for k, v in category_stats.items()}
        }, f, indent=2, default=str)

    print(f"\n📊 Summary saved to: {summary_path}")

    # Generate LaTeX table
    latex_table = generate_latex_table(category_stats)
    latex_path = results_dir / 'stats_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"📝 LaTeX table saved to: {latex_path}")


def generate_latex_table(category_stats):
    """Generate LaTeX table for paper."""
    latex = r"""
\begin{table}[t]
\centering
\small
\caption{AudioJudge similarity scores (0-100) with standard deviation across multiple generated versions.}
\label{tab:audiojudge-stats}
\begin{tabular}{lcccc}
\toprule
\textbf{Song} & \textbf{N} & \textbf{Melody} & \textbf{Rhythm} & \textbf{Harmony} \\
\midrule
"""

    for category, stats in category_stats.items():
        if not stats:
            continue
        n = stats.get('melody', {}).get('n', 0)
        melody = stats.get('melody', {})
        rhythm = stats.get('rhythm', {})
        harmony = stats.get('harmony', {})

        m_str = f"{melody.get('mean', 0):.1f} ± {melody.get('std', 0):.1f}" if melody else "N/A"
        r_str = f"{rhythm.get('mean', 0):.1f} ± {rhythm.get('std', 0):.1f}" if rhythm else "N/A"
        h_str = f"{harmony.get('mean', 0):.1f} ± {harmony.get('std', 0):.1f}" if harmony else "N/A"

        latex += f"{category} & {n} & {m_str} & {r_str} & {h_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


if __name__ == "__main__":
    main()
