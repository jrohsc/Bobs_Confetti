#!/usr/bin/env python3
"""Evaluate specific new songs that were added."""

import os
import json
import base64
from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment

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

# New files to evaluate
NEW_FILES = [
    ('Let_It_Be', 'let_it_be_v3.mp3'),
    ('We_Will_Rock_You', 'we will rock you_v3.mp3'),
    ('DOPE', 'DOPE-v2.mp3'),
    ('DOPE', 'DOPE-v3.mp3'),
    ('DOPE', 'DOPE-v4.mp3'),
]

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

def main():
    print("=" * 60)
    print("EVALUATING NEW SONGS")
    print("=" * 60)

    for song_name, gen_filename in NEW_FILES:
        print(f"\n>>> {song_name}: {gen_filename}")

        # Get paths
        song_dir = AUDIO_BASE / song_name
        orig_dir = song_dir / 'original'
        gen_dir = song_dir / 'generated'

        # Find original
        originals = list(orig_dir.glob('*.[mw][pa][3v]'))
        if not originals:
            print(f"  No original found!")
            continue
        original_path = originals[0]

        # Generated path
        generated_path = gen_dir / gen_filename
        if not generated_path.exists():
            print(f"  Generated file not found: {generated_path}")
            continue

        # Output path
        song_output_dir = OUTPUT_DIR / song_name
        song_output_dir.mkdir(exist_ok=True)
        result_file = song_output_dir / f"{Path(gen_filename).stem}.json"

        # Check if already evaluated
        if result_file.exists():
            print(f"  Already evaluated, skipping...")
            continue

        print(f"  Original: {original_path.name}")
        print(f"  Evaluating...")

        scores = run_audiojudge(original_path, generated_path)

        if scores:
            print(f"  → Melody: {scores.get('melody', 'N/A')}, Rhythm: {scores.get('rhythm', 'N/A')}, Harmony: {scores.get('harmony', 'N/A')}, Identity: {scores.get('vocalist', 'N/A')}, Style: {scores.get('style', 'N/A')}")

            with open(result_file, 'w') as f:
                json.dump({
                    'original': str(original_path),
                    'generated': str(generated_path),
                    'scores': scores
                }, f, indent=2)
        else:
            print(f"  → Failed to evaluate")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == '__main__':
    main()
