from pydub import AudioSegment
import os
from audiojudge import AudioJudge
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

system_prompt = """
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

# Conversion function
def convert_to_wav(input_path, output_dir):
    # Get filename without extension
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base + ".wav")

    # Load and export
    sound = AudioSegment.from_file(input_path, format="mp3")
    sound.export(output_path, format="wav")
    print(f"✅ Converted {input_path} → {output_path}")


def batch_convert_mp3_tree(input_root, output_root):
    # Walk the input tree and mirror structure into output_root, converting .mp3 -> .wav
    converted_count = 0
    for current_root, _, files in os.walk(input_root):
        relative_dir = os.path.relpath(current_root, input_root)
        target_dir = output_root if relative_dir == '.' else os.path.join(output_root, relative_dir)
        os.makedirs(target_dir, exist_ok=True)

        for file_name in files:
            if not file_name.lower().endswith('.mp3'):
                continue
            input_path = os.path.join(current_root, file_name)
            convert_to_wav(input_path, target_dir)
            converted_count += 1

    print(f"🎯 Total converted: {converted_count}")


def generate_audio_judge_result(audio1_path, audio2_path):
    # Initialize with API keys from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")
    judge = AudioJudge(
        openai_api_key=api_key,
    )

    # Simple pairwise comparison
    result = judge.judge_audio(
        audio1_path=audio1_path,
        audio2_path=audio2_path,
        system_prompt=system_prompt,
        model="gpt-4o-audio-preview"
    )

    print("=" * 100)
    print("Audio Judge Result")
    print(f'audio1_path: {audio1_path}')
    print(f'audio2_path: {audio2_path}')
    print(result["response"])
    print("=" * 100)
    print("\n")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def save_judge_outputs(output_base_path, result_dict, audio1_path, audio2_path):
    # Save human-readable
    txt_path = output_base_path + ".txt"
    ensure_parent_dir(txt_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result_dict.get("response", ""))

    # Save structured JSON
    json_path = output_base_path + ".json"
    payload = {
        "audio1_path": audio1_path,
        "audio2_path": audio2_path,
        "result": result_dict,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📝 Saved results: {txt_path} and {json_path}")


def judge_pair_and_save(audio1_path, audio2_path, output_base_path):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")
    judge = AudioJudge(
        openai_api_key=api_key,
    )

    result = judge.judge_audio(
        audio1_path=audio1_path,
        audio2_path=audio2_path,
        system_prompt=system_prompt,
        model="gpt-4o-audio-preview"
    )

    # Echo to console
    print("=" * 100)
    print("Audio Judge Result (Saving)")
    print(f'audio1_path: {audio1_path}')
    print(f'audio2_path: {audio2_path}')
    print(result["response"])
    print("=" * 100)

    save_judge_outputs(output_base_path, result, audio1_path, audio2_path)


GENERATED_DIR_NAMES = {"generated", "gen"}
ORIGINAL_DIR_NAMES = {"original", "orig"}


def derive_output_base_path(output_root, audio_wav_path):
    # Remove the marker directory (generated/original) from the relative path
    # audio_wav/.../<marker>/<file>.wav -> audiojudge_results/.../<file>
    parts = audio_wav_path.split(os.sep)
    try:
        # Find the index of the marker dir by scanning known names
        idx = None
        for i, p in enumerate(parts):
            if p.lower() in GENERATED_DIR_NAMES or p.lower() in ORIGINAL_DIR_NAMES:
                idx = i
                break
        if idx is None:
            rel = os.path.splitext(os.path.relpath(audio_wav_path, start="audio_wav"))[0]
            return os.path.join(output_root, rel)
        # Build without the marker
        without_marker = parts[:idx] + parts[idx+1:]
        # Compute relative to audio_wav, drop extension
        rel_path = os.path.splitext(os.path.join(*without_marker))[0]
        # If audio_wav exists in the path, cut before it
        rel_from_wav = rel_path
        if os.sep + "audio_wav" + os.sep in os.sep + audio_wav_path:
            # Recompute using relpath for robustness
            rel_from_wav = os.path.splitext(os.path.relpath(os.path.join(*without_marker), start="audio_wav"))[0]
        return os.path.join(output_root, rel_from_wav)
    except Exception:
        # Fallback: mirror relative to audio_wav
        rel = os.path.splitext(os.path.relpath(audio_wav_path, start="audio_wav"))[0]
        return os.path.join(output_root, rel)


def batch_judge_generated_vs_original(wav_root, output_root):
    # Walk wav_root, for each generated dir, find sibling original dir and pair files by fuzzy stem match
    total = 0
    skipped = 0
    for current_root, dirs, files in os.walk(wav_root):
        base = os.path.basename(current_root).lower()
        if base not in GENERATED_DIR_NAMES:
            continue

        # Determine sibling original directory
        parent = os.path.dirname(current_root)
        sibling_original = None
        for candidate in ORIGINAL_DIR_NAMES:
            cand_path = os.path.join(parent, candidate)
            if os.path.isdir(cand_path):
                sibling_original = cand_path
                break
        if not sibling_original:
            print(f"⚠️ No matching original dir for {current_root}")
            continue

        # Index originals by stem (lowercased)
        original_index = {}
        for f in os.listdir(sibling_original):
            if f.lower().endswith('.wav'):
                original_index[os.path.splitext(f)[0].lower()] = {
                    "path": os.path.join(sibling_original, f),
                    "stem": os.path.splitext(f)[0],
                }

        for f in files:
            if not f.lower().endswith('.wav'):
                continue
            gen_stem = os.path.splitext(f)[0]
            gen_stem_lower = gen_stem.lower()
            gen_path = os.path.join(current_root, f)
            # Exact match first, then substring match (original contained in generated)
            candidate = original_index.get(gen_stem_lower)
            if not candidate:
                # Fuzzy: any original stem contained in generated stem
                fuzzy_matches = [v for k, v in original_index.items() if k in gen_stem_lower]
                if fuzzy_matches:
                    # Prefer the longest original stem (most specific)
                    candidate = max(fuzzy_matches, key=lambda v: len(v["stem"]))
            if not candidate:
                print(f"⚠️ No original match (exact or fuzzy) for generated file: {gen_path}")
                skipped += 1
                continue

            orig_path = candidate["path"]
            orig_stem = candidate["stem"]

            # Build output path: mirror directories above marker and include both stems in filename
            rel_dir_from_wav = os.path.relpath(current_root, wav_root)
            parts = rel_dir_from_wav.split(os.sep)
            try:
                idx = None
                for i, p in enumerate(parts):
                    if p.lower() in GENERATED_DIR_NAMES:
                        idx = i
                        break
                if idx is not None:
                    parts_without_marker = parts[:idx] + parts[idx+1:]
                else:
                    parts_without_marker = parts
                mirrored_dir = os.path.join(*parts_without_marker) if parts_without_marker else ''
            except Exception:
                mirrored_dir = ''

            # File base name contains both original and generated stems
            combined_name = f"{orig_stem}__{gen_stem}"
            output_base = os.path.join(output_root, mirrored_dir, combined_name)
            
            # Check if results already exist
            txt_path = output_base + ".txt"
            json_path = output_base + ".json"
            if os.path.exists(txt_path) and os.path.exists(json_path):
                print(f"⏭️ Skipping {gen_path} vs {orig_path} - results already exist")
                skipped += 1
                continue
                
            try:
                judge_pair_and_save(orig_path, gen_path, output_base)
                total += 1
            except Exception as e:
                print(f"❌ Error judging pair {orig_path} vs {gen_path}: {e}")
                skipped += 1

    print(f"🏁 Judging complete. Pairs processed: {total}, skipped: {skipped}")


def main():
    input_root = 'audio_mp3'
    output_wav_root = 'audio_wav'
    output_root = 'audiojudge_results'

    os.makedirs(output_wav_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    # Convert all MP3 files under input_root to WAV under output_wav_root (mirroring structure)
    if os.path.exists(input_root):
        batch_convert_mp3_tree(input_root, output_wav_root)
    else:
        print(f"⚠️ Input root '{input_root}' does not exist. Skipping conversion.")

    # Run AudioJudge for generated vs original WAV pairs and save outputs
    if os.path.exists(output_wav_root):
        batch_judge_generated_vs_original(output_wav_root, output_root)
    else:
        print(f"⚠️ WAV root '{output_wav_root}' does not exist. Skipping judging.")


if __name__ == "__main__":
    main()