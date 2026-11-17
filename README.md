# Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation

This repository contains the code and data artifacts for the paper **"Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation"**.

**Demo Page:** [https://bobsconfetti.github.io/bobsconfetti/](https://bobsconfetti.github.io/bobsconfetti/)

## Abstract

Generative AI systems for music and video commonly use text-based filters to prevent the regurgitation of copyrighted material. We expose a fundamental flaw in this approach by introducing **Adversarial PhoneTic Prompting (APT)**, a novel attack that bypasses these safeguards by exploiting phonetic memorization. The APT attack replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving acoustic structure while altering meaning.

We demonstrate that leading Lyrics-to-Song (L2S) models like SUNO and YuE regenerate songs with striking melodic and rhythmic similarity to their copyrighted originals when prompted with these altered lyrics. This vulnerability extends across modalities—Text-to-Video (T2V) models like Veo 3 reconstruct visual scenes from original music videos when prompted with phonetically modified lyrics.

## Repository Structure

```
Bobs_Confetti/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── CMU_phoneme/                 # Phonetic analysis tools
│   ├── cmu_phoneme.py          # Core phonetic similarity (Phi metric)
│   ├── ablation_study.py       # Ablation study framework
│   ├── generate_paper_tables.py # Generate LaTeX tables
│   └── *.txt                   # Original and APT-modified lyrics
│
├── AudioJudge/                  # Audio similarity evaluation
│   ├── batch_audiojudge.py     # Batch evaluation using GPT-4o
│   ├── generate_heatmap_final.py # Publication-quality figures
│   ├── audio_mp3/              # Input audio files
│   ├── audio_wav/              # Converted WAV files
│   └── audiojudge_results/     # Evaluation results (JSON/TXT)
│
├── MiRA/                        # MiRA audio metrics (CLAP & CoverID)
│   ├── run_clap_coverid.py     # Batch evaluation script
│   └── MiRA results/           # CLAP and CoverID CSV results
│
├── gpt_embedding/               # GPT text embeddings (.npy)
├── yue_embedding/               # YuE model embeddings (.pt)
├── paper_lyrics/                # APT-modified lyrics used in paper
├── audio/                       # Audio samples
│
├── gpt_embedding.py            # Generate GPT embeddings
├── embedding_dist.py           # Calculate embedding distances
└── mp3_to_wav_converter.py     # Audio format conversion utility
```

## Key Components

### 1. Phonetic Similarity Metric (Phi)

The composite phonetic similarity metric captures seven sub-lexical features:

| Metric | Description |
|--------|-------------|
| S_ph   | Phoneme sequence similarity (SequenceMatcher) |
| S_rh   | Rhyme similarity (terminal phonemes) |
| S_sy   | Syllable count similarity |
| S_st   | Stress pattern alignment |
| S_jac  | Phoneme Jaccard similarity |
| S_cv   | Consonant-vowel pattern similarity |
| S_vow  | Vowel core similarity |

**Usage:**
```bash
cd CMU_phoneme
python cmu_phoneme.py original_lyrics.txt apt_lyrics.txt output.csv
```

### 2. AudioJudge Evaluation

Uses GPT-4o-audio-preview to evaluate audio similarity across five dimensions:
- Melody, Rhythm, Harmony, Vocalist Identity, Style

**Usage:**
```bash
cd AudioJudge
python batch_audiojudge.py
```

### 3. MiRA Audio Metrics (CLAP & CoverID)

[MiRA](https://github.com/roserbatlleroca/mira) (Music Information Retrieval Annotation) provides objective audio similarity metrics used in our evaluation:

- **CLAP**: Contrastive Language-Audio Pretraining score (higher = more similar)
- **CoverID**: Cover song identification distance (lower = more similar)

#### MiRA Setup

Follow the official MiRA installation guide at: https://github.com/roserbatlleroca/mira

```bash
# Clone MiRA repository
git clone https://github.com/roserbatlleroca/mira.git
cd mira

# Install MiRA (follow their README for detailed instructions)
pip install -e .
```

#### Running CLAP and CoverID

The `MiRA/run_clap_coverid.py` script automates evaluation for all songs. It expects the same audio folder structure as AudioJudge:

```
audio/
├── Song_Name/
│   ├── original/
│   │   └── original_song.mp3    # Original copyrighted audio
│   └── generated/
│       ├── generated_v1.mp3     # APT-generated versions
│       ├── generated_v2.mp3
│       └── ...
├── Another_Song/
│   ├── original/
│   └── generated/
└── ...
```

**Usage:**
```bash
cd MiRA

# Update paths in run_clap_coverid.py to match your setup:
# - AUDIO_BASE: path to your audio/ directory
# - MIRA_BASE: path to your mira installation
# - OUTPUT_DIR: where to save results

python run_clap_coverid.py
```

The script will:
1. Convert MP3 files to WAV format (required by MiRA)
2. Run CLAP similarity scoring
3. Run CoverID distance calculation
4. Save results as CSV files in the output directory

**Results** are saved in `MiRA/MiRA results/` as CSV files (e.g., `APT_clap.csv`, `APT_coverid.csv`).

### 4. Embedding Analysis

Compare text embeddings between original and APT-modified lyrics:
```bash
python embedding_dist.py gpt_embedding/DNA_orig.npy gpt_embedding/DNA_phoneme.npy
```

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)

### Setup
```bash
# Clone the repository
git clone https://github.com/[username]/Bobs_Confetti.git
cd Bobs_Confetti

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for CMU dictionary)
python -c "import nltk; nltk.download('cmudict')"
```

### API Keys
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

**Required for:**
- GPT embedding generation (`gpt_embedding.py`)
- AudioJudge evaluation (`AudioJudge/batch_audiojudge.py`)

## Reproducing Results

### Step 1: Generate APT-Modified Lyrics
The `paper_lyrics/` directory contains pre-generated APT variants. To generate new variants, use Claude or GPT with the prompt shown in the paper (Appendix, Figure 20).

### Step 2: Calculate Phonetic Similarity
```bash
cd CMU_phoneme
python cmu_phoneme.py lose_yourself_orig.txt lose_yourself_apt.txt results.csv
```

### Step 3: Generate Audio with L2S Models
1. Use SUNO (https://suno.ai) or YuE with APT-modified lyrics
2. Save generated audio to `AudioJudge/audio_mp3/[song]/generated/`
3. Place original audio in `AudioJudge/audio_mp3/[song]/original/`

### Step 4: Run AudioJudge Evaluation
```bash
cd AudioJudge
python batch_audiojudge.py
```

### Step 5: Generate Figures
```bash
cd AudioJudge
python generate_heatmap_final.py
```

## Pre-computed Results

### Embedding Distances

| Song | Model | Cosine Similarity |
|------|-------|-------------------|
| Lose Yourself | GPT | 0.746 |
| DNA | GPT | 0.725 |
| APT | GPT | 0.755 |
| Lose Yourself | YuE | 0.976 |
| DNA | YuE | 0.960 |
| APT | YuE | 0.513 |

### Phonetic Similarity (Phi)

| Song | S_ph | S_rh | S_sy | S_st | Phi_agg |
|------|------|------|------|------|---------|
| Lose Yourself | 0.817 | 0.835 | 0.936 | 0.964 | 0.874 |
| DNA | 0.791 | 0.809 | 0.859 | 0.935 | 0.835 |
| Jingle Bell Rock | 0.866 | 0.914 | 1.000 | 1.000 | 0.930 |

### Baseline Comparison (AudioJudge Scores: 0-100)

This table compares APT attacks against control baselines to validate that **phonetic structure** (not random input or semantic meaning) triggers memorization.

| Condition | Melody | Rhythm | Harmony | **Average** |
|-----------|--------|--------|---------|-------------|
| Random Lyrics | 10.0 | 18.8 | 11.7 | **13.5** |
| Semantic Paraphrase | 60.0 | 12.8 | 53.8 | **42.2** |
| **APT (Phonetic)** | **90.0** | **90.0** | **92.5** | **90.8** |

**Key Findings:**
- **Random Lyrics (Control)**: Very low scores (~13.5) confirm that arbitrary input does NOT trigger memorization
- **Semantic Paraphrase (Control)**: Mixed scores (~42.2) show that preserving meaning alone is insufficient
- **APT (Phonetic Attack)**: High scores (~90.8) demonstrate that phonetic similarity is the key trigger

| Comparison | Melody Δ | Rhythm Δ |
|------------|----------|----------|
| APT vs Random | **+80.0** | **+71.2** |
| APT vs Paraphrase | **+30.0** | **+77.2** |

## Evaluated Models

| Model | Type | Access |
|-------|------|--------|
| SUNO | L2S (Commercial) | https://suno.ai |
| YuE | L2S (Open-source) | [GitHub](https://github.com/multimodal-art-projection/YuE) |
| Veo 3 | T2V (Commercial) | Google DeepMind |

## Ethical Considerations

This research investigates vulnerabilities in commercial AI systems for responsible disclosure. We:
- Limited evaluation to well-known, widely-distributed songs
- Did not release automated attack tooling
- Did not test on independent artists' works
- Dedicated substantial effort to analyzing potential defenses

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CMU Pronouncing Dictionary for phonetic analysis
- AudioJudge framework for audio evaluation
- MiRA for audio replication detection metrics
