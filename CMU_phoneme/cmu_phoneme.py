import nltk
import re
from nltk.corpus import cmudict
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import sys

# Download required NLTK data
nltk.download('cmudict', quiet=True)

class PhoneticAnalyzer:
    def __init__(self):
        self.cmu_dict = cmudict.dict()
        
    def get_phonemes(self, word: str):
        """Get phonemes for a word using CMU dictionary."""
        word_clean = re.sub(r'[^a-zA-Z]', '', word.lower())
        if word_clean in self.cmu_dict:
            return self.cmu_dict[word_clean][0]  # First pronunciation
        return []
    
    def phoneme_similarity(self, word1: str, word2: str) -> float:
        """Phoneme-level similarity between words."""
        phonemes1 = self.get_phonemes(word1)
        phonemes2 = self.get_phonemes(word2)
        if not phonemes1 or not phonemes2:
            return 0.0
        return SequenceMatcher(None, phonemes1, phonemes2).ratio()
    
    def rhyme_similarity(self, word1: str, word2: str) -> float:
        """Compare last few phonemes for rhyme similarity."""
        phonemes1 = self.get_phonemes(word1)
        phonemes2 = self.get_phonemes(word2)
        if not phonemes1 or not phonemes2:
            return 0.0
        end1 = phonemes1[-3:] if len(phonemes1) >= 3 else phonemes1
        end2 = phonemes2[-3:] if len(phonemes2) >= 3 else phonemes2
        if end1 == end2:
            return 1.0
        vowel_match = any(p1 == p2 for p1, p2 in zip(reversed(end1), reversed(end2)))
        return 0.7 if vowel_match else 0.0
    
    def syllable_count(self, word: str) -> int:
        """Estimate syllables based on vowel phonemes."""
        phonemes = self.get_phonemes(word)
        if not phonemes:
            return 1
        return max(1, sum(1 for p in phonemes if p[-1].isdigit()))
    
    def stress_pattern(self, word: str):
        """Extract stress pattern."""
        phonemes = self.get_phonemes(word)
        return [int(p[-1]) for p in phonemes if p[-1].isdigit()] or [1]

    # --- Extra similarity metrics ---

    def phoneme_jaccard(self, word1: str, word2: str) -> float:
        """Jaccard similarity between phoneme sets of two words."""
        p1, p2 = self.get_phonemes(word1), self.get_phonemes(word2)
        if not p1 or not p2:
            return 0.0
        set1, set2 = set(p1), set(p2)
        return len(set1 & set2) / len(set1 | set2)

    def cv_pattern(self, word: str) -> str:
        """Convert phoneme sequence into consonant/vowel pattern."""
        phonemes = self.get_phonemes(word)
        return ''.join('V' if p[-1].isdigit() else 'C' for p in phonemes)

    def cv_similarity(self, word1: str, word2: str) -> float:
        """Compare consonant/vowel (CV) patterns."""
        pat1, pat2 = self.cv_pattern(word1), self.cv_pattern(word2)
        return SequenceMatcher(None, pat1, pat2).ratio() if pat1 and pat2 else 0.0

    def vowel_core_similarity(self, word1: str, word2: str) -> float:
        """Compare only vowel phonemes (ignore consonants)."""
        v1 = [p for p in self.get_phonemes(word1) if p[-1].isdigit()]
        v2 = [p for p in self.get_phonemes(word2) if p[-1].isdigit()]
        if not v1 or not v2:
            return 0.0
        return SequenceMatcher(None, v1, v2).ratio()

def calculate_line_similarity(original_line: str, modified_line: str, analyzer: PhoneticAnalyzer):
    """Calculate similarity for a pair of lines across all Φ components."""
    orig_words = re.findall(r'\b\w+\b', original_line.lower())
    mod_words = re.findall(r'\b\w+\b', modified_line.lower())
    
    # Pad shorter line
    max_len = max(len(orig_words), len(mod_words))
    orig_words.extend([''] * (max_len - len(orig_words)))
    mod_words.extend([''] * (max_len - len(mod_words)))
    
    # Collect similarities
    phoneme_sims, rhyme_sims, syllable_sims, stress_sims = [], [], [], []
    jacc_sims, cv_sims, vowel_sims = [], [], []

    for ow, mw in zip(orig_words, mod_words):
        if ow and mw:
            # S_ph
            phoneme_sims.append(analyzer.phoneme_similarity(ow, mw))
            # S_rh
            rhyme_sims.append(analyzer.rhyme_similarity(ow, mw))
            # S_sy (binary match of syllable counts)
            sc1, sc2 = analyzer.syllable_count(ow), analyzer.syllable_count(mw)
            syllable_sims.append(1.0 if sc1 == sc2 else 0.0)
            # S_st (SequenceMatcher on stress patterns)
            st1, st2 = analyzer.stress_pattern(ow), analyzer.stress_pattern(mw)
            stress_sims.append(SequenceMatcher(None, st1, st2).ratio() if st1 and st2 else 0.0)
            # Extras
            jacc_sims.append(analyzer.phoneme_jaccard(ow, mw))
            cv_sims.append(analyzer.cv_similarity(ow, mw))
            vowel_sims.append(analyzer.vowel_core_similarity(ow, mw))
        else:
            phoneme_sims.append(0.0)
            rhyme_sims.append(0.0)
            syllable_sims.append(0.0)
            stress_sims.append(0.0)
            jacc_sims.append(0.0)
            cv_sims.append(0.0)
            vowel_sims.append(0.0)
    
    return {
        "original_line": original_line.strip(),
        "modified_line": modified_line.strip(),
        "S_ph": round(np.mean(phoneme_sims), 3),
        "S_rh": round(np.mean(rhyme_sims), 3),
        "S_sy": round(np.mean(syllable_sims), 3),
        "S_st": round(np.mean(stress_sims), 3),
        "S_jac": round(np.mean(jacc_sims), 3),
        "S_cv": round(np.mean(cv_sims), 3),
        "S_vow": round(np.mean(vowel_sims), 3),
        "original_words": orig_words,
        "modified_words": mod_words,
    }

def analyze_files(original_file: str, modified_file: str, output_csv: str = None):
    """Main entrypoint: analyze two lyric files line by line."""
    with open(original_file, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
    with open(modified_file, 'r', encoding='utf-8') as f:
        modified_lines = f.readlines()
    
    max_len = max(len(original_lines), len(modified_lines))
    original_lines.extend([""] * (max_len - len(original_lines)))
    modified_lines.extend([""] * (max_len - len(modified_lines)))
    
    analyzer = PhoneticAnalyzer()
    results = []
    for o_line, m_line in zip(original_lines, modified_lines):
        results.append(calculate_line_similarity(o_line, m_line, analyzer))
    
    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python phonetic_analysis.py original.txt modified.txt [output.csv]")
        sys.exit(1)
    
    orig_file = sys.argv[1]
    mod_file = sys.argv[2]
    out_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    df = analyze_files(orig_file, mod_file, out_file)
    print(df[[
        "original_line", "modified_line",
        "S_ph", "S_rh", "S_sy", "S_st",
        "S_jac", "S_cv", "S_vow"
    ]].head(20))

    print("\nAverage S_ph (phoneme-sequence):", df["S_ph"].mean())
    print("Average S_rh (rhyme):", df["S_rh"].mean())
    print("Average S_sy (syllable count):", df["S_sy"].mean())
    print("Average S_st (stress pattern):", df["S_st"].mean())
    print("Average S_jac (phoneme Jaccard):", df["S_jac"].mean())
    print("Average S_cv (CV pattern):", df["S_cv"].mean())
    print("Average S_vow (vowel core):", df["S_vow"].mean())
