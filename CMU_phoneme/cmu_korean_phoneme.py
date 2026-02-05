"""
Korean + English Phonetic Similarity Calculator
Usage: python phonetic_similarity.py <original.txt> <modified.txt>
"""

import re
import sys
from typing import List, Dict, Tuple
import pandas as pd

try:
    from nltk.corpus import cmudict
    CMU_AVAILABLE = True
except ImportError:
    CMU_AVAILABLE = False


class PhoneticAnalyzer:
    """Phonetic analyzer supporting both Korean (Hangul) and English."""
    
    HANGUL_START = 0xAC00
    HANGUL_END = 0xD7A3
    CHOSEONG_BASE = 588
    JUNGSEONG_BASE = 28

    CHOSEONG_MAP = {
        'ㄱ': ['K'], 'ㄲ': ['K', 'K'], 'ㄴ': ['N'], 'ㄷ': ['D'], 'ㄸ': ['D', 'D'],
        'ㄹ': ['L'], 'ㅁ': ['M'], 'ㅂ': ['P'], 'ㅃ': ['P', 'P'], 'ㅅ': ['S'],
        'ㅆ': ['S', 'S'], 'ㅇ': [], 'ㅈ': ['CH'], 'ㅉ': ['CH', 'CH'], 'ㅊ': ['CH'],
        'ㅋ': ['K'], 'ㅌ': ['T'], 'ㅍ': ['P'], 'ㅎ': ['H'],
    }

    JUNGSEONG_MAP = {
        'ㅏ': ['AH'], 'ㅐ': ['EH'], 'ㅑ': ['Y', 'AH'], 'ㅒ': ['Y', 'EH'],
        'ㅓ': ['AO'], 'ㅔ': ['EH'], 'ㅕ': ['Y', 'AO'], 'ㅖ': ['Y', 'EH'],
        'ㅗ': ['OH'], 'ㅘ': ['W', 'AH'], 'ㅙ': ['W', 'EH'], 'ㅚ': ['W', 'EH'],
        'ㅛ': ['Y', 'OH'], 'ㅜ': ['UW'], 'ㅝ': ['W', 'AO'], 'ㅞ': ['W', 'EH'],
        'ㅟ': ['W', 'IY'], 'ㅠ': ['Y', 'UW'], 'ㅡ': ['IH'], 'ㅢ': ['IH', 'IY'], 'ㅣ': ['IY'],
    }

    JONGSEONG_MAP = {
        '': [], 'ㄱ': ['K'], 'ㄲ': ['K'], 'ㄳ': ['K'], 'ㄴ': ['N'], 'ㄵ': ['N'],
        'ㄶ': ['N'], 'ㄷ': ['T'], 'ㄹ': ['L'], 'ㄺ': ['K'], 'ㄻ': ['M'], 'ㄼ': ['L'],
        'ㄽ': ['L'], 'ㄾ': ['L'], 'ㄿ': ['P'], 'ㅀ': ['L'], 'ㅁ': ['M'], 'ㅂ': ['P'],
        'ㅄ': ['P'], 'ㅅ': ['T'], 'ㅆ': ['T'], 'ㅇ': ['NG'], 'ㅈ': ['T'], 'ㅊ': ['T'],
        'ㅋ': ['K'], 'ㅌ': ['T'], 'ㅍ': ['P'], 'ㅎ': ['T'],
    }

    CHOSEONG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
                     'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSEONG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                      'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
    JONGSEONG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                      'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                      'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    ENGLISH_CHAR_MAP = {
        'a': ['AE'], 'b': ['B'], 'c': ['K'], 'd': ['D'], 'e': ['EH'],
        'f': ['F'], 'g': ['G'], 'h': ['HH'], 'i': ['IH'], 'j': ['JH'],
        'k': ['K'], 'l': ['L'], 'm': ['M'], 'n': ['N'], 'o': ['OH'],
        'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'], 't': ['T'],
        'u': ['AH'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'], 'z': ['Z'],
    }

    KOREAN_VOWELS = set(JUNGSEONG_LIST) - {''}
    ENGLISH_VOWELS = set('aeiouy')

    def __init__(self):
        self.cmu_dict = cmudict.dict() if CMU_AVAILABLE else {}

    def is_hangul(self, char: str) -> bool:
        code = ord(char)
        return self.HANGUL_START <= code <= self.HANGUL_END

    def decompose_hangul(self, char: str) -> Tuple[str, str, str]:
        code = ord(char) - self.HANGUL_START
        cho_idx = code // self.CHOSEONG_BASE
        jung_idx = (code % self.CHOSEONG_BASE) // self.JUNGSEONG_BASE
        jong_idx = code % self.JUNGSEONG_BASE
        return (self.CHOSEONG_LIST[cho_idx], 
                self.JUNGSEONG_LIST[jung_idx], 
                self.JONGSEONG_LIST[jong_idx])

    def get_phonemes(self, text: str) -> List[str]:
        phonemes = []
        words = text.split()
        
        for word in words:
            has_hangul = any(self.is_hangul(c) for c in word)
            
            if has_hangul:
                for char in word:
                    if self.is_hangul(char):
                        cho, jung, jong = self.decompose_hangul(char)
                        phonemes.extend(self.CHOSEONG_MAP.get(cho, []))
                        phonemes.extend(self.JUNGSEONG_MAP.get(jung, []))
                        if jong:
                            phonemes.extend(self.JONGSEONG_MAP.get(jong, []))
                    elif char.isalpha():
                        phonemes.extend(self.ENGLISH_CHAR_MAP.get(char.lower(), []))
            else:
                word_clean = re.sub(r'[^a-zA-Z]', '', word.lower())
                if word_clean and word_clean in self.cmu_dict:
                    cmu_phonemes = [re.sub(r'\d', '', p) for p in self.cmu_dict[word_clean][0]]
                    phonemes.extend(cmu_phonemes)
                elif word_clean:
                    for char in word_clean:
                        phonemes.extend(self.ENGLISH_CHAR_MAP.get(char, []))
        
        return phonemes

    def get_vowels(self, text: str) -> List[str]:
        vowels = []
        for char in text:
            if self.is_hangul(char):
                _, jung, _ = self.decompose_hangul(char)
                vowels.append(jung)
            elif char.lower() in self.ENGLISH_VOWELS:
                vowels.append(char.lower())
        return vowels

    def get_consonants(self, text: str) -> List[str]:
        consonants = []
        for char in text:
            if self.is_hangul(char):
                cho, _, jong = self.decompose_hangul(char)
                if cho != 'ㅇ':
                    consonants.append(cho)
                if jong:
                    consonants.append(jong)
            elif char.isalpha() and char.lower() not in self.ENGLISH_VOWELS:
                consonants.append(char.lower())
        return consonants

    def count_syllables(self, text: str) -> int:
        count = 0
        prev_vowel = False
        
        for char in text:
            if self.is_hangul(char):
                count += 1
                prev_vowel = False
            elif char.lower() in self.ENGLISH_VOWELS:
                if not prev_vowel:
                    count += 1
                prev_vowel = True
            elif char.isalpha():
                prev_vowel = False
        
        return max(1, count) if text.strip() else 0


def lcs_similarity(seq1: List[str], seq2: List[str]) -> float:
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[m][n]
    return (2 * lcs_len) / (m + n)


def jaccard_similarity(seq1: List[str], seq2: List[str]) -> float:
    set1, set2 = set(seq1), set(seq2)
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def syllable_similarity(count1: int, count2: int) -> float:
    if count1 == 0 and count2 == 0:
        return 1.0
    return 1 - abs(count1 - count2) / max(count1, count2)


def calculate_phonetic_similarity(original: str, modified: str, analyzer: PhoneticAnalyzer) -> Dict[str, float]:
    ph1 = analyzer.get_phonemes(original)
    ph2 = analyzer.get_phonemes(modified)
    
    vow1 = analyzer.get_vowels(original)
    vow2 = analyzer.get_vowels(modified)
    cons1 = analyzer.get_consonants(original)
    cons2 = analyzer.get_consonants(modified)
    
    syl1 = analyzer.count_syllables(original)
    syl2 = analyzer.count_syllables(modified)
    
    return {
        'S_ph': round(lcs_similarity(ph1, ph2), 3),
        'S_rh': round(lcs_similarity(ph1, ph2), 3),
        'S_sy': round(syllable_similarity(syl1, syl2), 3),
        'S_st': round(syllable_similarity(syl1, syl2), 3),
        'S_jac': round(jaccard_similarity(ph1, ph2), 3),
        'S_cv': round(lcs_similarity(cons1, cons2), 3),
        'S_vow': round(lcs_similarity(vow1, vow2), 3),
    }


def calculate_phi(scores: Dict[str, float]) -> float:
    components = ['S_ph', 'S_rh', 'S_sy', 'S_st', 'S_jac', 'S_cv', 'S_vow']
    values = [scores[c] for c in components if c in scores]
    return round(sum(values) / len(values), 3) if values else 0.0


def read_lyrics(filepath: str) -> List[str]:
    """Read lyrics from a text file, one line per lyric."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def format_latex_row(df: pd.DataFrame, song_name: str) -> str:
    """Convert results DataFrame to LaTeX table row."""
    avg_s_ph = df['S_ph'].mean()
    avg_s_rh = df['S_rh'].mean()
    avg_s_sy = df['S_sy'].mean()
    avg_s_st = df['S_st'].mean()
    avg_s_jac = df['S_jac'].mean()
    avg_s_cv = df['S_cv'].mean()
    avg_s_vow = df['S_vow'].mean()
    avg_phi = df['Phi'].mean()
    
    latex = f"{song_name}\n"
    latex += f"        & {avg_s_ph:.3f} & {avg_s_rh:.3f} & {avg_s_sy:.3f} & {avg_s_st:.3f} "
    latex += f"& {avg_s_jac:.3f} & {avg_s_cv:.3f} & {avg_s_vow:.3f} & {avg_phi:.3f} \\\\"
    
    return latex

def main(orig_file: str, apt_file: str, output_file: str = None):
    """Main function to compare original and APT lyrics."""
    
    print("=" * 80)
    print("Korean + English Phonetic Similarity Calculator")
    print(f"Original: {orig_file}")
    print(f"Modified: {apt_file}")
    print("=" * 80)
    
    # Read files
    original_lines = read_lyrics(orig_file)
    modified_lines = read_lyrics(apt_file)
    
    if len(original_lines) != len(modified_lines):
        print(f"WARNING: Line count mismatch! Original: {len(original_lines)}, Modified: {len(modified_lines)}")
        min_len = min(len(original_lines), len(modified_lines))
        original_lines = original_lines[:min_len]
        modified_lines = modified_lines[:min_len]
    
    analyzer = PhoneticAnalyzer()
    results = []
    
    for i, (orig, mod) in enumerate(zip(original_lines, modified_lines)):
        scores = calculate_phonetic_similarity(orig, mod, analyzer)
        phi = calculate_phi(scores)
        
        results.append({
            'line': i + 1,
            'original_line': orig,
            'modified_line': mod,
            **scores,
            'Phi': phi
        })
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(df[['line', 'S_ph', 'S_rh', 'S_sy', 'S_st', 'S_jac', 'S_cv', 'S_vow', 'Phi']].to_string(index=False))
    
    # Calculate overall Phi
    avg_phi = df['Phi'].mean()
    print(f"\n{'=' * 80}")
    print(f"OVERALL Φ: {avg_phi:.3f}")
    print(f"{'=' * 80}")
    
    # Save to CSV
    if output_file is None:
        output_file = orig_file.replace('.txt', '_phonetic_analysis.csv')
    
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    song_name = orig_file.replace('_orig.txt', '').replace('.txt', '')
    latex_row = format_latex_row(df, song_name)
    print("\n" + "=" * 80)
    print("LATEX ROW")
    print("=" * 80)
    print(latex_row)
    
    
    return df, avg_phi


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python phonetic_similarity.py <original.txt> <apt.txt> [output.csv]")
        print("Example: python phonetic_similarity.py DOPE_orig.txt DOPE_apt.txt")
        sys.exit(1)
    
    orig_file = sys.argv[1]
    apt_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    main(orig_file, apt_file, output_file)