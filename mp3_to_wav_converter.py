#!/usr/bin/env python3
"""
MP3 to WAV Converter
Converts MP3 files to WAV format using pydub library.
"""

import os
import sys
from pathlib import Path
from pydub import AudioSegment

def convert_mp3_to_wav(input_path, output_path=None, sample_rate=44100):
    """
    Convert MP3 file to WAV format.
    
    Args:
        input_path (str): Path to the input MP3 file
        output_path (str, optional): Path for the output WAV file. 
                                   If None, creates WAV file in same directory
        sample_rate (int): Sample rate for output WAV file (default: 44100)
    
    Returns:
        str: Path to the converted WAV file
    """
    try:
        # Load the MP3 file
        print(f"Loading MP3 file: {input_path}")
        audio = AudioSegment.from_mp3(input_path)
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}.wav"
        
        # Convert to WAV format
        print(f"Converting to WAV format...")
        audio.export(output_path, format="wav", parameters=["-ar", str(sample_rate)])
        
        print(f"Successfully converted to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return None

def batch_convert_mp3_to_wav(directory_path, sample_rate=44100):
    """
    Convert all MP3 files in a directory to WAV format.
    
    Args:
        directory_path (str): Path to directory containing MP3 files
        sample_rate (int): Sample rate for output WAV files (default: 44100)
    
    Returns:
        list: List of converted WAV file paths
    """
    directory = Path(directory_path)
    mp3_files = list(directory.glob("*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {directory_path}")
        return []
    
    converted_files = []
    for mp3_file in mp3_files:
        wav_path = convert_mp3_to_wav(str(mp3_file), sample_rate=sample_rate)
        if wav_path:
            converted_files.append(wav_path)
    
    return converted_files

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mp3_to_wav_converter.py <mp3_file_path> [output_wav_path]")
        print("  python mp3_to_wav_converter.py --batch <directory_path>")
        print("\nExamples:")
        print("  python mp3_to_wav_converter.py AudioJudge/orig/单车.mp3")
        print("  python mp3_to_wav_converter.py AudioJudge/orig/单车.mp3 AudioJudge/orig/单车.wav")
        print("  python mp3_to_wav_converter.py --batch AudioJudge/")
        return
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: Please provide directory path for batch conversion")
            return
        
        directory_path = sys.argv[2]
        if not os.path.exists(directory_path):
            print(f"Error: Directory {directory_path} does not exist")
            return
        
        print(f"Batch converting MP3 files in: {directory_path}")
        converted_files = batch_convert_mp3_to_wav(directory_path)
        print(f"Converted {len(converted_files)} files")
        
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist")
            return
        
        convert_mp3_to_wav(input_path, output_path)

if __name__ == "__main__":
    main()

