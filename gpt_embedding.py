import openai
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_text_embedding(text_file_path):
    """
    Calculate GPT embeddings for text from a .txt file using text-embedding-3-large
    
    Args:
        text_file_path (str): Path to the .txt file
    
    Returns:
        list: Embedding vector
    """
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize OpenAI client with API key from .env
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    client = OpenAI(api_key=api_key)
    
    try:
        # Read text from file
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text_content = file.read().strip()
        
        if not text_content:
            raise ValueError("The text file is empty")
        
        print(f"Processing text from: {text_file_path}")
        print(f"Text length: {len(text_content)} characters")
        
        # Get embedding using text-embedding-3-large
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text_content
        )
        
        # Extract embedding vector
        embedding = response.data[0].embedding
        
        print(f"Embedding dimension: {len(embedding)}")
        print("Embedding calculated successfully!")
        
        return embedding
        
    except FileNotFoundError:
        print(f"Error: File '{text_file_path}' not found")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def save_embedding(embedding, output_file_path):
    """
    Save embedding to a file
    
    Args:
        embedding (list): The embedding vector
        output_file_path (str): Path to save the embedding
    """
    if embedding:
        # Save as numpy array
        np.save(output_file_path, np.array(embedding))
        print(f"Embedding saved to: {output_file_path}")

def main():
    # Configuration
    song_name = 'DNA'

    song_names = [f'{song_name}_orig', f'{song_name}_phoneme']

    for song_name in song_names:
        TEXT_FILE_PATH = f"lyrics/{song_name}_lyrics.txt"  # Change this to your text file path
        OUTPUT_FILE_PATH = f"gpt_embedding/{song_name}.npy"  # Output file for embedding

        print("GPT Text Embedding Calculator")
        print("=" * 40)
        print("Loading API key from .env file...")

        # Calculate embedding
        embedding = get_text_embedding(TEXT_FILE_PATH)

        if embedding:
            # Save embedding
            save_embedding(embedding, OUTPUT_FILE_PATH)
            
            # Display first few values of the embedding
            print(f"\nFirst 10 embedding values:")
            for i, val in enumerate(embedding[:10]):
                print(f"  [{i}]: {val:.6f}")
            
            # Calculate embedding magnitude (L2 norm)
            magnitude = np.linalg.norm(embedding)
            print(f"\nEmbedding magnitude (L2 norm): {magnitude:.6f}")

if __name__ == "__main__":
    main()