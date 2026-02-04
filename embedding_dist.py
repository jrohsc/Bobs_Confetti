import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import torch

def load_embedding(file_path):
    """
    Load embedding from either .npy or .pt file
    Args:
        file_path: Path to embedding file (.npy or .pt)
    Returns:
        numpy array of the embedding
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.npy':
        return np.load(file_path)
    elif file_ext == '.pt':
        # Load PyTorch tensor and convert to numpy
        tensor = torch.load(file_path, map_location='cpu')
        if isinstance(tensor, torch.Tensor):
            # Convert BFloat16 to Float32 first, then to numpy
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            return tensor.detach().numpy()
        else:
            # Handle case where .pt file contains other data structures
            raise ValueError(f"Expected torch.Tensor in {file_path}, got {type(tensor)}")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .npy, .pt")

def calculate_embedding_distance(file1, file2, output_json=None):
    """
    Calculate distance between two embeddings saved as .npy or .pt files
    Args:
        file1: Path to first embedding file (.npy or .pt)
        file2: Path to second embedding file (.npy or .pt)
        output_json: Path to save JSON results (optional)
    """
    # Load embeddings (handles both .npy and .pt)
    emb1 = load_embedding(file1)
    emb2 = load_embedding(file2)
    
    print(f"Embedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")
    
    # Handle different embedding dimensions
    if emb1.shape != emb2.shape:
        print(f"Warning: Embeddings have different shapes. Using pooling to make them comparable.")
        
        # If they're 3D (batch, sequence, features), pool over sequence dimension
        if len(emb1.shape) == 3 and len(emb2.shape) == 3:
            # Mean pooling over sequence dimension
            emb1_pooled = np.mean(emb1, axis=1)  # (batch, features)
            emb2_pooled = np.mean(emb2, axis=1)  # (batch, features)
            print(f"After mean pooling - Emb1: {emb1_pooled.shape}, Emb2: {emb2_pooled.shape}")
        else:
            # Flatten and pad/truncate to same size
            emb1_flat = emb1.flatten()
            emb2_flat = emb2.flatten()
            
            min_size = min(len(emb1_flat), len(emb2_flat))
            emb1_pooled = emb1_flat[:min_size]
            emb2_pooled = emb2_flat[:min_size]
            print(f"After truncation - Both embeddings: ({min_size},)")
    else:
        # Same shape, can compare directly
        emb1_pooled = emb1
        emb2_pooled = emb2
    
    # Flatten final embeddings
    emb1_final = emb1_pooled.flatten()
    emb2_final = emb2_pooled.flatten()
    
    print(f"Final comparison - Emb1: {emb1_final.shape}, Emb2: {emb2_final.shape}")
    
    # Calculate different distance metrics
    
    # 1. Cosine Similarity (most common for embeddings)
    cos_sim = cosine_similarity([emb1_final], [emb2_final])[0][0]
    cos_dist = 1 - cos_sim
    
    # 2. Euclidean Distance
    euclidean_dist = np.linalg.norm(emb1_final - emb2_final)
    
    # 3. Manhattan Distance
    manhattan_dist = np.sum(np.abs(emb1_final - emb2_final))
    
    # 4. Dot Product
    dot_product = np.dot(emb1_final, emb2_final)
    
    # Create results dictionary
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'file1': os.path.abspath(file1),
            'file2': os.path.abspath(file2),
            'original_embedding1_shape': list(emb1.shape),
            'original_embedding2_shape': list(emb2.shape),
            'final_embedding1_shape': list(emb1_final.shape),
            'final_embedding2_shape': list(emb2_final.shape),
            'shape_mismatch': emb1.shape != emb2.shape,
            'pooling_method': 'mean_pooling' if len(emb1.shape) == 3 else 'truncation' if emb1.shape != emb2.shape else 'none'
        },
        'distance_metrics': {
            'cosine_similarity': float(cos_sim),
            'cosine_distance': float(cos_dist),
            'euclidean_distance': float(euclidean_dist),
            'manhattan_distance': float(manhattan_dist),
            'dot_product': float(dot_product)
        },
        'interpretation': {
            'similarity_level': get_similarity_interpretation(cos_sim),
            'cosine_similarity_threshold': {
                'very_similar': bool(cos_sim > 0.9),
                'moderately_similar': bool(0.7 < cos_sim <= 0.9),
                'somewhat_similar': bool(0.5 < cos_sim <= 0.7),
                'quite_different': bool(cos_sim <= 0.5)
            }
        }
    }
    
    # Print results
    print(f"\n=== Distance Metrics ===")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"Cosine Distance:   {cos_dist:.6f}")
    print(f"Euclidean Distance: {euclidean_dist:.6f}")
    print(f"Manhattan Distance: {manhattan_dist:.6f}")
    print(f"Dot Product:       {dot_product:.6f}")
    
    # Interpretation
    print(f"\n=== Interpretation ===")
    print(results['interpretation']['similarity_level'])
    
    # Save to JSON
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n=== Results saved to {output_json} ===")
    
    return results

def get_similarity_interpretation(cos_sim):
    """Get text interpretation of cosine similarity"""
    if cos_sim > 0.9:
        return "Very similar embeddings (cosine similarity > 0.9)"
    elif cos_sim > 0.7:
        return "Moderately similar embeddings (cosine similarity > 0.7)"
    elif cos_sim > 0.5:
        return "Somewhat similar embeddings (cosine similarity > 0.5)"
    else:
        return "Quite different embeddings (cosine similarity <= 0.5)"

def compare_multiple_embeddings(embedding_files, output_json="multiple_comparisons.json"):
    """
    Compare multiple embeddings and save all pairwise distances
    Args:
        embedding_files: List of embedding file paths (.npy or .pt)
        output_json: Output JSON file path
    """
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(embedding_files),
            'files': [os.path.abspath(f) for f in embedding_files]
        },
        'pairwise_comparisons': []
    }
    
    print(f"Comparing {len(embedding_files)} embeddings pairwise...")
    
    for i in range(len(embedding_files)):
        for j in range(i+1, len(embedding_files)):
            print(f"\nComparing {embedding_files[i]} vs {embedding_files[j]}")
            
            # Calculate distances without saving individual JSON
            result = calculate_embedding_distance(embedding_files[i], embedding_files[j])
            
            comparison = {
                'file1': os.path.basename(embedding_files[i]),
                'file2': os.path.basename(embedding_files[j]),
                'file1_path': os.path.abspath(embedding_files[i]),
                'file2_path': os.path.abspath(embedding_files[j]),
                'distances': result['distance_metrics'],
                'interpretation': result['interpretation']
            }
            
            all_results['pairwise_comparisons'].append(comparison)
    
    # Save all results
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== All pairwise comparisons saved to {output_json} ===")
    return all_results

# Example usage
if __name__ == "__main__":

    embedding_type = input("Enter embedding type to compare ('gpt' or 'yue'): ").strip().lower()
    song_name = input("Enter song name ('lose_yourself'): ").strip()

    output_folder = f'{embedding_type}_embedding'

    # Replace with your actual file paths (now supports both .npy and .pt)

    if embedding_type == 'gpt':
        file1 = f"{output_folder}/{song_name}_orig.npy"
        file2 = f"{output_folder}/{song_name}_phoneme.npy"
    elif embedding_type == 'yue':
        file1 = f"{output_folder}/{song_name}_orig_embeddings.pt"
        file2 = f"{output_folder}/{song_name}_phoneme_embeddings.pt"

    # Specify output JSON file (optional)
    output_file = f"embedding_dist_{embedding_type}_{song_name}.json"

    try:
        results = calculate_embedding_distance(file1, file2, output_json=output_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the embedding files exist and paths are correct")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error calculating distance: {e}")