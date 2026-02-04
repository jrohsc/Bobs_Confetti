#!/usr/bin/env python3
"""
Generate publication-quality AudioJudge heatmap visualization.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Set up matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def parse_scores_from_response(response_text):
    """Extract scores from the AudioJudge response text."""
    scores = {}

    # Patterns to match scores like "MELODY SIMILARITY (85.000/100.00)" or "85.00/100.00"
    patterns = {
        'melody': r'MELODY SIMILARITY[^\d]*(\d+\.?\d*)',
        'rhythm': r'RHYTHMIC SIMILARITY[^\d]*(\d+\.?\d*)',
        'harmony': r'HARMONY SIMILARITY[^\d]*(\d+\.?\d*)',
        'identity': r'VOCALIST IDENTITY SIMILARITY[^\d]*(\d+\.?\d*)',
        'style': r'STYLISTIC SIMILARITY[^\d]*(\d+\.?\d*)',
        'overall': r'Overall Similarity Score[^\d]*(\d+\.?\d*)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            scores[key] = float(match.group(1))

    return scores


def load_all_results(results_dir):
    """Load all JSON results from the audiojudge_results directory."""
    all_results = []

    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith('.json'):
                json_path = os.path.join(root, f)
                try:
                    with open(json_path, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)

                    response = data.get('result', {}).get('response', '')
                    scores = parse_scores_from_response(response)

                    if scores:
                        # Extract category from path
                        category = os.path.basename(root)

                        # Extract original and generated names from filename
                        # Format: "original__generated.json"
                        base_name = os.path.splitext(f)[0]
                        parts = base_name.split('__')
                        if len(parts) >= 2:
                            original_name = parts[0]
                            generated_name = parts[1]
                        else:
                            original_name = base_name
                            generated_name = base_name

                        all_results.append({
                            'category': category,
                            'original': original_name,
                            'generated': generated_name,
                            'scores': scores,
                            'file': f,
                        })
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")

    return all_results


def create_category_heatmap(results, category_name, output_path, figsize=(8, 6)):
    """Create a heatmap for a specific category showing overall similarity scores."""

    # Get unique originals and generated for this category
    originals = sorted(set(r['original'] for r in results))
    generated = sorted(set(r['generated'] for r in results))

    # Create mapping for quick lookup
    score_map = {}
    for r in results:
        key = (r['original'], r['generated'])
        # Compute overall as average of melody and rhythm (primary metrics)
        scores = r['scores']
        if 'melody' in scores and 'rhythm' in scores:
            overall = (scores['melody'] + scores['rhythm']) / 2
        elif 'overall' in scores:
            overall = scores['overall']
        else:
            overall = np.mean([v for v in scores.values()])
        score_map[key] = overall

    # Build matrix
    n_orig = len(originals)
    n_gen = len(generated)
    matrix = np.zeros((n_orig, n_gen))
    matrix[:] = np.nan  # Initialize with NaN for missing values

    for i, orig in enumerate(originals):
        for j, gen in enumerate(generated):
            key = (orig, gen)
            if key in score_map:
                matrix[i, j] = score_map[key]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom colormap (red -> yellow -> green)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Overall Similarity Score', fontsize=10)

    # Add text annotations
    for i in range(n_orig):
        for j in range(n_gen):
            val = matrix[i, j]
            if not np.isnan(val):
                # Choose text color based on background
                text_color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color)

    # Clean up labels - truncate long names
    def truncate_label(label, max_len=20):
        if len(label) > max_len:
            return label[:max_len-3] + '...'
        return label

    x_labels = [truncate_label(g) for g in generated]
    y_labels = [truncate_label(o) for o in originals]

    ax.set_xticks(np.arange(n_gen))
    ax.set_yticks(np.arange(n_orig))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Generated', fontsize=11)
    ax.set_ylabel('Original', fontsize=11)
    ax.set_title(f'AudioJudge Similarity: {category_name}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_combined_heatmap(all_results, output_path, categories_to_include=None):
    """Create a combined heatmap across multiple categories with diagonal highlighting."""

    if categories_to_include:
        results = [r for r in all_results if r['category'] in categories_to_include]
    else:
        results = all_results

    if not results:
        print("No results to plot")
        return

    # Group by category for organization
    by_category = defaultdict(list)
    for r in results:
        by_category[r['category']].append(r)

    # Build combined list of all unique songs
    all_songs = []
    category_ranges = {}

    for cat in sorted(by_category.keys()):
        cat_results = by_category[cat]
        originals = sorted(set(r['original'] for r in cat_results))
        start_idx = len(all_songs)
        all_songs.extend([(cat, orig) for orig in originals])
        category_ranges[cat] = (start_idx, len(all_songs))

    n = len(all_songs)
    matrix = np.full((n, n), np.nan)

    # Fill matrix
    song_to_idx = {song: i for i, song in enumerate(all_songs)}

    for r in results:
        orig_key = (r['category'], r['original'])
        if orig_key in song_to_idx:
            i = song_to_idx[orig_key]
            # For generated, try to find matching original
            gen_key = (r['category'], r['generated'])
            if gen_key in song_to_idx:
                j = song_to_idx[gen_key]
            else:
                # Try fuzzy match - generated name might contain original name
                for key, idx in song_to_idx.items():
                    if key[0] == r['category'] and key[1] in r['generated']:
                        j = idx
                        break
                else:
                    j = i  # Self-comparison

            scores = r['scores']
            if 'melody' in scores and 'rhythm' in scores:
                overall = (scores['melody'] + scores['rhythm']) / 2
            elif 'overall' in scores:
                overall = scores['overall']
            else:
                overall = np.mean([v for v in scores.values()])

            matrix[i, j] = overall

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)
    cmap.set_bad(color='white')

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Overall Similarity Score', fontsize=11)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=7, fontweight='bold', color=text_color)

    # Labels
    labels = [f"{s[1]}" for s in all_songs]
    labels = [l[:15] + '...' if len(l) > 15 else l for l in labels]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Add category separators
    for cat, (start, end) in category_ranges.items():
        # Draw rectangle around category block
        rect = mpatches.Rectangle((start - 0.5, start - 0.5), end - start, end - start,
                                   linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlabel('Generated', fontsize=12)
    ax.set_ylabel('Original', fontsize=12)
    ax.set_title('AudioJudge Similarity Heatmap\n(Melody + Rhythm Average)',
                fontsize=14, fontweight='bold')

    # Add legend for categories
    legend_handles = []
    for i, cat in enumerate(sorted(category_ranges.keys())):
        color = plt.cm.tab10(i / len(category_ranges))
        legend_handles.append(mpatches.Patch(color=color, label=cat))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_detailed_multi_metric_heatmap(results, category_name, output_path):
    """Create a detailed heatmap showing all 5 metrics for each comparison."""

    metrics = ['melody', 'rhythm', 'harmony', 'identity', 'style']
    metric_labels = ['Melody', 'Rhythm', 'Harmony', 'Identity', 'Style']

    n_results = len(results)
    if n_results == 0:
        return

    # Prepare data matrix
    matrix = np.zeros((n_results, len(metrics)))
    labels = []

    for i, r in enumerate(results):
        labels.append(f"{r['original'][:12]} vs {r['generated'][:12]}")
        for j, m in enumerate(metrics):
            matrix[i, j] = r['scores'].get(m, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(4, n_results * 0.5)))

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Similarity Score', fontsize=10)

    # Add text annotations
    for i in range(n_results):
        for j in range(len(metrics)):
            val = matrix[i, j]
            text_color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=text_color)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(n_results))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Comparison', fontsize=11)
    ax.set_title(f'AudioJudge Multi-Metric Analysis: {category_name}',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_paper_style_heatmap(all_results, output_dir):
    """Create heatmaps in the style of Figure 7 from the paper."""

    os.makedirs(output_dir, exist_ok=True)

    # Group results by category
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    # Create category mapping for paper-style presentation
    category_groups = {
        'Phoneme': 'Phoneme',
        'English': 'English',
        'Mandarin': 'Mandarin',
        'Cantonese_YuE': 'Cantonese',
        'SUNO_AVT_Cantonese': 'Cantonese (SUNO AVT)',
        'DNA': 'Rap (DNA)',
        'HUMBLE': 'Rap (HUMBLE)',
        'Gangnam Style': 'Korean (Gangnam Style)',
    }

    # Create individual category heatmaps
    for cat, results in by_category.items():
        if len(results) > 0:
            display_name = category_groups.get(cat, cat)
            output_path = os.path.join(output_dir, f'heatmap_{cat}.png')
            create_detailed_multi_metric_heatmap(results, display_name, output_path)

    # Create combined summary heatmap
    create_summary_heatmap(all_results, os.path.join(output_dir, 'heatmap_summary.png'))


def create_summary_heatmap(all_results, output_path):
    """Create a summary heatmap like Figure 7 showing pairwise similarities."""

    # Group by category
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    # Create subplots for each major category group
    categories_to_plot = ['Phoneme', 'English', 'Mandarin', 'Cantonese_YuE']
    available_cats = [c for c in categories_to_plot if c in by_category]

    if not available_cats:
        available_cats = list(by_category.keys())[:4]

    n_cats = len(available_cats)
    if n_cats == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)

    for idx, (cat, ax) in enumerate(zip(available_cats, axes)):
        results = by_category[cat]

        # Get unique songs
        originals = sorted(set(r['original'] for r in results))

        # Create matrix - use melody+rhythm average as overall score
        n = len(originals)
        matrix = np.full((n, n), np.nan)

        orig_to_idx = {o: i for i, o in enumerate(originals)}

        for r in results:
            i = orig_to_idx.get(r['original'])
            # Try to find matching index for generated
            j = None
            for orig, idx_j in orig_to_idx.items():
                if orig in r['generated'] or r['generated'] in orig:
                    j = idx_j
                    break
            if j is None:
                j = i  # Self comparison

            if i is not None and j is not None:
                scores = r['scores']
                if 'melody' in scores and 'rhythm' in scores:
                    overall = (scores['melody'] + scores['rhythm']) / 2
                else:
                    overall = np.mean([v for v in scores.values()])
                matrix[i, j] = overall

        # Plot
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

        # Annotations
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < 50 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                           fontsize=9, fontweight='bold', color=text_color)

        # Labels
        labels = [o[:12] + '...' if len(o) > 12 else o for o in originals]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        ax.set_xlabel('Generated', fontsize=10)
        ax.set_ylabel('Original', fontsize=10)
        ax.set_title(cat, fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(available_cats), 4):
        axes[idx].axis('off')

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Overall Similarity Score', fontsize=11)

    # Add legend
    legend_text = "Green: High Similarity (80-100)\nYellow: Moderate (40-79)\nRed: Low Similarity (0-39)"
    fig.text(0.02, 0.02, legend_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('AudioJudge Similarity Heatmaps\n(Melody + Rhythm Average)',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Find the results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'audiojudge_results')
    output_dir = os.path.join(script_dir, 'heatmap_figures')

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    all_results = load_all_results(results_dir)
    print(f"Loaded {len(all_results)} results")

    if not all_results:
        print("No results found!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate heatmaps
    create_paper_style_heatmap(all_results, output_dir)

    print(f"\nAll heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()
