#!/usr/bin/env python3
"""
Generate publication-quality AudioJudge heatmap visualization (Version 2).
Creates cross-comparison heatmaps similar to Figure 7 in the paper.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# Set up matplotlib for publication quality with CJK support
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Hiragino Sans GB', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False,
})


def parse_scores_from_response(response_text):
    """Extract scores from the AudioJudge response text."""
    scores = {}

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
                        category = os.path.basename(root)
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


# Song name translations for display
SONG_TRANSLATIONS = {
    '后来': 'Houlai (Later)',
    '天后': 'Tianhou (Diva)',
    '朋友': 'Pengyou (Friends)',
    '甜蜜蜜': 'Tianmimi (Sweet)',
    '光辉岁月': 'Glorious Years',
    '单车': 'Bicycle',
    '海阔天空': 'Boundless Sea',
    '红日': 'Red Sun',
}


def get_display_name(name, max_len=15):
    """Get display name with translation if available."""
    # Check for translation
    for cn, en in SONG_TRANSLATIONS.items():
        if cn in name:
            return en[:max_len] if len(en) > max_len else en

    # Truncate if needed
    if len(name) > max_len:
        return name[:max_len-3] + '...'
    return name


def create_cross_comparison_heatmap(all_results, category_name, output_path):
    """
    Create a cross-comparison heatmap like Figure 7 in the paper.
    Shows how each generated song compares to ALL original songs.
    """

    # Get all unique originals and generated for this category
    cat_results = [r for r in all_results if r['category'] == category_name]

    if not cat_results:
        print(f"No results for category: {category_name}")
        return

    originals = sorted(set(r['original'] for r in cat_results))

    # For cross-comparison, we need to generate scores for all pairs
    # First, collect what we have
    score_map = {}
    for r in cat_results:
        scores = r['scores']
        if 'melody' in scores and 'rhythm' in scores:
            overall = (scores['melody'] + scores['rhythm']) / 2
        else:
            overall = np.mean([v for v in scores.values()])
        score_map[(r['original'], r['generated'])] = overall

    n = len(originals)

    # Create matrix where rows are originals and columns are generated (matched to originals)
    matrix = np.full((n, n), np.nan)

    for i, orig_i in enumerate(originals):
        for j, orig_j in enumerate(originals):
            # Find any result where original is orig_i and generated matches orig_j
            for (o, g), score in score_map.items():
                if o == orig_i:
                    # Check if generated name matches or contains orig_j
                    if orig_j in g or g in orig_j or orig_i == orig_j:
                        if i == j:  # Diagonal - matched pair
                            matrix[i, j] = score
                            break

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Custom colormap (red -> yellow -> green)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)
    cmap.set_bad(color='white')

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                       fontsize=11, fontweight='bold', color=text_color)

    # Labels with translations
    labels = [get_display_name(o) for o in originals]

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel('Generated', fontsize=12, fontweight='bold')
    ax.set_ylabel('Original', fontsize=12, fontweight='bold')
    ax.set_title(f'{category_name}', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Overall Similarity Score', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_paper_figure7_style(all_results, output_dir):
    """
    Create a multi-panel figure similar to Figure 7 in the paper.
    Shows cross-comparison heatmaps for different categories.
    """

    # Group results by category
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    # Define category groups to display
    category_display = {
        'Phoneme': 'Phoneme',
        'English': 'English',
        'Mandarin': 'Mandarin',
        'Cantonese_YuE': 'Cantonese',
    }

    # Filter to available categories
    available = [c for c in category_display.keys() if c in by_category and len(by_category[c]) >= 2]

    if len(available) < 2:
        print("Not enough categories with sufficient data")
        return

    # Create multi-panel figure
    n_panels = min(4, len(available))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)
    cmap.set_bad(color='lightgray')

    for idx, cat in enumerate(available[:4]):
        ax = axes[idx]
        results = by_category[cat]

        # Get unique songs
        originals = sorted(set(r['original'] for r in results))
        n = len(originals)

        # Build score lookup
        score_map = {}
        for r in results:
            scores = r['scores']
            if 'melody' in scores and 'rhythm' in scores:
                overall = (scores['melody'] + scores['rhythm']) / 2
            else:
                overall = np.mean([v for v in scores.values()])
            score_map[(r['original'], r['generated'])] = overall

        # Create n x n matrix for cross-comparisons
        matrix = np.full((n, n), np.nan)

        # Fill diagonal with actual scores (original vs its generated version)
        for i, orig in enumerate(originals):
            for (o, g), score in score_map.items():
                if o == orig:
                    # This is the matched pair - put on diagonal
                    matrix[i, i] = score
                    break

        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

        # Add annotations
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < 50 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                           fontsize=10, fontweight='bold', color=text_color)

        # Labels
        labels = [get_display_name(o, 12) for o in originals]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        ax.set_xlabel('Generated', fontsize=10)
        ax.set_ylabel('Original', fontsize=10)
        ax.set_title(category_display.get(cat, cat), fontsize=12, fontweight='bold')

    # Hide unused subplots
    for idx in range(len(available), 4):
        axes[idx].axis('off')

    # Add colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Overall Similarity Score', fontsize=11)

    # Add legend
    legend_text = "Green: High Similarity (80-100)\nYellow: Moderate (40-79)\nRed: Low Similarity (0-39)\nGray: No Data"
    fig.text(0.02, 0.02, legend_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.suptitle('AudioJudge Similarity Heatmaps\n(Melody + Rhythm Average)',
                fontsize=14, fontweight='bold', y=0.98)

    output_path = os.path.join(output_dir, 'audiojudge_heatmap_figure7.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_detailed_metrics_bar_chart(all_results, output_dir):
    """
    Create a bar chart showing all 5 metrics for each song comparison.
    """

    # Group by category
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    metrics = ['melody', 'rhythm', 'harmony', 'identity', 'style']
    metric_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for cat, results in by_category.items():
        if len(results) == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 0.8)))

        labels = []
        data = {m: [] for m in metrics}

        for r in results:
            label = f"{get_display_name(r['original'], 10)} → {get_display_name(r['generated'], 10)}"
            labels.append(label)
            for m in metrics:
                data[m].append(r['scores'].get(m, 0))

        x = np.arange(len(labels))
        width = 0.15

        for i, m in enumerate(metrics):
            offset = (i - 2) * width
            bars = ax.barh(x + offset, data[m], width, label=m.capitalize(), color=metric_colors[i])

            # Add value labels
            for bar, val in zip(bars, data[m]):
                if val > 0:
                    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}',
                           va='center', fontsize=7)

        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Similarity Score', fontsize=11)
        ax.set_xlim(0, 110)
        ax.set_title(f'AudioJudge Detailed Metrics: {cat}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.axvline(x=80, color='green', linestyle='--', alpha=0.5, label='High similarity threshold')
        ax.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Moderate similarity threshold')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'metrics_bar_{cat}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")


def create_single_panel_heatmap(all_results, output_dir, category_filter=None):
    """
    Create individual heatmap panels for each category.
    """

    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)
    cmap.set_bad(color='lightgray')

    categories = [category_filter] if category_filter else by_category.keys()

    for cat in categories:
        if cat not in by_category:
            continue

        results = by_category[cat]
        if len(results) == 0:
            continue

        originals = sorted(set(r['original'] for r in results))
        n = len(originals)

        # Build score lookup
        score_map = {}
        for r in results:
            scores = r['scores']
            if 'melody' in scores and 'rhythm' in scores:
                overall = (scores['melody'] + scores['rhythm']) / 2
            else:
                overall = np.mean([v for v in scores.values()])
            score_map[(r['original'], r['generated'])] = overall

        matrix = np.full((n, n), np.nan)

        for i, orig in enumerate(originals):
            for (o, g), score in score_map.items():
                if o == orig:
                    matrix[i, i] = score
                    break

        # Create figure
        fig_size = max(6, n * 1.2)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

        # Add annotations
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < 50 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                           fontsize=12, fontweight='bold', color=text_color)

        labels = [get_display_name(o, 15) for o in originals]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)

        ax.set_xlabel('Generated', fontsize=12, fontweight='bold')
        ax.set_ylabel('Original', fontsize=12, fontweight='bold')
        ax.set_title(f'AudioJudge Similarity: {cat}', fontsize=14, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Overall Similarity\n(Melody + Rhythm avg)', fontsize=10)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'heatmap_single_{cat}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'audiojudge_results')
    output_dir = os.path.join(script_dir, 'heatmap_figures_v2')

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    all_results = load_all_results(results_dir)
    print(f"Loaded {len(all_results)} results")

    if not all_results:
        print("No results found!")
        return

    # Print summary
    by_cat = defaultdict(list)
    for r in all_results:
        by_cat[r['category']].append(r)

    print("\nResults by category:")
    for cat, results in sorted(by_cat.items()):
        print(f"  {cat}: {len(results)} comparisons")

    os.makedirs(output_dir, exist_ok=True)

    # Generate heatmaps
    print("\nGenerating Figure 7-style multi-panel heatmap...")
    create_paper_figure7_style(all_results, output_dir)

    print("\nGenerating individual category heatmaps...")
    create_single_panel_heatmap(all_results, output_dir)

    print("\nGenerating detailed metrics bar charts...")
    create_detailed_metrics_bar_chart(all_results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
