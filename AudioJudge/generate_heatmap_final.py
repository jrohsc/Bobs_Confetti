#!/usr/bin/env python3
"""
Generate publication-quality AudioJudge heatmap visualization (Final Version).
Creates clean visualizations matching the paper's Figure 7 style.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# Set up matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False,
})


def parse_scores_from_response(response_text):
    """Extract scores from the AudioJudge response text."""
    scores = {}

    # Try multiple patterns for each metric
    metric_categories = [
        ('melody', 'MELODY SIMILARITY'),
        ('rhythm', 'RHYTHMIC SIMILARITY'),
        ('harmony', 'HARMONY SIMILARITY'),
        ('identity', 'VOCALIST IDENTITY SIMILARITY'),
        ('style', 'STYLISTIC SIMILARITY'),
    ]

    for key, category in metric_categories:
        # Pattern 1: 'Score: XX.XXX' or '**Score: XX.XXX' format
        pattern1 = rf'{category}.*?Score[:\s]*(\d+\.?\d*)'
        # Pattern 2: '(XX.XXX/100' format (without Score prefix)
        pattern2 = rf'{category}[^\(]*\((\d+\.?\d*)\s*/\s*100'

        for pattern in [pattern1, pattern2]:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                val = float(match.group(1))
                scores[key] = val
                break

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

                    if scores and len(scores) >= 3:  # Need at least 3 metrics
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
    '后来': 'Houlai',
    '天后': 'Tianhou',
    '朋友': 'Pengyou',
    '甜蜜蜜': 'Tianmimi',
    '光辉岁月': 'Glorious Years',
    '单车': 'Bicycle',
    '海阔天空': 'Boundless Sea...',
    '红日': 'Red Sun',
}


def get_display_name(name, max_len=15):
    """Get display name with translation if available."""
    for cn, en in SONG_TRANSLATIONS.items():
        if cn in name:
            return en[:max_len] if len(en) > max_len else en
    if len(name) > max_len:
        return name[:max_len-2] + '..'
    return name


def create_multi_metric_heatmap(all_results, output_dir):
    """
    Create a heatmap showing all 5 metrics for each song comparison,
    grouped by category. This is the main publication figure.
    """

    # Group by category
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    # Define display order and names
    category_order = ['Phoneme', 'English', 'Mandarin', 'Cantonese_YuE', 'DNA', 'HUMBLE']
    category_display = {
        'Phoneme': 'Phoneme (APT)',
        'English': 'English (AVT)',
        'Mandarin': 'Mandarin',
        'Cantonese_YuE': 'Cantonese',
        'DNA': 'DNA (Rap)',
        'HUMBLE': 'HUMBLE (Rap)',
    }

    available_cats = [c for c in category_order if c in by_category]
    if not available_cats:
        available_cats = list(by_category.keys())

    # Create figure with subplots for each category
    n_cats = len(available_cats)
    n_cols = min(2, n_cats)
    n_rows = (n_cats + 1) // 2

    fig = plt.figure(figsize=(16, 5 * n_rows))

    metrics = ['melody', 'rhythm', 'harmony', 'identity', 'style']
    metric_labels = ['Mel', 'Rhy', 'Har', 'ID', 'Sty']

    # Custom colormap (red -> yellow -> green)
    colors = ['#c62828', '#ef5350', '#ffca28', '#9ccc65', '#43a047']
    cmap = LinearSegmentedColormap.from_list('similarity', colors, N=256)

    for cat_idx, cat in enumerate(available_cats):
        ax = fig.add_subplot(n_rows, n_cols, cat_idx + 1)
        results = by_category[cat]

        # Sort results by original name
        results = sorted(results, key=lambda r: r['original'])
        n_results = len(results)

        if n_results == 0:
            ax.axis('off')
            continue

        # Build data matrix
        matrix = np.zeros((n_results, len(metrics)))
        labels = []

        for i, r in enumerate(results):
            orig = get_display_name(r['original'], 12)
            gen = get_display_name(r['generated'], 12)
            labels.append(f"{orig}")
            for j, m in enumerate(metrics):
                matrix[i, j] = r['scores'].get(m, np.nan)

        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto')

        # Add text annotations
        for i in range(n_results):
            for j in range(len(metrics)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < 60 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                           fontsize=10, fontweight='bold', color=text_color)

        # Configure axes
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(n_results))
        ax.set_xticklabels(metric_labels, fontsize=11, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=10)

        ax.set_title(category_display.get(cat, cat), fontsize=13, fontweight='bold', pad=10)

        # Add grid
        ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_results, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

    # Add colorbar
    fig.subplots_adjust(right=0.88, hspace=0.3)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Similarity Score (0-100)', fontsize=12)
    cbar.set_ticks([0, 25, 50, 75, 100])

    # Add main title
    fig.suptitle('AudioJudge Similarity Analysis\nMelody | Rhythm | Harmony | Identity | Style',
                fontsize=16, fontweight='bold', y=0.98)

    output_path = os.path.join(output_dir, 'audiojudge_metrics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_overall_similarity_heatmap(all_results, output_dir):
    """
    Create a single heatmap showing overall similarity (melody+rhythm average)
    for all results, organized by category.
    """

    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    # Flatten and organize data
    all_comparisons = []
    category_boundaries = []
    current_idx = 0

    category_order = ['Phoneme', 'English', 'Mandarin', 'Cantonese_YuE', 'DNA', 'HUMBLE',
                      'SUNO_AVT_Cantonese', 'Gangnam Style']

    for cat in category_order:
        if cat not in by_category:
            continue
        results = sorted(by_category[cat], key=lambda r: r['original'])
        category_boundaries.append((cat, current_idx, current_idx + len(results)))
        for r in results:
            scores = r['scores']
            melody = scores.get('melody', 0)
            rhythm = scores.get('rhythm', 0)
            overall = (melody + rhythm) / 2
            all_comparisons.append({
                'category': cat,
                'label': get_display_name(r['original'], 15),
                'melody': melody,
                'rhythm': rhythm,
                'overall': overall,
            })
        current_idx += len(results)

    if not all_comparisons:
        return

    n = len(all_comparisons)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.4)))

    # Data for horizontal bar chart
    labels = [c['label'] for c in all_comparisons]
    melody_scores = [c['melody'] for c in all_comparisons]
    rhythm_scores = [c['rhythm'] for c in all_comparisons]
    overall_scores = [c['overall'] for c in all_comparisons]

    y_pos = np.arange(n)
    bar_height = 0.25

    # Plot bars
    bars1 = ax.barh(y_pos - bar_height, melody_scores, bar_height, label='Melody', color='#e53935')
    bars2 = ax.barh(y_pos, rhythm_scores, bar_height, label='Rhythm', color='#1e88e5')
    bars3 = ax.barh(y_pos + bar_height, overall_scores, bar_height, label='Overall (Avg)', color='#43a047')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.0f}',
                       va='center', fontsize=8)

    # Add category separators and labels
    for cat, start, end in category_boundaries:
        if start > 0:
            ax.axhline(y=start - 0.5, color='gray', linestyle='--', alpha=0.5)
        # Add category label
        mid = (start + end) / 2 - 0.5
        ax.text(-8, mid, cat, fontsize=10, fontweight='bold', va='center', ha='right')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_xlim(0, 110)
    ax.invert_yaxis()

    ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, linewidth=2)
    ax.axvline(x=50, color='orange', linestyle='--', alpha=0.3, linewidth=2)

    ax.legend(loc='lower right', fontsize=10)
    ax.set_title('AudioJudge Similarity Scores by Song', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'audiojudge_overall_bars.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_compact_heatmap_grid(all_results, output_dir):
    """
    Create a compact 2x2 grid of heatmaps for the main categories,
    similar to Figure 7 in the paper. Only shows existing results.
    """

    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    # Select main categories
    main_cats = ['Phoneme', 'English', 'Mandarin', 'Cantonese_YuE']
    available = [c for c in main_cats if c in by_category and len(by_category[c]) >= 2]

    if len(available) < 2:
        print("Not enough data for compact grid")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.flatten()

    colors = ['#c62828', '#ef5350', '#ffca28', '#9ccc65', '#43a047']
    cmap = LinearSegmentedColormap.from_list('sim', colors, N=256)
    cmap.set_bad(color='#e0e0e0')  # Gray for NaN

    display_names = {
        'Phoneme': 'Phoneme',
        'English': 'English',
        'Mandarin': 'Mandarin',
        'Cantonese_YuE': 'Cantonese'
    }

    for idx, cat in enumerate(available[:4]):
        ax = axes[idx]
        results = by_category[cat]

        # Filter to only results with valid melody and rhythm scores
        valid_results = []
        for r in results:
            scores = r['scores']
            if 'melody' in scores and 'rhythm' in scores:
                valid_results.append(r)

        if not valid_results:
            ax.axis('off')
            continue

        # Get unique originals from valid results only
        originals = []
        seen = set()
        for r in sorted(valid_results, key=lambda x: x['original']):
            if r['original'] not in seen:
                originals.append(r['original'])
                seen.add(r['original'])

        n = len(originals)
        matrix = np.full((n, n), np.nan)

        # Fill diagonal with overall scores from existing results
        orig_to_idx = {o: i for i, o in enumerate(originals)}
        for r in valid_results:
            i = orig_to_idx.get(r['original'])
            if i is not None:
                scores = r['scores']
                melody = scores.get('melody', 0)
                rhythm = scores.get('rhythm', 0)
                overall = (melody + rhythm) / 2
                # Only set if we have valid scores
                if melody > 0 or rhythm > 0:
                    matrix[i, i] = overall

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect='equal')

        # Add annotations only for non-NaN values
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < 60 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                           fontsize=12, fontweight='bold', color=text_color)

        # Labels with translations
        labels = [get_display_name(o, 12) for o in originals]
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        ax.set_xlabel('Generated', fontsize=10)
        ax.set_ylabel('Original', fontsize=10)
        ax.set_title(display_names.get(cat, cat), fontsize=12, fontweight='bold')

    # Hide unused
    for idx in range(len(available), 4):
        axes[idx].axis('off')

    # Colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Overall Similarity\n(Melody + Rhythm avg)', fontsize=10)

    # Legend
    legend_text = "Green: High (80-100)\nYellow: Moderate (50-79)\nRed: Low (0-49)\nGray: No data"
    fig.text(0.02, 0.02, legend_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    fig.suptitle('AudioJudge Similarity Heatmaps', fontsize=14, fontweight='bold', y=0.98)

    output_path = os.path.join(output_dir, 'audiojudge_heatmap_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table_figure(all_results, output_dir):
    """
    Create a clean summary table as a figure, similar to tables in the paper.
    """

    by_category = defaultdict(list)
    for r in all_results:
        by_category[r['category']].append(r)

    # Collect data for table
    table_data = []
    for cat in sorted(by_category.keys()):
        for r in sorted(by_category[cat], key=lambda x: x['original']):
            scores = r['scores']
            table_data.append([
                cat[:10],
                get_display_name(r['original'], 15),
                f"{scores.get('melody', 0):.0f}",
                f"{scores.get('rhythm', 0):.0f}",
                f"{scores.get('harmony', 0):.0f}",
                f"{scores.get('identity', 0):.0f}",
                f"{scores.get('style', 0):.0f}",
            ])

    if not table_data:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(4, len(table_data) * 0.35)))
    ax.axis('off')

    columns = ['Category', 'Song', 'Melody↑', 'Rhythm↑', 'Harmony↑', 'Identity↑', 'Style↑']

    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#e0e0e0']*len(columns))

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color cells based on value
    for i in range(len(table_data)):
        for j in range(2, 7):  # Score columns
            cell = table[(i+1, j)]
            try:
                val = float(table_data[i][j])
                if val >= 80:
                    cell.set_facecolor('#c8e6c9')
                elif val >= 50:
                    cell.set_facecolor('#fff9c4')
                else:
                    cell.set_facecolor('#ffcdd2')
            except:
                pass

    plt.title('AudioJudge Similarity Scores Summary', fontsize=14, fontweight='bold', pad=20)

    output_path = os.path.join(output_dir, 'audiojudge_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'audiojudge_results')
    output_dir = os.path.join(script_dir, 'heatmap_figures_final')

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

    print("\n--- Generating Figures ---")

    print("\n1. Multi-metric heatmap (main figure)...")
    create_multi_metric_heatmap(all_results, output_dir)

    print("\n2. Compact heatmap grid (Figure 7 style)...")
    create_compact_heatmap_grid(all_results, output_dir)

    print("\n3. Overall similarity bar chart...")
    create_overall_similarity_heatmap(all_results, output_dir)

    print("\n4. Summary table...")
    create_summary_table_figure(all_results, output_dir)

    print(f"\n=== All figures saved to: {output_dir} ===")


if __name__ == "__main__":
    main()
