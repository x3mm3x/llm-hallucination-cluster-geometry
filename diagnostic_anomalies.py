#!/usr/bin/env python3
"""
Geometric Hallucination Taxonomy — Anomaly Diagnostics (v2)
==============================================================

Deep-dive analysis of three architectural anomaly classes:
  - ALBERT-base: factorized embedding compression (128D)
  - MiniLM-L6: distillation-induced isotropy (384D)
  - GPT-2-small: decoder-specific radial weakness (768D)
  - BERT-base: full-size encoder baseline (768D, control)

Changes from diagnostic_gpt2_anomalies.py:
  - Removed Part 1 (GPT-2 loading diagnostic) — survey handles GPT-2 now
  - Added GPT-2-small as a third anomaly class (decoder divergence)
  - β uses ALL other centroids (not one random pick)
  - Radial binning uses CONFIG['n_radial_bins'] consistently (40 bins)
  - Removed CSV merge — geometric_survey.py is the single source of truth

Produces (in ./results_anomaly_diagnostics/):
    - anomaly_analysis.txt          : detailed diagnostic report
    - fig_anomaly_deep_dive.png     : radial profiles + PCA for all anomaly classes

Usage:
    python diagnostic_anomalies.py
"""

import os
import sys
import time
import gc
import warnings
import json
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CONFIG = {
    'n_clusters': 40,
    'min_cluster_size': 10,
    'min_antonym_pairs': 2,
    'n_radial_bins': 40,
    'min_bin_count': 10,
    'random_seed': 42,
    'output_dir': './results_anomaly_diagnostics',
    'figure_dpi': 150,
}

ANTONYM_PAIRS = [
    ("good", "bad"), ("hot", "cold"), ("big", "small"),
    ("fast", "slow"), ("happy", "sad"), ("light", "dark"),
    ("old", "young"), ("rich", "poor"), ("strong", "weak"),
    ("high", "low"), ("long", "short"), ("hard", "soft"),
    ("true", "false"), ("safe", "dangerous"), ("clean", "dirty"),
    ("deep", "shallow"), ("wide", "narrow"), ("loud", "quiet"),
    ("sharp", "dull"), ("smooth", "rough"), ("thick", "thin"),
    ("sweet", "bitter"), ("alive", "dead"), ("empty", "full"),
    ("wet", "dry"), ("ancient", "modern"), ("brave", "cowardly"),
    ("cheap", "expensive"), ("beautiful", "ugly"), ("generous", "selfish"),
    ("honest", "dishonest"), ("innocent", "guilty"), ("patient", "impatient"),
    ("polite", "rude"), ("proud", "humble"), ("rare", "common"),
    ("serious", "funny"), ("straight", "crooked"), ("tight", "loose"),
    ("visible", "hidden"), ("public", "private"), ("natural", "artificial"),
    ("permanent", "temporary"), ("positive", "negative"), ("active", "passive"),
    ("major", "minor"), ("simple", "complex"), ("heavy", "light"),
    ("bright", "dim"), ("calm", "angry"), ("certain", "uncertain"),
    ("difficult", "easy"), ("early", "late"), ("fair", "unfair"),
    ("familiar", "strange"), ("foreign", "domestic"), ("formal", "informal"),
    ("frequent", "rare"), ("fresh", "stale"), ("friendly", "hostile"),
    ("global", "local"), ("huge", "tiny"), ("known", "unknown"),
    ("legal", "illegal"), ("loyal", "disloyal"), ("mature", "immature"),
    ("normal", "abnormal"), ("obvious", "subtle"), ("open", "closed"),
    ("painful", "painless"), ("partial", "complete"), ("pleasant", "unpleasant"),
    ("popular", "unpopular"), ("powerful", "powerless"), ("precise", "vague"),
    ("present", "absent"), ("real", "fake"), ("rural", "urban"),
    ("sacred", "profane"), ("sick", "healthy"), ("superior", "inferior"),
    ("tame", "wild"), ("terrible", "wonderful"), ("transparent", "opaque"),
    ("trivial", "important"), ("voluntary", "compulsory"), ("wise", "foolish"),
    ("open", "close"), ("love", "hate"), ("begin", "end"),
    ("buy", "sell"), ("give", "take"), ("rise", "fall"),
    ("win", "lose"), ("push", "pull"), ("create", "destroy"),
    ("accept", "reject"), ("attack", "defend"), ("arrive", "depart"),
    ("remember", "forget"), ("increase", "decrease"),
    ("appear", "disappear"), ("agree", "disagree"),
    ("allow", "forbid"), ("borrow", "lend"),
    ("catch", "release"), ("combine", "separate"),
    ("connect", "disconnect"), ("continue", "stop"),
    ("expand", "contract"), ("export", "import"),
    ("fail", "succeed"), ("float", "sink"),
    ("gather", "scatter"), ("grow", "shrink"),
    ("hide", "reveal"), ("hire", "fire"),
    ("include", "exclude"), ("join", "leave"),
    ("laugh", "cry"), ("lead", "follow"),
    ("live", "die"), ("lock", "unlock"),
    ("praise", "criticize"), ("raise", "lower"),
    ("reward", "punish"), ("save", "spend"),
    ("send", "receive"), ("sleep", "wake"),
    ("start", "finish"), ("strengthen", "weaken"),
    ("teach", "learn"), ("trust", "distrust"), ("unite", "divide"),
    ("success", "failure"), ("friend", "enemy"),
    ("peace", "war"), ("pleasure", "pain"),
    ("king", "queen"), ("male", "female"),
    ("morning", "evening"), ("summer", "winter"),
    ("north", "south"), ("east", "west"),
    ("black", "white"), ("left", "right"),
    ("top", "bottom"), ("front", "back"),
    ("birth", "death"), ("boy", "girl"),
    ("brother", "sister"), ("day", "night"),
    ("father", "mother"), ("husband", "wife"),
    ("joy", "sorrow"), ("land", "sea"),
    ("master", "servant"), ("maximum", "minimum"),
    ("order", "chaos"), ("profit", "loss"),
    ("son", "daughter"), ("strength", "weakness"),
    ("student", "teacher"), ("supply", "demand"),
    ("truth", "lie"), ("victory", "defeat"),
    ("wealth", "poverty"), ("wisdom", "folly"),
]



# ──────────────────────────────────────────────────────────────────────
# NOTE: Part 1 (GPT-2 diagnostic/loading) removed — handled by
# geometric_survey.py which now correctly extracts GPT-2 embeddings.
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# ANOMALY DEEP-DIVE (ALBERT, MiniLM, GPT-2, + BERT control)
# ──────────────────────────────────────────────────────────────────────


def deep_dive_anomalies():
    """Investigate architectural anomalies: ALBERT, MiniLM, GPT-2, with BERT as control."""
    print("\n" + "=" * 70)
    print("ANOMALY ANALYSIS — ALBERT, MiniLM, GPT-2")
    print("=" * 70)

    import torch
    from transformers import AutoModel, AutoTokenizer
    from wordfreq import word_frequency

    anomaly_models = [
        ('albert-base-v2', 'ALBERT-base', 128, 'Factorized embedding (128D)', 'word_embeddings'),
        ('nreimers/MiniLM-L6-H384-uncased', 'MiniLM-L6', 384, 'Distillation-induced isotropy (384D)', 'word_embeddings'),
        ('gpt2', 'GPT-2-small', 768, 'Decoder-only radial weakness (768D)', 'wte'),
        # Include BERT-base as control
        ('bert-base-uncased', 'BERT-base', 768, 'Full-size encoder baseline (768D)', 'word_embeddings'),
    ]

    analysis = {}

    for model_id, short_name, expected_dim, description, emb_method in anomaly_models:
        print(f"\n{'─' * 60}")
        print(f"  {short_name}: {description}")
        print(f"{'─' * 60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)

        with torch.no_grad():
            if emb_method == 'wte':
                # GPT-2: try model.wte first, then model.transformer.wte
                if hasattr(model, 'wte'):
                    emb = model.wte.weight.cpu().numpy()
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                    emb = model.transformer.wte.weight.cpu().numpy()
                else:
                    print(f"  FAILED: cannot find wte on {model_id}")
                    continue
            else:
                emb = model.embeddings.word_embeddings.weight.cpu().numpy()

        print(f"  Embedding shape: {emb.shape}")
        actual_dim = emb.shape[1]

        # Check for ALBERT's factorized structure
        if 'albert' in model_id:
            print(f"  ALBERT embedding projection: {actual_dim}D → {model.config.hidden_size}D")
            print(f"  This means the token embeddings live in a COMPRESSED {actual_dim}D space")
            print(f"  before being projected up to the hidden dimension.")

        # Filter vocabulary
        vocab = tokenizer.get_vocab()
        special = set(tokenizer.all_special_tokens)
        is_bpe = any(t.startswith('Ġ') for t in list(vocab.keys())[:1000])

        indices, words, freqs = [], [], []
        for token, idx in vocab.items():
            if token in special:
                continue
            if is_bpe:
                if not token.startswith('Ġ'):
                    continue
                word = token[1:].lower()
            else:
                if token.startswith('##'):
                    continue
                word = token.lower()
            if len(word) < 2 or not word.isalpha():
                continue
            freq = word_frequency(word, 'en')
            if freq > 0:
                indices.append(idx)
                words.append(word)
                freqs.append(freq)

        indices = np.array(indices)
        filtered = emb[indices]
        freqs = np.array(freqs)
        self_info = -np.log2(freqs)

        print(f"  Filtered tokens: {len(words)}")

        # ── Deep geometry analysis ──

        norms = np.linalg.norm(filtered, axis=1)
        print(f"\n  Norm statistics:")
        print(f"    Mean: {norms.mean():.4f} ± {norms.std():.4f}")
        print(f"    Range: [{norms.min():.4f}, {norms.max():.4f}]")
        print(f"    CoV (std/mean): {norms.std()/norms.mean():.4f}")

        # Effective dimensionality via PCA
        pca_full = PCA(random_state=42)
        pca_full.fit(filtered)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        dim_90 = np.searchsorted(cumvar, 0.90) + 1
        dim_95 = np.searchsorted(cumvar, 0.95) + 1
        dim_99 = np.searchsorted(cumvar, 0.99) + 1

        print(f"\n  Effective dimensionality:")
        print(f"    Nominal: {actual_dim}D")
        print(f"    90% variance: {dim_90}D")
        print(f"    95% variance: {dim_95}D")
        print(f"    99% variance: {dim_99}D")
        print(f"    Utilization (dim_95/nominal): {dim_95/actual_dim:.3f}")

        # Radial binning — use CONFIG for consistency with main survey
        n_bins = CONFIG['n_radial_bins']
        bin_edges = np.linspace(norms.min(), norms.max(), n_bins + 1)
        r_vals, I_vals, I_stds, counts = [], [], [], []
        for i in range(n_bins):
            mask = (norms >= bin_edges[i]) & (norms < bin_edges[i+1])
            if i == n_bins - 1:
                mask = (norms >= bin_edges[i]) & (norms <= bin_edges[i+1])
            if mask.sum() >= CONFIG['min_bin_count']:
                r_vals.append((bin_edges[i] + bin_edges[i+1]) / 2)
                I_vals.append(self_info[mask].mean())
                I_stds.append(self_info[mask].std())
                counts.append(mask.sum())

        r_vals = np.array(r_vals)
        I_vals = np.array(I_vals)
        I_stds = np.array(I_stds)
        counts = np.array(counts)

        # Fit degree 1, 2, 3 polynomials
        fits = {}
        ss_tot = np.sum((I_vals - I_vals.mean())**2)
        for deg in [1, 2, 3]:
            coeffs = np.polyfit(r_vals, I_vals, deg)
            pred = np.polyval(coeffs, r_vals)
            ss_res = np.sum((I_vals - pred)**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            fits[deg] = {'coeffs': coeffs, 'r2': r2, 'ss_res': ss_res}

        # F-tests: deg1 vs deg2, deg2 vs deg3
        n = len(r_vals)
        for deg_low, deg_high in [(1, 2), (2, 3)]:
            p_low = deg_low + 1
            p_high = deg_high + 1
            df1 = p_high - p_low
            df2 = n - p_high
            if df2 > 0 and fits[deg_high]['ss_res'] > 0:
                f = ((fits[deg_low]['ss_res'] - fits[deg_high]['ss_res'])/df1) / \
                    (fits[deg_high]['ss_res']/df2)
                p = 1 - stats.f.cdf(f, df1, df2)
            else:
                f, p = 0, 1.0
            fits[f'{deg_low}v{deg_high}'] = {'f_stat': f, 'p_value': p}

        print(f"\n  Polynomial fits:")
        print(f"    Degree 1: R² = {fits[1]['r2']:.4f}")
        print(f"    Degree 2: R² = {fits[2]['r2']:.4f} (Λₛ = {fits[2]['coeffs'][0]:.4f})")
        print(f"    Degree 3: R² = {fits[3]['r2']:.4f}")
        print(f"    F-test deg1 vs deg2: F={fits['1v2']['f_stat']:.2f}, p={fits['1v2']['p_value']:.6f}")
        print(f"    F-test deg2 vs deg3: F={fits['2v3']['f_stat']:.2f}, p={fits['2v3']['p_value']:.6f}")

        # Norm distribution shape analysis
        norm_skew = stats.skew(norms)
        norm_kurtosis = stats.kurtosis(norms)
        print(f"\n  Norm distribution shape:")
        print(f"    Skewness: {norm_skew:.4f}")
        print(f"    Kurtosis: {norm_kurtosis:.4f}")

        # Isotropy measure: how uniformly are directions used?
        # Compute mean cosine similarity (high = anisotropic, low = isotropic)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(filtered), min(2000, len(filtered)), replace=False)
        sample = filtered[sample_idx]
        # Normalize to unit vectors
        sample_normed = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-10)
        mean_vec = sample_normed.mean(axis=0)
        mean_cos = np.mean(sample_normed @ mean_vec)
        # Also: average pairwise cosine
        rand_pairs = 3000
        ia = rng.choice(len(sample), rand_pairs)
        ib = rng.choice(len(sample), rand_pairs)
        mask = ia != ib
        pair_cos = np.mean([np.dot(sample_normed[a], sample_normed[b])
                            for a, b in zip(ia[mask][:2000], ib[mask][:2000])])

        print(f"\n  Isotropy analysis:")
        print(f"    Mean cos to centroid direction: {mean_cos:.4f} (lower = more isotropic)")
        print(f"    Mean pairwise cosine: {pair_cos:.4f} (0 = perfectly isotropic)")

        analysis[short_name] = {
            'dim': actual_dim,
            'n_tokens': len(words),
            'norm_mean': norms.mean(), 'norm_std': norms.std(),
            'norm_cov': norms.std()/norms.mean(),
            'dim_90': dim_90, 'dim_95': dim_95, 'dim_99': dim_99,
            'utilization': dim_95/actual_dim,
            'r2_deg1': fits[1]['r2'], 'r2_deg2': fits[2]['r2'], 'r2_deg3': fits[3]['r2'],
            'lambda_s': fits[2]['coeffs'][0],
            'f_1v2': fits['1v2']['f_stat'], 'p_1v2': fits['1v2']['p_value'],
            'f_2v3': fits['2v3']['f_stat'], 'p_2v3': fits['2v3']['p_value'],
            'skewness': norm_skew, 'kurtosis': norm_kurtosis,
            'mean_cos_centroid': mean_cos, 'mean_pairwise_cos': pair_cos,
            'r_vals': r_vals, 'I_vals': I_vals, 'I_stds': I_stds,
            'fits': fits,
            'cumvar': cumvar,
            'description': description,
        }

        del model, emb, filtered, tokenizer
        gc.collect()

    return analysis


# ──────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS (same as multi_model_survey, condensed)
# ──────────────────────────────────────────────────────────────────────

def analyze_model(embeddings, words, self_info, model_name):
    """Full analysis on one model."""
    rng = np.random.RandomState(CONFIG['random_seed'])
    results = {'model': model_name, 'n_tokens': len(words),
               'embedding_dim': embeddings.shape[1]}

    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=CONFIG['n_clusters'],
                             random_state=CONFIG['random_seed'],
                             batch_size=1024, max_iter=300, n_init=5)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    sizes = np.bincount(labels)

    # β centroid — CORRECTED: all other centroids
    betas_diff = []
    betas_ratio = []
    n_c = len(centroids)
    for c in range(n_c):
        if sizes[c] < CONFIG['min_cluster_size']:
            continue
        members = np.where(labels == c)[0]
        if len(members) > 300:
            members = rng.choice(members, 300, replace=False)
        own = cosine_similarity(embeddings[members], centroids[c:c+1]).flatten().mean()
        # Similarity to ALL other centroids
        other_idx = np.arange(n_c) != c
        other_cents = centroids[other_idx]
        other_sims = cosine_similarity(embeddings[members], other_cents)
        mean_other = other_sims.mean()
        betas_diff.append(own - mean_other)
        if mean_other > 1e-6:
            betas_ratio.append((own / mean_other) - 1)

    betas_diff = np.array(betas_diff)
    betas_ratio = np.array(betas_ratio)
    if len(betas_diff) > 2:
        t, p = stats.ttest_1samp(betas_diff, 0)
        p_os = p/2 if t > 0 else 1-p/2
    else:
        t, p_os = 0, 1.0

    results['beta'] = {
        'diff_mean': float(betas_diff.mean()), 'diff_std': float(betas_diff.std()),
        'ratio_mean': float(betas_ratio.mean()) if len(betas_ratio) > 0 else 0.0,
        'n_clusters': len(betas_diff), 'p_value': float(p_os),
        'pass': bool(p_os < 0.05 and betas_diff.mean() > 0),
    }

    # α
    word_to_idx = {w: i for i, w in enumerate(words)}
    seen = set()
    valid = []
    for w1, w2 in ANTONYM_PAIRS:
        key = tuple(sorted([w1, w2]))
        if key in seen: continue
        seen.add(key)
        if w1 in word_to_idx and w2 in word_to_idx:
            valid.append((word_to_idx[w1], word_to_idx[w2]))

    cpairs = {c: [] for c in range(n_c)}
    same = 0
    for i1, i2 in valid:
        if labels[i1] == labels[i2]:
            cpairs[labels[i1]].append((i1, i2))
            same += 1

    alphas = []
    for c in range(n_c):
        pairs = cpairs[c]
        if len(pairs) < CONFIG['min_antonym_pairs'] or sizes[c] < CONFIG['min_cluster_size']:
            continue
        diffs = np.array([embeddings[i1]-embeddings[i2] for i1, i2 in pairs])
        if len(diffs) >= 2:
            pca = PCA(n_components=1); pca.fit(diffs)
            members = np.where(labels == c)[0]
            proj = embeddings[members] @ pca.components_[0]
            span = proj.max() - proj.min()
            radius = np.linalg.norm(embeddings[members]-centroids[c], axis=1).mean()
            if radius > 0:
                alphas.append(span/radius)

    alphas = np.array(alphas) if alphas else np.array([])
    results['alpha'] = {'mean': float(alphas.mean()) if len(alphas) > 0 else 0,
                         'n_clusters': len(alphas),
                         'pass': bool(len(alphas) >= 2 and alphas.mean() > 0.5)}

    # Λₛ
    norms = np.linalg.norm(embeddings, axis=1)
    edges = np.linspace(norms.min(), norms.max(), CONFIG['n_radial_bins']+1)
    rv, iv = [], []
    for i in range(CONFIG['n_radial_bins']):
        mask = (norms >= edges[i]) & (norms < edges[i+1])
        if i == CONFIG['n_radial_bins']-1:
            mask = (norms >= edges[i]) & (norms <= edges[i+1])
        if mask.sum() >= CONFIG['min_bin_count']:
            rv.append((edges[i]+edges[i+1])/2)
            iv.append(self_info[mask].mean())

    rv, iv = np.array(rv), np.array(iv)
    if len(rv) >= 5:
        c1 = np.polyfit(rv, iv, 1)
        c2 = np.polyfit(rv, iv, 2)
        pred1 = np.polyval(c1, rv)
        pred2 = np.polyval(c2, rv)
        ss1 = np.sum((iv-pred1)**2)
        ss2 = np.sum((iv-pred2)**2)
        sst = np.sum((iv-iv.mean())**2)
        r2l = 1-ss1/sst if sst > 0 else 0
        r2q = 1-ss2/sst if sst > 0 else 0

        n = len(rv)
        df2 = n-3
        if df2 > 0 and ss2 > 0:
            f = ((ss1-ss2)/1)/(ss2/df2)
            p = 1-stats.f.cdf(f, 1, df2)
        else:
            f, p = 0, 1.0

        results['lambda_s'] = {'value': float(c2[0]), 'r2_linear': float(r2l),
                                'r2_quadratic': float(r2q), 'f_stat': float(f),
                                'p_value': float(p), 'n_bins': int(n),
                                'pass': bool(p < 0.05)}
        results['radial_profile'] = {'r': rv.tolist(), 'I': iv.tolist()}
    else:
        results['lambda_s'] = {'value': 0, 'r2_linear': 0, 'r2_quadratic': 0,
                                'f_stat': 0, 'p_value': 1.0, 'n_bins': len(rv), 'pass': False}
        results['radial_profile'] = {'r': [], 'I': []}

    return results


# ──────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────

def generate_diagnostic_figures(gpt2_results, anomaly_analysis, output_dir):
    """Generate all diagnostic figures."""
    dpi = CONFIG['figure_dpi']

    # ── Figure 1: GPT-2 radial profiles ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, r) in enumerate(gpt2_results.items()):
        if r is None:
            continue
        profile = r['radial_profile']
        if profile['r']:
            ax = axes[i]
            rv = np.array(profile['r'])
            iv = np.array(profile['I'])

            ax.scatter(rv, iv, color='#2c3e50', s=20, alpha=0.7, zorder=3)

            # Fit curves
            if len(rv) >= 5:
                rs = np.linspace(rv.min(), rv.max(), 200)
                c1 = np.polyfit(rv, iv, 1)
                c2 = np.polyfit(rv, iv, 2)
                ax.plot(rs, np.polyval(c1, rs), '--', color='#e74c3c', linewidth=2,
                        label=f'Linear (R²={r["lambda_s"]["r2_linear"]:.3f})')
                ax.plot(rs, np.polyval(c2, rs), '-', color='#2980b9', linewidth=2,
                        label=f'Quadratic (R²={r["lambda_s"]["r2_quadratic"]:.3f})')

            ls = r['lambda_s']
            sig = "p < 0.001" if ls['p_value'] < 0.001 else f"p = {ls['p_value']:.4f}"
            ax.annotate(f"Λₛ = {ls['value']:.4f}\nF = {ls['f_stat']:.2f} ({sig})",
                        xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
                        va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Embedding Norm', fontsize=11)
            ax.set_ylabel('Mean Self-Information [bits]', fontsize=11)
            ax.set_title(f'{name} (Decoder)', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_gpt2_radial.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_gpt2_radial.png")

    # ── Figure 2: Anomaly deep dive ──────────────────────────────────
    if anomaly_analysis:
        models = list(anomaly_analysis.keys())
        n_models = len(models)

        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

        # Row 1: Radial profiles with multiple polynomial fits
        for i, name in enumerate(models):
            ax = axes[0, i]
            a = anomaly_analysis[name]
            rv, iv, istd = a['r_vals'], a['I_vals'], a['I_stds']

            ax.errorbar(rv, iv, yerr=istd, fmt='o', markersize=3,
                        color='#2c3e50', alpha=0.5, capsize=1)

            rs = np.linspace(rv.min(), rv.max(), 200)
            for deg, color, ls in [(1, '#e74c3c', '--'), (2, '#2980b9', '-'), (3, '#27ae60', ':')]:
                pred = np.polyval(a['fits'][deg]['coeffs'], rs)
                ax.plot(rs, pred, ls, color=color, linewidth=2,
                        label=f'Deg {deg} (R²={a["fits"][deg]["r2"]:.3f})')

            ax.set_title(f'{name} ({a["dim"]}D)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Embedding Norm', fontsize=10)
            ax.set_ylabel('Self-Info [bits]', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Row 2: PCA cumulative variance (effective dimensionality)
        for i, name in enumerate(models):
            ax = axes[1, i]
            a = anomaly_analysis[name]
            cumvar = a['cumvar'][:min(100, len(a['cumvar']))]

            ax.plot(range(1, len(cumvar)+1), cumvar, '-', color='#8e44ad', linewidth=2)
            ax.axhline(y=0.90, color='#e74c3c', linestyle='--', alpha=0.5, label='90%')
            ax.axhline(y=0.95, color='#f39c12', linestyle='--', alpha=0.5, label='95%')

            ax.axvline(x=a['dim_90'], color='#e74c3c', linestyle=':', alpha=0.3)
            ax.axvline(x=a['dim_95'], color='#f39c12', linestyle=':', alpha=0.3)

            ax.annotate(f"90%: {a['dim_90']}D\n95%: {a['dim_95']}D\n99%: {a['dim_99']}D",
                        xy=(0.55, 0.3), xycoords='axes fraction', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            ax.set_title(f'{name} — Effective Dimensionality', fontsize=11, fontweight='bold')
            ax.set_xlabel('Principal Components', fontsize=10)
            ax.set_ylabel('Cumulative Variance', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, min(100, len(cumvar)))

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'fig_anomaly_deep_dive.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print("  ✓ fig_anomaly_deep_dive.png")

    # ── Figure 3: Dimension vs Λₛ relationship ───────────────────────
    # Load previous results if available
    prev_csv = './results_survey/full_results.csv'
    if os.path.exists(prev_csv):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Parse CSV
        dims, lambdas, names_list, p_vals = [], [], [], []
        with open(prev_csv) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    names_list.append(parts[0])
                    dims.append(int(parts[1]))
                    lambdas.append(float(parts[3]))
                    p_vals.append(float(parts[4]))

        # Add GPT-2 if available
        for name, r in gpt2_results.items():
            if r is not None:
                names_list.append(name)
                dims.append(r['embedding_dim'])
                lambdas.append(r['lambda_s']['value'])
                p_vals.append(r['lambda_s']['p_value'])

        dims = np.array(dims)
        lambdas = np.array(lambdas)
        p_vals = np.array(p_vals)
        colors = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_vals]

        ax1.scatter(dims, lambdas, c=colors, s=100, alpha=0.8, edgecolors='white', linewidth=1)
        for i, name in enumerate(names_list):
            ax1.annotate(name, (dims[i], lambdas[i]), fontsize=7,
                         xytext=(5, 5), textcoords='offset points')
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_xlabel('Embedding Dimension', fontsize=11)
        ax1.set_ylabel('Λₛ', fontsize=11)
        ax1.set_title('Embedding Dimension vs Λₛ', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Isotropy comparison (if available)
        if anomaly_analysis:
            model_names = list(anomaly_analysis.keys())
            iso_vals = [anomaly_analysis[m]['mean_pairwise_cos'] for m in model_names]
            ls_vals = [anomaly_analysis[m]['lambda_s'] for m in model_names]

            ax2.scatter(iso_vals, ls_vals, c='#3498db', s=150, alpha=0.8,
                        edgecolors='white', linewidth=1)
            for i, name in enumerate(model_names):
                ax2.annotate(name, (iso_vals[i], ls_vals[i]), fontsize=9,
                             xytext=(5, 5), textcoords='offset points')
            ax2.set_xlabel('Mean Pairwise Cosine (anisotropy)', fontsize=11)
            ax2.set_ylabel('Λₛ', fontsize=11)
            ax2.set_title('Anisotropy vs Λₛ', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'fig_dimension_effect.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print("  ✓ fig_dimension_effect.png")


# ──────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────

def write_diagnostic_report(gpt2_results, anomaly_analysis, output_dir):
    """Write comprehensive diagnostic report."""
    path = os.path.join(output_dir, 'diagnostic_report.txt')
    lines = []

    lines.append("=" * 70)
    lines.append("ANOMALY DIAGNOSTICS REPORT")
    lines.append("=" * 70)

    # Anomaly analysis (primary content of this script)
    lines.append("\n" + "─" * 70)
    lines.append("ANOMALY ANALYSIS: ALBERT, MiniLM, GPT-2")
    lines.append("─" * 70)

    if anomaly_analysis:
        for name, a in anomaly_analysis.items():
            lines.append(f"\n  {name} ({a['description']}):")
            lines.append(f"    Nominal dim: {a['dim']}D")
            lines.append(f"    Effective dim (95%): {a['dim_95']}D")
            lines.append(f"    Dim utilization: {a['utilization']:.3f}")
            lines.append(f"    Norm CoV: {a['norm_cov']:.4f}")
            lines.append(f"    Skewness: {a['skewness']:.4f}")
            lines.append(f"    Mean pairwise cos: {a['mean_pairwise_cos']:.4f}")
            lines.append(f"    Λₛ = {a['lambda_s']:.4f}")
            lines.append(f"    F(1v2) = {a['f_1v2']:.2f}, p = {a['p_1v2']:.6f}")
            lines.append(f"    R² deg1={a['r2_deg1']:.3f}, deg2={a['r2_deg2']:.3f}, "
                         f"deg3={a['r2_deg3']:.3f}")

    # Interpretation
    lines.append("\n" + "─" * 70)
    lines.append("INTERPRETATION")
    lines.append("─" * 70)
    lines.append("""
  ALBERT (128D):
    ALBERT uses factorized embeddings: tokens are first mapped to a
    128D space, then projected up to 768D. This means ALL token
    relationships must be encoded in just 128 dimensions — a severe
    compression. The hypothesis: with so few dimensions, the radial
    structure gets compressed. Tokens that would spread across
    different radial shells in a 768D space get packed together in
    128D, destroying the norm-information correlation.

    Key diagnostic: If effective dimensionality is close to nominal
    dimensionality, the space is "full" — no room for the radial
    structure to breathe. If norm CoV is low, tokens cluster at
    similar distances, flattening the radial profile.

  MiniLM (384D):
    MiniLM is aggressively distilled — it learns to mimic a larger
    model's representations in half the dimensions. The distillation
    objective optimizes for preserving relative similarities, not
    absolute geometry. This can flatten the radial entropy gradient
    while preserving cluster structure (which is similarity-based).

    Key diagnostic: If pairwise cosine is high (anisotropic),
    the space is degenerate — all vectors point roughly the same
    direction, destroying radial structure while preserving angular
    (cosine-based) structure.

  Implication for the paper:
    The Λₛ non-significance in ALBERT and MiniLM is not a weakness
    of the framework — it's a FINDING. Models with compressed
    embedding spaces lose radial structure, which predicts they
    should be MORE susceptible to Type 1 (center-drift) hallucinations.
    This is a testable hypothesis.
""")

    lines.append("=" * 70)
    lines.append("END OF DIAGNOSTIC REPORT")
    lines.append("=" * 70)

    report = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(report)

    return report



# NOTE: merge_results removed — geometric_survey.py is the single
# source of truth for all primary results (Table 1, CSV).


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ANOMALY DIAGNOSTICS (v2)")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Run anomaly analysis (ALBERT, MiniLM, GPT-2, + BERT control)
    anomaly_analysis = deep_dive_anomalies()

    # Generate figures
    print(f"\n{'─' * 70}")
    print("GENERATING FIGURES")
    print(f"{'─' * 70}")
    generate_diagnostic_figures({}, anomaly_analysis, CONFIG['output_dir'])

    # Write report
    report = write_diagnostic_report({}, anomaly_analysis, CONFIG['output_dir'])

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total runtime: {time.time() - t_start:.1f}s")
    print(f"Results in: {os.path.abspath(CONFIG['output_dir'])}")
    print(f"{'=' * 70}")
    print("\n" + report)


if __name__ == '__main__':
    main()
