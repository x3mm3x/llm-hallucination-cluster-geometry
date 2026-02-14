#!/usr/bin/env python3
"""
Geometric Hallucination Taxonomy — Geometric Survey (v2)
=========================================================

Corrected survey script. Changes from multi_model_survey.py:
    - β uses ALL other centroids (not one random pick)
    - Reports β_diff (primary) and β_ratio (secondary)
    - GPT-2 extraction handles both model.wte and model.transformer.wte
    - CSV includes β_diff, β_diff_p, β_ratio columns

Hardware: CPU only, 16GB RAM
Runtime: ~60-90 min for all models (downloads ~8GB on first run)

Usage:
    python geometric_survey.py

Outputs (in ./results_geometric_survey/):
    - cross_model_comparison.txt    : summary table
    - fig_lambda_comparison.png     : Λₛ across models
    - fig_cross_model_summary.png   : α, β, Λₛ comparative
    - full_results.csv              : machine-readable results
    - full_results.json             : detailed JSON results
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

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

CONFIG = {
    'n_clusters': 40,
    'min_cluster_size': 10,
    'min_antonym_pairs': 2,
    'n_radial_bins': 40,
    'min_bin_count': 10,
    'random_seed': 42,
    'output_dir': './results_geometric_survey',
    'figure_dpi': 150,
}

# Models that fit in 16GB RAM on CPU
# Each entry: (huggingface_id, short_name, embedding_extraction_method)
# Methods:
#   'word_embeddings'  — standard transformer word embedding layer
#   'wte'              — GPT-2 style (model.transformer.wte)

MODELS = [
    # BERT family
    ('bert-base-uncased',           'BERT-base',        'word_embeddings'),
    ('bert-large-uncased',          'BERT-large',       'word_embeddings'),
    ('distilbert-base-uncased',     'DistilBERT',       'word_embeddings'),

    # RoBERTa family
    ('roberta-base',                'RoBERTa-base',     'roberta_embeddings'),
    ('roberta-large',               'RoBERTa-large',    'roberta_embeddings'),

    # ALBERT (shared embeddings — interesting test case)
    ('albert-base-v2',              'ALBERT-base',      'albert_embeddings'),

    # Electra
    ('google/electra-base-discriminator', 'ELECTRA-base', 'word_embeddings'),

    # GPT-2 family (decoder-only)
    ('gpt2',                        'GPT-2-small',      'wte'),
    ('gpt2-medium',                 'GPT-2-medium',     'wte'),

    # DeBERTa
    ('microsoft/deberta-base',      'DeBERTa-base',     'word_embeddings'),

    # MiniLM (distilled, small)
    ('nreimers/MiniLM-L6-H384-uncased', 'MiniLM-L6',   'word_embeddings'),

    # Phi-2 — uncomment if you want to try, needs ~6GB RAM
    # WARNING: slow on CPU, ~10 min just for loading
    # ('microsoft/phi-2',           'Phi-2',            'wte'),
]

# ──────────────────────────────────────────────────────────────────────
# ANTONYM PAIRS (same expanded set from v2)
# ──────────────────────────────────────────────────────────────────────

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
    ("clear", "cloudy"), ("comfortable", "uncomfortable"),
    ("correct", "incorrect"), ("cruel", "kind"), ("dense", "sparse"),
    ("difficult", "easy"), ("early", "late"), ("even", "odd"),
    ("fair", "unfair"), ("familiar", "strange"), ("fat", "thin"),
    ("fertile", "barren"), ("fierce", "gentle"), ("flat", "steep"),
    ("foreign", "domestic"), ("formal", "informal"), ("free", "bound"),
    ("frequent", "rare"), ("fresh", "stale"), ("friendly", "hostile"),
    ("glad", "sorry"), ("global", "local"), ("grateful", "ungrateful"),
    ("guilty", "innocent"), ("hollow", "solid"), ("huge", "tiny"),
    ("identical", "different"), ("indoor", "outdoor"), ("junior", "senior"),
    ("known", "unknown"), ("lazy", "busy"), ("legal", "illegal"),
    ("literal", "figurative"), ("logical", "irrational"),
    ("loyal", "disloyal"), ("mature", "immature"), ("mild", "severe"),
    ("mobile", "stationary"), ("moist", "dry"), ("moral", "immoral"),
    ("native", "foreign"), ("neat", "messy"), ("noisy", "silent"),
    ("normal", "abnormal"), ("obvious", "subtle"), ("official", "unofficial"),
    ("open", "closed"), ("oral", "written"), ("ordinary", "extraordinary"),
    ("original", "copy"), ("painful", "painless"), ("partial", "complete"),
    ("peculiar", "ordinary"), ("plain", "fancy"), ("pleasant", "unpleasant"),
    ("plentiful", "scarce"), ("popular", "unpopular"),
    ("powerful", "powerless"), ("precise", "vague"),
    ("present", "absent"), ("raw", "cooked"), ("real", "fake"),
    ("rigid", "flexible"), ("rough", "smooth"), ("rural", "urban"),
    ("sacred", "profane"), ("sane", "insane"), ("secret", "public"),
    ("separate", "together"), ("severe", "mild"), ("sharp", "blunt"),
    ("shy", "bold"), ("sick", "healthy"), ("singular", "plural"),
    ("slim", "fat"), ("sober", "drunk"), ("sour", "sweet"),
    ("specific", "general"), ("steady", "unsteady"),
    ("stiff", "flexible"), ("stupid", "intelligent"),
    ("superior", "inferior"), ("tame", "wild"), ("temporary", "permanent"),
    ("tender", "tough"), ("terrible", "wonderful"), ("thick", "thin"),
    ("timid", "bold"), ("together", "apart"), ("transparent", "opaque"),
    ("trivial", "important"), ("vacant", "occupied"), ("vast", "tiny"),
    ("vertical", "horizontal"), ("voluntary", "compulsory"),
    ("warm", "cool"), ("weak", "strong"), ("wealthy", "poor"),
    ("wicked", "virtuous"), ("wise", "foolish"),
    ("open", "close"), ("love", "hate"), ("begin", "end"),
    ("buy", "sell"), ("give", "take"), ("rise", "fall"),
    ("win", "lose"), ("push", "pull"), ("create", "destroy"),
    ("accept", "reject"), ("attack", "defend"), ("arrive", "depart"),
    ("remember", "forget"), ("increase", "decrease"),
    ("appear", "disappear"), ("agree", "disagree"),
    ("allow", "forbid"), ("borrow", "lend"), ("build", "demolish"),
    ("catch", "release"), ("collect", "distribute"),
    ("combine", "separate"), ("connect", "disconnect"),
    ("continue", "stop"), ("demand", "supply"),
    ("encourage", "discourage"), ("enter", "exit"),
    ("expand", "contract"), ("export", "import"),
    ("extend", "shorten"), ("fail", "succeed"),
    ("float", "sink"), ("gain", "lose"),
    ("gather", "scatter"), ("grow", "shrink"),
    ("hide", "reveal"), ("hire", "fire"),
    ("include", "exclude"), ("join", "leave"),
    ("laugh", "cry"), ("lead", "follow"),
    ("lend", "borrow"), ("lift", "drop"),
    ("live", "die"), ("lock", "unlock"),
    ("marry", "divorce"), ("obey", "disobey"),
    ("praise", "criticize"), ("protect", "endanger"),
    ("raise", "lower"), ("reward", "punish"),
    ("save", "spend"), ("scatter", "gather"),
    ("send", "receive"), ("simplify", "complicate"),
    ("sleep", "wake"), ("spread", "contain"),
    ("start", "finish"), ("strengthen", "weaken"),
    ("stretch", "compress"), ("teach", "learn"),
    ("trust", "distrust"), ("unite", "divide"),
    ("success", "failure"), ("friend", "enemy"),
    ("peace", "war"), ("pleasure", "pain"),
    ("question", "answer"), ("king", "queen"),
    ("heaven", "hell"), ("male", "female"),
    ("morning", "evening"), ("summer", "winter"),
    ("north", "south"), ("east", "west"),
    ("black", "white"), ("left", "right"),
    ("top", "bottom"), ("front", "back"),
    ("inside", "outside"), ("above", "below"),
    ("birth", "death"), ("boy", "girl"),
    ("brother", "sister"), ("ceiling", "floor"),
    ("city", "country"), ("comedy", "tragedy"),
    ("danger", "safety"), ("day", "night"),
    ("earth", "sky"), ("entrance", "exit"),
    ("father", "mother"), ("husband", "wife"),
    ("joy", "sorrow"), ("land", "sea"),
    ("master", "servant"), ("maximum", "minimum"),
    ("order", "chaos"), ("profit", "loss"),
    ("reward", "punishment"), ("silence", "noise"),
    ("son", "daughter"), ("strength", "weakness"),
    ("student", "teacher"), ("supply", "demand"),
    ("truth", "lie"), ("victory", "defeat"),
    ("wealth", "poverty"), ("wisdom", "folly"),
]


# ──────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def extract_embeddings(model_id, method):
    """Extract embedding matrix from a model. Returns (embeddings, tokenizer) or None on failure."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        model.eval()

        with torch.no_grad():
            if method == 'word_embeddings':
                emb = model.embeddings.word_embeddings.weight.cpu().numpy()
            elif method == 'wte':
                # GPT-2: AutoModel returns GPT2Model where embeddings are at model.wte
                # (model.transformer.wte only exists on GPT2LMHeadModel)
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                    emb = model.transformer.wte.weight.cpu().numpy()
                elif hasattr(model, 'wte'):
                    emb = model.wte.weight.cpu().numpy()
                else:
                    print(f"    FAILED: cannot find wte on {model_id}")
                    return None, None
            elif method == 'roberta_embeddings':
                emb = model.embeddings.word_embeddings.weight.cpu().numpy()
            elif method == 'albert_embeddings':
                emb = model.embeddings.word_embeddings.weight.cpu().numpy()
            else:
                print(f"    Unknown method: {method}")
                return None, None

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return emb, tokenizer

    except Exception as e:
        print(f"    FAILED to load {model_id}: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────────────
# FILTERING — handles both BERT-style and BPE tokenizers
# ──────────────────────────────────────────────────────────────────────

def filter_vocabulary(embeddings, tokenizer, model_id):
    """Filter to meaningful tokens. Adapts to tokenizer type."""
    from wordfreq import word_frequency

    vocab = tokenizer.get_vocab()
    special_tokens = set(tokenizer.all_special_tokens)

    # Detect tokenizer type
    is_bpe = any(token.startswith('Ġ') for token in list(vocab.keys())[:1000])
    is_bert = any(token.startswith('##') for token in list(vocab.keys())[:1000])

    filtered_indices = []
    words = []
    frequencies = []

    for token, idx in vocab.items():
        # Skip special tokens
        if token in special_tokens:
            continue

        # Normalize token to word
        if is_bpe:
            # GPT-2 / RoBERTa style: Ġword = word-initial token
            if not token.startswith('Ġ'):
                continue
            word = token[1:].lower()
        elif is_bert:
            # BERT style: skip ##subword tokens
            if token.startswith('##'):
                continue
            word = token.lower()
        else:
            word = token.lower()

        # Quality filters
        if len(word) < 2:
            continue
        if not word.isalpha():
            continue

        freq = word_frequency(word, 'en')
        if freq > 0:
            filtered_indices.append(idx)
            words.append(word)
            frequencies.append(freq)

    filtered_indices = np.array(filtered_indices)
    filtered_embeddings = embeddings[filtered_indices]
    frequencies = np.array(frequencies)
    self_info = -np.log2(frequencies)

    return filtered_embeddings, words, self_info


# ──────────────────────────────────────────────────────────────────────
# CORE ANALYSIS (condensed from v2)
# ──────────────────────────────────────────────────────────────────────

def analyze_model(embeddings, words, self_info, model_name):
    """Run full analysis on one model. Returns results dict."""
    rng = np.random.RandomState(CONFIG['random_seed'])
    results = {'model': model_name, 'n_tokens': len(words),
               'embedding_dim': embeddings.shape[1]}

    # ── Clustering ──
    kmeans = MiniBatchKMeans(
        n_clusters=CONFIG['n_clusters'], random_state=CONFIG['random_seed'],
        batch_size=1024, max_iter=300, n_init=5)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    sizes = np.bincount(labels)
    results['cluster_sizes'] = {'min': int(sizes.min()), 'max': int(sizes.max()),
                                'mean': float(sizes.mean())}

    # ── β (centroid method — CORRECTED: all other centroids) ──
    betas_diff = []
    betas_ratio = []
    all_own_sims = []
    all_other_sims = []
    n_clusters = len(centroids)
    for c in range(n_clusters):
        if sizes[c] < CONFIG['min_cluster_size']:
            continue
        members = np.where(labels == c)[0]
        if len(members) > 300:
            members = rng.choice(members, 300, replace=False)

        # Similarity to own centroid
        own_sims = cosine_similarity(embeddings[members], centroids[c:c+1]).flatten()
        mean_own = own_sims.mean()
        all_own_sims.extend(own_sims.tolist())

        # Similarity to ALL other centroids (not one random pick)
        other_idx = np.arange(n_clusters) != c
        other_cents = centroids[other_idx]  # shape: (n_clusters-1, dim)
        other_sims = cosine_similarity(embeddings[members], other_cents)  # (n_members, n_clusters-1)
        mean_other = other_sims.mean()  # grand mean
        all_other_sims.extend(other_sims.mean(axis=1).tolist())  # per-member mean

        betas_diff.append(mean_own - mean_other)
        if mean_other > 1e-6:
            betas_ratio.append((mean_own / mean_other) - 1)

    betas_diff = np.array(betas_diff)
    betas_ratio = np.array(betas_ratio)
    if len(betas_diff) > 2:
        t_stat, p_val = stats.ttest_1samp(betas_diff, 0)
        p_onesided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    else:
        t_stat, p_onesided = 0, 1.0

    results['beta'] = {
        'diff_mean': float(betas_diff.mean()), 'diff_std': float(betas_diff.std()),
        'diff_median': float(np.median(betas_diff)),
        'ratio_mean': float(betas_ratio.mean()) if len(betas_ratio) > 0 else 0.0,
        'ratio_std': float(betas_ratio.std()) if len(betas_ratio) > 0 else 0.0,
        'pct_positive': float((betas_diff > 0).mean() * 100),
        'n_clusters': len(betas_diff),
        't_stat': float(t_stat), 'p_value': float(p_onesided),
        'pass': bool(p_onesided < 0.05 and betas_diff.mean() > 0),
        'mean_own_centroid_sim': float(np.mean(all_own_sims)),
        'mean_other_centroid_sim': float(np.mean(all_other_sims)),
    }

    # ── α (polarity coupling) ──
    word_to_idx = {w: i for i, w in enumerate(words)}
    seen = set()
    valid_pairs = []
    for w1, w2 in ANTONYM_PAIRS:
        key = tuple(sorted([w1, w2]))
        if key in seen:
            continue
        seen.add(key)
        # Check both original and lowercase
        w1l, w2l = w1.lower(), w2.lower()
        if w1l in word_to_idx and w2l in word_to_idx:
            valid_pairs.append((w1l, w2l, word_to_idx[w1l], word_to_idx[w2l]))

    cluster_pairs = {c: [] for c in range(n_clusters)}
    same_cluster = 0
    for w1, w2, i1, i2 in valid_pairs:
        c1, c2 = labels[i1], labels[i2]
        if c1 == c2:
            cluster_pairs[c1].append((i1, i2))
            same_cluster += 1

    alphas = []
    for c in range(n_clusters):
        pairs = cluster_pairs[c]
        if len(pairs) < CONFIG['min_antonym_pairs'] or sizes[c] < CONFIG['min_cluster_size']:
            continue

        diffs = np.array([embeddings[i1] - embeddings[i2] for i1, i2 in pairs])
        if len(diffs) >= 2:
            pca = PCA(n_components=1)
            pca.fit(diffs)
            axis = pca.components_[0]

            members = np.where(labels == c)[0]
            projections = embeddings[members] @ axis
            span = projections.max() - projections.min()
            radius = np.linalg.norm(embeddings[members] - centroids[c], axis=1).mean()

            if radius > 0:
                alphas.append(span / radius)

    alphas = np.array(alphas) if alphas else np.array([])
    results['alpha'] = {
        'mean': float(alphas.mean()) if len(alphas) > 0 else 0,
        'std': float(alphas.std()) if len(alphas) > 0 else 0,
        'n_clusters': len(alphas),
        'n_valid_pairs': len(valid_pairs),
        'n_same_cluster': same_cluster,
        'pass': bool(len(alphas) >= 2 and (alphas.mean() if len(alphas) > 0 else 0) > 0.5),
    }

    # ── Λₛ (radial entropy gradient) ──
    norms = np.linalg.norm(embeddings, axis=1)
    bin_edges = np.linspace(norms.min(), norms.max(), CONFIG['n_radial_bins'] + 1)

    r_vals, I_vals = [], []
    for i in range(CONFIG['n_radial_bins']):
        mask = (norms >= bin_edges[i]) & (norms < bin_edges[i + 1])
        if i == CONFIG['n_radial_bins'] - 1:
            mask = (norms >= bin_edges[i]) & (norms <= bin_edges[i + 1])
        if mask.sum() >= CONFIG['min_bin_count']:
            r_vals.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            I_vals.append(self_info[mask].mean())

    r_vals = np.array(r_vals)
    I_vals = np.array(I_vals)

    if len(r_vals) >= 5:
        coeffs_1 = np.polyfit(r_vals, I_vals, 1)
        pred_1 = np.polyval(coeffs_1, r_vals)
        ss_res_1 = np.sum((I_vals - pred_1) ** 2)
        ss_tot = np.sum((I_vals - I_vals.mean()) ** 2)
        r2_lin = 1 - ss_res_1 / ss_tot if ss_tot > 0 else 0

        coeffs_2 = np.polyfit(r_vals, I_vals, 2)
        pred_2 = np.polyval(coeffs_2, r_vals)
        ss_res_2 = np.sum((I_vals - pred_2) ** 2)
        r2_quad = 1 - ss_res_2 / ss_tot if ss_tot > 0 else 0

        lambda_s = coeffs_2[0]

        n = len(r_vals)
        df1, df2 = 1, n - 3
        if df2 > 0 and ss_res_2 > 0:
            f_stat = ((ss_res_1 - ss_res_2) / df1) / (ss_res_2 / df2)
            p_val = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            f_stat, p_val = 0, 1.0

        results['lambda_s'] = {
            'value': float(lambda_s),
            'r2_linear': float(r2_lin),
            'r2_quadratic': float(r2_quad),
            'f_stat': float(f_stat),
            'p_value': float(p_val),
            'n_bins': int(n),
            'pass': bool(p_val < 0.05),
        }
        results['radial_profile'] = {'r': r_vals.tolist(), 'I': I_vals.tolist()}
    else:
        results['lambda_s'] = {'value': 0, 'r2_linear': 0, 'r2_quadratic': 0,
                                'f_stat': 0, 'p_value': 1.0, 'n_bins': len(r_vals),
                                'pass': False}
        results['radial_profile'] = {'r': [], 'I': []}

    # ── Embedding stats ──
    results['norm_stats'] = {
        'mean': float(norms.mean()), 'std': float(norms.std()),
        'min': float(norms.min()), 'max': float(norms.max()),
    }

    return results


# ──────────────────────────────────────────────────────────────────────
# CROSS-MODEL FIGURES
# ──────────────────────────────────────────────────────────────────────

def generate_cross_model_figures(all_results, output_dir):
    """Generate comparative figures across models."""
    dpi = CONFIG['figure_dpi']

    # Filter to successful runs
    results = [r for r in all_results if r is not None]
    if not results:
        print("  No successful results to plot.")
        return

    names = [r['model'] for r in results]
    n = len(names)

    # ── Figure 1: Λₛ across models ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    lambda_vals = [r['lambda_s']['value'] for r in results]
    r2_quads = [r['lambda_s']['r2_quadratic'] for r in results]
    p_vals = [r['lambda_s']['p_value'] for r in results]
    colors = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_vals]

    bars = ax1.barh(range(n), lambda_vals, color=colors, alpha=0.8, edgecolor='white')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('Λₛ (Quadratic Coefficient)', fontsize=11)
    ax1.set_title('Λₛ Across Models (green=significant, red=not)', fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='x')

    # R² comparison
    x = np.arange(n)
    width = 0.35
    ax2.bar(x - width/2, [r['lambda_s']['r2_linear'] for r in results],
            width, label='R² Linear', color='#e74c3c', alpha=0.7)
    ax2.bar(x + width/2, r2_quads,
            width, label='R² Quadratic', color='#2980b9', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('R²', fontsize=11)
    ax2.set_title('Linear vs Quadratic Fit Quality', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_lambda_comparison.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_lambda_comparison.png")

    # ── Figure 2: Cross-model summary (α, β, Λₛ) ────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    # β (diff — primary metric)
    beta_means = [r['beta']['diff_mean'] for r in results]
    beta_pass = [r['beta']['pass'] for r in results]
    colors_b = ['#2ecc71' if p else '#e74c3c' for p in beta_pass]
    ax1.barh(range(n), beta_means, color=colors_b, alpha=0.8, edgecolor='white')
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('β_diff (Own − Other Centroid Similarity)', fontsize=11)
    ax1.set_title('β_diff Across Models', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='x')

    # α
    alpha_means = [r['alpha']['mean'] for r in results]
    alpha_n = [r['alpha']['n_clusters'] for r in results]
    alpha_pass = [r['alpha']['pass'] for r in results]
    colors_a = ['#2ecc71' if p else '#e74c3c' for p in alpha_pass]
    bars_a = ax2.barh(range(n), alpha_means, color=colors_a, alpha=0.8, edgecolor='white')
    # Annotate with cluster count
    for i, (val, nc) in enumerate(zip(alpha_means, alpha_n)):
        ax2.text(val + 0.02, i, f'n={nc}', va='center', fontsize=8, color='#555')
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(names, fontsize=10)
    ax2.set_xlabel('α (Polarity Coupling)', fontsize=11)
    ax2.set_title('α Across Models', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Dims and token counts
    dims = [r['embedding_dim'] for r in results]
    tokens = [r['n_tokens'] for r in results]
    ax3.scatter(dims, tokens, s=100, c='#3498db', alpha=0.7, edgecolors='white', linewidth=1)
    for i, name in enumerate(names):
        ax3.annotate(name, (dims[i], tokens[i]), fontsize=8,
                     xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Embedding Dimension', fontsize=11)
    ax3.set_ylabel('Filtered Tokens', fontsize=11)
    ax3.set_title('Model Characteristics', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_cross_model_summary.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_cross_model_summary.png")

    # ── Figure 3: Overlaid radial profiles ────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab10
    for i, r in enumerate(results):
        profile = r['radial_profile']
        if profile['r']:
            # Normalize r to [0,1] for comparison across different norm ranges
            r_arr = np.array(profile['r'])
            I_arr = np.array(profile['I'])
            r_norm = (r_arr - r_arr.min()) / (r_arr.max() - r_arr.min() + 1e-10)
            ax.plot(r_norm, I_arr, '-o', markersize=2, color=cmap(i % 10),
                    alpha=0.7, linewidth=1.5, label=r['model'])

    ax.set_xlabel('Normalized Embedding Norm (0=min, 1=max)', fontsize=11)
    ax.set_ylabel('Mean Self-Information I(τ) [bits]', fontsize=11)
    ax.set_title('Radial Entropy Profiles Across Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_radial_profiles.png'),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig_radial_profiles.png")


# ──────────────────────────────────────────────────────────────────────
# REPORT
# ──────────────────────────────────────────────────────────────────────

def write_cross_model_report(all_results, output_dir):
    """Write comparative summary."""
    results = [r for r in all_results if r is not None]
    path = os.path.join(output_dir, 'cross_model_comparison.txt')

    lines = []
    lines.append("=" * 90)
    lines.append("GEOMETRIC HALLUCINATION TAXONOMY — MULTI-MODEL SURVEY")
    lines.append("=" * 90)
    lines.append(f"\nModels analyzed: {len(results)}")
    lines.append(f"Clusters per model: {CONFIG['n_clusters']}")
    lines.append(f"Random seed: {CONFIG['random_seed']}")

    # Summary table
    lines.append("\n" + "─" * 100)
    lines.append(f"{'Model':<20} {'Dim':>5} {'Tokens':>7} {'Λₛ':>10} {'Λₛ p':>10} "
                 f"{'β_diff':>8} {'β_ratio':>8} {'β pass':>7} {'α mean':>8} {'α n':>5}")
    lines.append("─" * 100)

    for r in results:
        ls = r['lambda_s']
        b = r['beta']
        a = r['alpha']
        lines.append(
            f"{r['model']:<20} {r['embedding_dim']:>5} {r['n_tokens']:>7} "
            f"{ls['value']:>10.4f} {ls['p_value']:>10.6f} "
            f"{b['diff_mean']:>8.4f} {b['ratio_mean']:>8.4f} {'✓' if b['pass'] else '✗':>7} "
            f"{a['mean']:>8.4f} {a['n_clusters']:>5}"
        )

    lines.append("─" * 100)

    # Universality check
    lines.append("\n" + "─" * 90)
    lines.append("UNIVERSALITY CHECK")
    lines.append("─" * 90)

    n_lambda_pass = sum(1 for r in results if r['lambda_s']['pass'])
    n_beta_pass = sum(1 for r in results if r['beta']['pass'])
    n_alpha_pass = sum(1 for r in results if r['alpha']['pass'])

    lines.append(f"  Λₛ significant (p<0.05): {n_lambda_pass}/{len(results)}")
    lines.append(f"  β significant (p<0.05):  {n_beta_pass}/{len(results)}")
    lines.append(f"  α > 0.5 with 2+ clusters: {n_alpha_pass}/{len(results)}")

    if n_lambda_pass == len(results):
        lines.append("\n  Λₛ: UNIVERSAL ✓ — nonlinear radial structure found in ALL models")
    else:
        lines.append(f"\n  Λₛ: PARTIAL — found in {n_lambda_pass}/{len(results)} models")

    if n_beta_pass >= len(results) * 0.8:
        lines.append(f"  β:  STRONG — cluster cohesion confirmed in {n_beta_pass}/{len(results)} models")
    else:
        lines.append(f"  β:  MODERATE — confirmed in {n_beta_pass}/{len(results)} models")

    if n_alpha_pass >= len(results) * 0.5:
        lines.append(f"  α:  CONFIRMED — polarity structure in {n_alpha_pass}/{len(results)} models")
    else:
        lines.append(f"  α:  LIMITED — only {n_alpha_pass}/{len(results)} models")

    # Architecture comparison
    lines.append("\n" + "─" * 90)
    lines.append("ENCODER vs DECODER COMPARISON")
    lines.append("─" * 90)

    encoders = [r for r in results if r['model'] not in ('GPT-2-small', 'GPT-2-medium', 'Phi-2')]
    decoders = [r for r in results if r['model'] in ('GPT-2-small', 'GPT-2-medium', 'Phi-2')]

    if encoders:
        enc_lambda = np.mean([r['lambda_s']['value'] for r in encoders])
        enc_beta = np.mean([r['beta']['diff_mean'] for r in encoders])
        lines.append(f"  Encoders (n={len(encoders)}): mean Λₛ = {enc_lambda:.4f}, mean β_diff = {enc_beta:.4f}")

    if decoders:
        dec_lambda = np.mean([r['lambda_s']['value'] for r in decoders])
        dec_beta = np.mean([r['beta']['diff_mean'] for r in decoders])
        lines.append(f"  Decoders (n={len(decoders)}): mean Λₛ = {dec_lambda:.4f}, mean β_diff = {dec_beta:.4f}")

    lines.append("\n" + "=" * 90)
    lines.append("END OF SURVEY")
    lines.append("=" * 90)

    report = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(report)

    # Also save as CSV
    csv_path = os.path.join(output_dir, 'full_results.csv')
    with open(csv_path, 'w') as f:
        f.write("model,dim,tokens,lambda_s,lambda_p,r2_lin,r2_quad,"
                "beta_diff_mean,beta_diff_p,beta_ratio_mean,"
                "mean_own_sim,mean_other_sim,alpha_mean,alpha_n\n")
        for r in results:
            f.write(f"{r['model']},{r['embedding_dim']},{r['n_tokens']},"
                    f"{r['lambda_s']['value']:.6f},{r['lambda_s']['p_value']:.8f},"
                    f"{r['lambda_s']['r2_linear']:.4f},{r['lambda_s']['r2_quadratic']:.4f},"
                    f"{r['beta']['diff_mean']:.6f},{r['beta']['p_value']:.8f},"
                    f"{r['beta']['ratio_mean']:.6f},"
                    f"{r['beta']['mean_own_centroid_sim']:.6f},"
                    f"{r['beta']['mean_other_centroid_sim']:.6f},"
                    f"{r['alpha']['mean']:.4f},{r['alpha']['n_clusters']}\n")

    # Save full JSON
    json_path = os.path.join(output_dir, 'full_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return report


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GEOMETRIC HALLUCINATION TAXONOMY — MULTI-MODEL SURVEY")
    print(f"Models to analyze: {len(MODELS)}")
    print("=" * 70)

    t_start = time.time()
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    all_results = []

    for i, (model_id, short_name, method) in enumerate(MODELS):
        print(f"\n{'━' * 70}")
        print(f"[{i+1}/{len(MODELS)}] {short_name} ({model_id})")
        print(f"{'━' * 70}")

        t_model = time.time()

        # Extract
        print(f"  Loading model...")
        embeddings, tokenizer = extract_embeddings(model_id, method)
        if embeddings is None:
            print(f"  SKIPPED — could not load model")
            all_results.append(None)
            continue

        print(f"  Embedding matrix: {embeddings.shape}")

        # Filter
        print(f"  Filtering vocabulary...")
        filtered_emb, words, self_info = filter_vocabulary(embeddings, tokenizer, model_id)
        print(f"  Filtered tokens: {len(words)}")

        if len(words) < 500:
            print(f"  SKIPPED — too few tokens ({len(words)})")
            all_results.append(None)
            del embeddings, tokenizer
            gc.collect()
            continue

        # Free full embedding matrix
        del embeddings, tokenizer
        gc.collect()

        # Analyze
        print(f"  Running analysis...")
        result = analyze_model(filtered_emb, words, self_info, short_name)

        # Print quick summary
        ls = result['lambda_s']
        b = result['beta']
        a = result['alpha']
        print(f"\n  Results for {short_name}:")
        print(f"    Λₛ = {ls['value']:.4f} (R²={ls['r2_quadratic']:.3f}, "
              f"F={ls['f_stat']:.1f}, p={ls['p_value']:.6f}) {'✓' if ls['pass'] else '✗'}")
        print(f"    β_diff = {b['diff_mean']:.4f} (p={b['p_value']:.6f}) {'✓' if b['pass'] else '✗'}")
        print(f"    β_ratio = {b['ratio_mean']:.4f}")
        print(f"    α  = {a['mean']:.4f} (n={a['n_clusters']} clusters) {'✓' if a['pass'] else '✗'}")
        print(f"    Time: {time.time() - t_model:.1f}s")

        all_results.append(result)

        # Free memory
        del filtered_emb, words, self_info
        gc.collect()

    # Cross-model analysis
    print(f"\n{'━' * 70}")
    print("GENERATING CROSS-MODEL ANALYSIS")
    print(f"{'━' * 70}")

    generate_cross_model_figures(all_results, CONFIG['output_dir'])
    report = write_cross_model_report(all_results, CONFIG['output_dir'])

    print(f"\n{'=' * 70}")
    print(f"SURVEY COMPLETE — Total runtime: {time.time() - t_start:.1f}s")
    print(f"Results in: {os.path.abspath(CONFIG['output_dir'])}")
    print(f"{'=' * 70}")
    print("\n" + report)


if __name__ == '__main__':
    main()
