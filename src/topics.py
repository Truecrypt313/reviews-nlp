import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Coherence
try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
except Exception:
    Dictionary = None
    CoherenceModel = None

# -------- Einstellungen --------
IN_CSV        = "data/processed/reviews_clean.csv"
REPORT_DIR    = Path("reports")
NMAX          = 8000
SEED          = 42
TOPN          = 10                  # Top-Wörter pro Topic

# k-Werte:
N_TOPICS_LDA  = 12                  # <- hier dein LDA-k eintragen
N_TOPICS_NMF  = 12                  # <- hier dein NMF-k eintragen

# Grid fürs Tuning:
K_GRID        = [6, 8, 10, 12, 14]
# --------------------------------

def _top_terms(components, vocab, n=TOPN, unigrams_only=False):
    out = []
    for row in components:
        idxs = np.argsort(row)[::-1]
        terms = []
        for i in idxs:
            t = vocab[i]
            if unigrams_only and (" " in t):
                continue
            terms.append(t)
            if len(terms) == n:
                break
        out.append(terms)
    return out

def _save(model_name, doc_topics, terms):
    pd.DataFrame({"doc_id": np.arange(len(doc_topics)), "topic": doc_topics}).to_csv(
        REPORT_DIR / f"{model_name}_doc_topics.csv", index=False
    )
    pd.DataFrame({"topic": np.arange(len(terms)), "top_terms": [", ".join(t) for t in terms]}).to_csv(
        REPORT_DIR / f"{model_name}_top_terms.csv", index=False
    )

# -------- Training --------
def run_lda(texts_lemma, n_topics):
    vec = CountVectorizer(min_df=5, max_df=0.9)
    X = vec.fit_transform(texts_lemma)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method="batch", random_state=SEED)
    doc_dist = lda.fit_transform(X)
    terms = _top_terms(lda.components_, vec.get_feature_names_out(), TOPN)
    _save("lda", doc_dist.argmax(1), terms)

def run_nmf(texts_lemma, n_topics):
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9)
    X = tfidf.fit_transform(texts_lemma)
    nmf = NMF(n_components=n_topics, init="nndsvd", random_state=SEED, max_iter=400)
    W = nmf.fit_transform(X)
    terms = _top_terms(nmf.components_, tfidf.get_feature_names_out(), TOPN)
    _save("nmf", W.argmax(1), terms)

def save_embedding_stats(texts_clean):
    if SentenceTransformer is None:
        print("[WARN] sentence-transformers fehlt – Embedding-Statistik übersprungen.")
        return
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts_clean, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    pd.Series({"n_docs": int(emb.shape[0]), "dim": int(emb.shape[1])}).to_json(REPORT_DIR / "embeddings_stats.json")

# -------- Coherence --------
def _tokens_from_lemma(texts_lemma):
    return [t.split() if isinstance(t, str) else [] for t in texts_lemma]

def _coherence_cv(topic_terms, tokens_list):
    if Dictionary is None or CoherenceModel is None:
        raise RuntimeError("gensim ist nicht installiert.")
    dictionary = Dictionary(tokens_list)
    cm = CoherenceModel(topics=topic_terms, texts=tokens_list, dictionary=dictionary, coherence="c_v")
    return float(cm.get_coherence())

def tune_coherence(texts_lemma):
    rows = []
    tokens_list = _tokens_from_lemma(texts_lemma)

    # LDA (Counts)
    vec = CountVectorizer(min_df=5, max_df=0.9)
    Xc = vec.fit_transform(texts_lemma)
    feats_c = vec.get_feature_names_out()
    for k in K_GRID:
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=SEED).fit(Xc)
        terms = _top_terms(lda.components_, feats_c, TOPN, unigrams_only=True)
        cv = _coherence_cv(terms, tokens_list)
        print(f"LDA  k={k:>2}  c_v={cv:.4f}")
        rows.append({"model":"LDA", "k":k, "c_v":cv})

    # NMF (TF-IDF)
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9)
    Xt = tfidf.fit_transform(texts_lemma)
    feats_t = tfidf.get_feature_names_out()
    for k in K_GRID:
        nmf = NMF(n_components=k, init="nndsvd", random_state=SEED, max_iter=400).fit(Xt)
        terms = _top_terms(nmf.components_, feats_t, TOPN, unigrams_only=True)
        cv = _coherence_cv(terms, tokens_list)
        print(f"NMF  k={k:>2}  c_v={cv:.4f}")
        rows.append({"model":"NMF", "k":k, "c_v":cv})

    df = pd.DataFrame(rows).sort_values(["model","k"])
    out = REPORT_DIR / "coherence_grid.csv"
    df.to_csv(out, index=False)
    
    for m in ["LDA", "NMF"]:
            sub = df[df["model"] == m]
            if not sub.empty:
                best = sub.loc[sub["c_v"].idxmax()]
                print(f"BEST {m}: k={int(best.k)}  c_v={best.c_v:.4f}")

# -------------------- Main --------------------
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Lade {IN_CSV}")
    df = pd.read_csv(IN_CSV)

    n = min(len(df), NMAX)
    print(f"[*] Sample {n} / {len(df)} (seed={SEED})")
    df = df.sample(n, random_state=SEED).reset_index(drop=True)

    texts_lemma = df["text_lemma"].tolist()
    texts_clean = df["text_clean"].tolist()

    if "--tune" in sys.argv:
        tune_coherence(texts_lemma)
        return

    print(f"[*] LDA (k={N_TOPICS_LDA})")
    run_lda(texts_lemma, N_TOPICS_LDA)

    print(f"[*] NMF (k={N_TOPICS_NMF})")
    run_nmf(texts_lemma, N_TOPICS_NMF)

    print("[*] Embedding-Statistik")
    save_embedding_stats(texts_clean)

    print(f"[OK] Ergebnisse in {REPORT_DIR.resolve()}")

if __name__ == "__main__":
    main()
