# nlp_pipeline.py

import sys
import re
import argparse
from pathlib import Path
from textwrap import shorten

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import spacy
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# Pfade (robust relativ zur Datei)
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_CSV = ROOT / "data" / "processed" / "reviews_clean.csv"
REPORT_DIR = ROOT / "reports"
PRETTY_DIR = REPORT_DIR / "pretty"

LDA_DOC = REPORT_DIR / "lda_doc_topics.csv"
LDA_TERMS = REPORT_DIR / "lda_top_terms.csv"
NMF_DOC = REPORT_DIR / "nmf_doc_topics.csv"
NMF_TERMS = REPORT_DIR / "nmf_top_terms.csv"

SEED = 42
NMAX = 8000
TOPN = 10
K_GRID_DEF = [6, 8, 10, 12, 14]

# --------------------- Preprocessing ---------------------
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = BeautifulSoup(s, "lxml").get_text(" ", strip=True)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pick_text_cols(df):
    pref = [c for c in ["Review Text", "Title"] if c in df.columns]
    if pref:
        cols = pref
    else:
        cols = [c for c in df.columns if any(k in c.lower() for k in ["review", "text", "title", "comment", "content"])]
    if not cols:
        raise ValueError("Keine sinnvollen Textspalten gefunden.")
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)

def lemmatize(texts):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    out = []
    for doc in nlp.pipe(texts, batch_size=400):
        toks = []
        for t in doc:
            if t.is_stop or t.is_punct or t.like_num:
                continue
            toks.append(t.lemma_.lower())
        out.append(" ".join(toks))
    return out

def preprocess():
    paths = sorted(RAW_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"Keine CSV in {RAW_DIR} gefunden.")
    src = paths[0]
    print(f"[*] Lese {src}")
    df = pd.read_csv(src, low_memory=False)

    print("[*] Spalten zusammenführen")
    df["text_raw"] = pick_text_cols(df)

    print("[*] Texte bereinigen")
    df["text_clean"] = df["text_raw"].map(clean_text)

    print("[*] Filterung (kurz/Duplikate)")
    before = len(df)
    df = df[df["text_clean"].str.len() > 20].drop_duplicates(subset=["text_clean"]).copy()
    print(f"    behalten: {len(df)}/{before}")

    print("[*] Lemmatisierung")
    df["text_lemma"] = lemmatize(df["text_clean"].tolist())

    keep = ["text_clean", "text_lemma"]
    for c in ["Rating", "Title"]:
        if c in df.columns:
            keep.append(c)

    PROC_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(PROC_CSV, index=False)
    print(f"[OK] Gespeichert: {PROC_CSV} (Zeilen={len(df)})")

# --------------------- Topic Modeling ---------------------
def top_terms(components, vocab, n=TOPN, unigrams_only=False):
    terms = []
    for row in components:
        idxs = np.argsort(row)[::-1]
        chosen = []
        for i in idxs:
            t = vocab[i]
            if unigrams_only and (" " in t):
                continue
            chosen.append(t)
            if len(chosen) == n:
                break
        terms.append(chosen)
    return terms

def save_topics(model_name, doc_topics, terms):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"doc_id": np.arange(len(doc_topics)), "topic": doc_topics}).to_csv(
        REPORT_DIR / f"{model_name}_doc_topics.csv", index=False
    )
    pd.DataFrame({"topic": np.arange(len(terms)), "top_terms": [", ".join(t) for t in terms]}).to_csv(
        REPORT_DIR / f"{model_name}_top_terms.csv", index=False
    )

def run_lda(texts_lemma, k):
    vec = CountVectorizer(min_df=5, max_df=0.9)
    X = vec.fit_transform(texts_lemma)
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=SEED)
    doc_dist = lda.fit_transform(X)
    terms = top_terms(lda.components_, vec.get_feature_names_out(), TOPN)
    save_topics("lda", doc_dist.argmax(1), terms)

def run_nmf(texts_lemma, k):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
    X = tfidf.fit_transform(texts_lemma)
    nmf = NMF(n_components=k, init="nndsvd", random_state=SEED, max_iter=400)
    W = nmf.fit_transform(X)
    terms = top_terms(nmf.components_, tfidf.get_feature_names_out(), TOPN)
    save_topics("nmf", W.argmax(1), terms)

# --------- Coherence (c_v) ---------
def tokens_from_lemma(texts_lemma):
    out = []
    for t in texts_lemma:
        if isinstance(t, str):
            out.append(t.split())
        else:
            out.append([])
    return out

def coherence_cv(topic_terms, tokens_list):
    dct = Dictionary(tokens_list)
    cm = CoherenceModel(topics=topic_terms, texts=tokens_list, dictionary=dct, coherence="c_v")
    return float(cm.get_coherence())

def tune(texts_lemma, k_grid):
    rows = []
    toks = tokens_from_lemma(texts_lemma)

    vec = CountVectorizer(min_df=5, max_df=0.9)
    Xc = vec.fit_transform(texts_lemma)
    feats_c = vec.get_feature_names_out()
    for k in k_grid:
        lda = LatentDirichletAllocation(n_components=k, learning_method="batch", random_state=SEED).fit(Xc)
        terms = top_terms(lda.components_, feats_c, TOPN, unigrams_only=True)
        cv = coherence_cv(terms, toks)
        print(f"LDA  k={k:>2}  c_v={cv:.4f}")
        rows.append({"model": "LDA", "k": k, "c_v": cv})

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
    Xt = tfidf.fit_transform(texts_lemma)
    feats_t = tfidf.get_feature_names_out()
    for k in k_grid:
        nmf = NMF(n_components=k, init="nndsvd", random_state=SEED, max_iter=400).fit(Xt)
        terms = top_terms(nmf.components_, feats_t, TOPN, unigrams_only=True)
        cv = coherence_cv(terms, toks)
        print(f"NMF  k={k:>2}  c_v={cv:.4f}")
        rows.append({"model": "NMF", "k": k, "c_v": cv})

    df = pd.DataFrame(rows).sort_values(["model", "k"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "coherence_grid.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    for m in ["LDA", "NMF"]:
        sub = df[df["model"] == m]
        if len(sub):
            best = sub.loc[sub["c_v"].idxmax()]
            print(f"BEST {m}: k={int(best.k)}  c_v={best.c_v:.4f}")

# --------------------- Visualisierung ---------------------
def ensure_pretty():
    PRETTY_DIR.mkdir(parents=True, exist_ok=True)

def join_assign_terms(df_base, assign_path, term_path):
    assign = pd.read_csv(assign_path)
    if not {"doc_id", "topic"} <= set(assign.columns):
        raise ValueError(f"Spalten fehlen in {assign_path}")
    terms = pd.read_csv(term_path)
    if not {"topic", "top_terms"} <= set(terms.columns):
        raise ValueError(f"Spalten fehlen in {term_path}")

    df = df_base.merge(assign, on="doc_id", how="inner")
    terms = terms.copy()
    terms["label"] = terms["top_terms"].astype(str).str.split(",").str[0].fillna("").str.strip()
    return df, terms

def build_summary(joined, terms, rating_col="Rating"):
    vc = joined["topic"].value_counts().sort_index()
    if rating_col in joined.columns:
        avg = joined.groupby("topic", as_index=True)[rating_col].mean()
    else:
        avg = pd.Series(index=vc.index, dtype="float64")
    t = terms.set_index("topic")[["label", "top_terms"]].reindex(vc.index).fillna("")
    s = pd.DataFrame({
        "topic": vc.index,
        "label": t["label"],
        "top_terms": t["top_terms"],
        "freq": vc.values,
        "avg_rating": avg.round(2),
    }).reset_index(drop=True)
    return s.sort_values("topic").reset_index(drop=True)

def base_ax(figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

def plot_freq(summary, title):
    s = summary.sort_values("topic")
    ax = base_ax((9, max(4, 0.40 * len(s))))
    ax.barh(s["topic"].astype(str), s["freq"], edgecolor="none")
    ax.set_title(f"{title} – Häufigkeit (n={int(s['freq'].sum())})")
    ax.set_xlabel("Anzahl Reviews")
    ax.set_ylabel("Topic ID")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    out = PRETTY_DIR / f"{title.lower()}_freq.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close(ax.figure)
    return out

def plot_rating(summary, title):
    s = summary.dropna(subset=["avg_rating"])
    if s.empty:
        return None
    s = s.sort_values("topic")
    ax = base_ax((9, max(4, 0.40 * len(s))))
    ax.barh(s["topic"].astype(str), s["avg_rating"], edgecolor="none")
    ax.set_title(f"{title} – Ø Rating je Topic")
    ax.set_xlabel("Ø Rating")
    ax.set_ylabel("Topic ID")
    xlim = ax.get_xlim()
    offset = (xlim[1] - xlim[0]) * 0.012 if xlim[1] > xlim[0] else 0.05
    vals = s["avg_rating"].to_numpy(dtype=float)
    for y, val in enumerate(vals):
        ax.text(val + offset, y, f"{val:.2f}", va="center")
    out = PRETTY_DIR / f"{title.lower()}_avg_rating.png"
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close(ax.figure)
    return out

def render_index(sections, total_docs, data_source):
    if not sections:
        raise ValueError("Keine Modelle zum Rendern.")
    def table_html(df):
        df = df.sort_values("topic")
        rows = []
        for r in df.itertuples():
            label = (r.label or str(r.top_terms).split(",")[0]).strip()
            rows.append(
                "<tr>"
                f"<td>#{int(r.topic)}</td>"
                f"<td class='lbl'><div>{shorten(label, 64, placeholder='…')}<br>"
                f"<small class='muted'>{shorten(str(r.top_terms), 96, placeholder='…')}</small></div></td>"
                f"<td>{int(r.freq)}</td>"
                f"<td>{'–' if pd.isna(r.avg_rating) else f'{r.avg_rating:.2f}'}</td>"
                "</tr>"
            )
        colgroup = (
            "<colgroup>"
            "<col style='width:14%'><col style='width:58%'><col style='width:14%'><col style='width:14%'>"
            "</colgroup>"
        )
        thead = (
            "<thead><tr>"
            "<th>Topic</th><th>Label &amp; Top-Wörter</th>"
            "<th>Anzahl</th><th>Ø Rating</th>"
            "</tr></thead>"
        )
        return f"<table class='tbl'>{colgroup}{thead}<tbody>{''.join(rows)}</tbody></table>"

    css = (
        "body{font-family:system-ui,Arial,sans-serif;margin:24px;line-height:1.5}"
        "h1{margin:0 0 6px} h2{margin:22px 0 10px}"
        ".meta{color:#6b7280;margin:0 0 12px}"
        ".tbl{width:100%;border-collapse:collapse;table-layout:fixed;margin:10px 0}"
        ".tbl th,.tbl td{border-bottom:1px solid #e5e7eb;padding:8px 10px;text-align:center;vertical-align:top}"
        ".tbl thead{background:#f3f4f6}"
        ".tbl td.lbl{text-align:left}"
        ".muted{color:#6b7280}"
        "figure{margin:10px 0}"
        "img{max-width:100%;height:auto;display:block}"
    )

    html = [
        "<!doctype html><html lang='de'><meta charset='utf-8'>",
        "<title>NLP Topics – Report</title>",
        f"<style>{css}</style>",
        "<h1>Topics – Report</h1>",
        f"<p class='meta'>Quelle: <code>{data_source}</code> · Reviews im Sample: <b>{total_docs}</b></p>"
    ]

    for title, payload in sections.items():
        files = payload["files"]
        summary = payload["summary"]
        html.append(f"<h2>{title}</h2>")
        html.append(table_html(summary))
        if files.get("freq"):
            html += ["<figure>", f"<img src='{files['freq']}' alt='{title} Häufigkeit'>", "</figure>"]
        if files.get("rating"):
            html += ["<figure>", f"<img src='{files['rating']}' alt='{title} Ø Rating je Topic'>", "</figure>"]

    html.append("</html>")
    out = PRETTY_DIR / "index.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out

# --------------------- Pipeline ---------------------
def sample_base_df(proc_csv, nmax=NMAX, seed=SEED):
    df = pd.read_csv(proc_csv)
    n = min(len(df), nmax)
    df = df.sample(n, random_state=seed).reset_index(drop=True)
    df.insert(0, "doc_id", np.arange(n, dtype=int))
    return df

def train_and_viz(k_lda, k_nmf):
    base = sample_base_df(PROC_CSV, NMAX, SEED)
    texts_lemma = base["text_lemma"].tolist()

    print(f"[*] LDA (k={k_lda})")
    run_lda(texts_lemma, k_lda)
    print(f"[*] NMF (k={k_nmf})")
    run_nmf(texts_lemma, k_nmf)

    ensure_pretty()
    sections = {}

    if LDA_DOC.exists() and LDA_TERMS.exists():
        joined, terms = join_assign_terms(base, LDA_DOC, LDA_TERMS)
        summary = build_summary(joined, terms)
        files = {"freq": plot_freq(summary, "LDA").name}
        rp = plot_rating(summary, "LDA")
        if rp is not None:
            files["rating"] = rp.name
        sections["LDA"] = {"files": files, "summary": summary}

    if NMF_DOC.exists() and NMF_TERMS.exists():
        joined, terms = join_assign_terms(base, NMF_DOC, NMF_TERMS)
        summary = build_summary(joined, terms)
        files = {"freq": plot_freq(summary, "NMF").name}
        rp = plot_rating(summary, "NMF")
        if rp is not None:
            files["rating"] = rp.name
        sections["NMF"] = {"files": files, "summary": summary}

    index_html = render_index(sections, total_docs=len(base), data_source=PROC_CSV)
    print("[OK] Visuals:", PRETTY_DIR.resolve())
    print("     Öffne:", index_html)

# --------------------- CLI ---------------------
def main(argv):
    ap = argparse.ArgumentParser(description="NLP-Pipeline (Preprocess | Tune | All).")
    ap.add_argument("--preprocess", action="store_true", help="Nur Preprocessing.")
    ap.add_argument("--tune", action="store_true", help="Nur Coherence-Tuning (c_v).")
    ap.add_argument("--all", action="store_true", help="Preprocess (falls nötig) + Train + Visualisierung.")
    ap.add_argument("--k-grid", type=str, default=None, help="k-Liste für --tune, z.B. 6,8,10,12,14")
    ap.add_argument("--k-lda", type=int, default=None, help="Topics für LDA (nur bei --all).")
    ap.add_argument("--k-nmf", type=int, default=None, help="Topics für NMF (nur bei --all).")
    args = ap.parse_args(argv)

    if args.preprocess:
        preprocess()
        return

    if args.tune:
            # falls processed fehlt, automatisch preprocess
        if not PROC_CSV.exists():
            preprocess()
        texts_lemma = sample_base_df(PROC_CSV, NMAX, SEED)["text_lemma"].tolist()
        if args.k_grid:
            k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
        else:
            k_grid = K_GRID_DEF
        print("[*] Coherence-Tuning (c_v)")
        tune(texts_lemma, k_grid)
        return

    if args.all:
        if not PROC_CSV.exists():
            preprocess()
        if args.k_lda is None or args.k_nmf is None:
            raise SystemExit("Bitte --k-lda und --k-nmf angeben (z. B. aus --tune übernehmen).")
        train_and_viz(args.k_lda, args.k_nmf)
        return

    ap.print_help()

if __name__ == "__main__":
    main(sys.argv[1:])
