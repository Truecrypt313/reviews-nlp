from __future__ import annotations

from pathlib import Path
from textwrap import shorten
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ------------------ Pfade ------------------
DATA_CSV  = Path("data/processed/reviews_clean.csv")
LDA_DOC   = Path("reports/lda_doc_topics.csv")
LDA_TERMS = Path("reports/lda_top_terms.csv")
NMF_DOC   = Path("reports/nmf_doc_topics.csv")
NMF_TERMS = Path("reports/nmf_top_terms.csv")
OUTDIR    = Path("reports/pretty")


# --------------- Utilities -----------------
def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

def infer_doc_count(assign_path: Path) -> int:
    a = pd.read_csv(assign_path, usecols=["doc_id"])
    if a.empty:
        raise ValueError(f"Leere Datei: {assign_path}")
    return int(a["doc_id"].max()) + 1

def load_base_df(n_docs: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    if len(df) < n_docs:
        raise ValueError(f"Benötige {n_docs} Zeilen, gefunden: {len(df)}")
    df = df.sample(n_docs, random_state=42).reset_index(drop=True)
    df.insert(0, "doc_id", np.arange(n_docs, dtype=int))
    return df

def join_assign_terms(df_base: pd.DataFrame, assign_path: Path, term_path: Path):
    assign = pd.read_csv(assign_path)
    if not {"doc_id","topic"} <= set(assign.columns):
        raise ValueError(f"Erwarte Spalten 'doc_id' & 'topic' in {assign_path}")
    terms = pd.read_csv(term_path)
    if not {"topic","top_terms"} <= set(terms.columns):
        raise ValueError(f"Erwarte Spalten 'topic' & 'top_terms' in {term_path}")

    df = df_base.merge(assign, on="doc_id", how="inner")

    # Label 1 Top-Wort
    terms = terms.copy()
    terms["label_guess"] = (
        terms["top_terms"]
        .astype(str)
        .str.split(",")
        .str[0]
        .fillna("")
        .str.strip()
    )
    return df, terms

def build_topic_summary(joined: pd.DataFrame, terms: pd.DataFrame, rating_col="Rating") -> pd.DataFrame:
    """Nur das Nötigste: Topic-ID, Label/Top-Wörter, Häufigkeit, Ø-Rating."""
    vc = joined["topic"].value_counts().sort_index()  # Häufigkeit je Topic-ID

    if rating_col in joined.columns:
        avg = joined.groupby("topic", as_index=True)[rating_col].mean()
    else:
        avg = pd.Series(index=vc.index, dtype="float64")

    meta = terms.set_index("topic")[["label_guess","top_terms"]].reindex(vc.index).fillna("")
    summary = pd.DataFrame({
        "topic": vc.index,
        "label": meta["label_guess"],
        "top_terms": meta["top_terms"],
        "freq": vc.values,
        "avg_rating": avg.round(2),
    }).reset_index(drop=True)

    # Immer nach Topic-ID sortieren
    return summary.sort_values("topic").reset_index(drop=True)


# --------------- Plots (nach Topic-ID) ---------------
def _base_ax(figsize: Tuple[float,float] = (9,6)) -> plt.Axes:
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

def plot_topic_frequency(summary: pd.DataFrame, title: str) -> Path:
    s = summary.sort_values("topic", ascending=True)
    ax = _base_ax((9, max(4, 0.40 * len(s))))
    ax.barh(s["topic"].astype(str), s["freq"], edgecolor="none")
    ax.set_title(f"{title} – Häufigkeit (n={int(s['freq'].sum())})")
    ax.set_xlabel("Anzahl Reviews")
    ax.set_ylabel("Topic ID")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    out = OUTDIR / f"{title.lower()}_freq.png"
    plt.tight_layout(); plt.savefig(out, dpi=170); plt.close(ax.figure)
    return out

def plot_topic_rating(summary: pd.DataFrame, title: str) -> Path | None:
    s = summary.dropna(subset=["avg_rating"])
    if s.empty:
        return None
    s = s.sort_values("topic", ascending=True)
    ax = _base_ax((9, max(4, 0.40 * len(s))))
    ax.barh(s["topic"].astype(str), s["avg_rating"], edgecolor="none")
    ax.set_title(f"{title} – Ø Rating je Topic")
    ax.set_xlabel("Ø Rating"); ax.set_ylabel("Topic ID")

    # Werte an die Balken schreiben
    xlim = ax.get_xlim()
    offset = (xlim[1] - xlim[0]) * 0.012 if xlim[1] > xlim[0] else 0.05
    for y, val in enumerate(s["avg_rating"].to_numpy(dtype=float)):
        ax.text(val + offset, y, f"{val:.2f}", va="center")

    out = OUTDIR / f"{title.lower()}_avg_rating.png"
    plt.tight_layout(); plt.savefig(out, dpi=170); plt.close(ax.figure)
    return out


# --------------- HTML ---------------
def render_html(sections: Dict[str, Dict[str, Any]], total_docs: int, data_source: Path) -> Path:
    if not sections:
        raise ValueError("Keine Modell-Abschnitte zum Rendern gefunden.")

    def table_html(df: pd.DataFrame) -> str:
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
        files = payload["files"]; summary = payload["summary"]
        html.append(f"<h2>{title}</h2>")
        html.append(table_html(summary))
        if files.get("freq"):
            html += ["<figure>", f"<img src='{files['freq']}' alt='{title} Häufigkeit (Topic-ID sortiert)'>", "</figure>"]
        if files.get("rating"):
            html += ["<figure>", f"<img src='{files['rating']}' alt='{title} Ø Rating je Topic'>", "</figure>"]

    html.append("</html>")
    out = OUTDIR / "index.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out


# --------------- Pipeline ----------------
def run_model_block(name: str, doc_path: Path, term_path: Path, base_df: pd.DataFrame) -> Dict[str, Any]:
    joined, terms = join_assign_terms(base_df, doc_path, term_path)
    summary = build_topic_summary(joined, terms)
    freq_png   = plot_topic_frequency(summary, name)
    rating_png = plot_topic_rating(summary, name)
    return {
        "files": {"freq": freq_png.name, "rating": rating_png.name if rating_png else None},
        "summary": summary,
    }

def main() -> None:
    ensure_outdir()
    n_lda = infer_doc_count(LDA_DOC) if LDA_DOC.exists() else 0
    n_nmf = infer_doc_count(NMF_DOC) if NMF_DOC.exists() else 0
    n_docs = max(n_lda, n_nmf)
    if n_docs == 0:
        raise FileNotFoundError("Keine Topic-Zuweisungsdateien gefunden. Erst src/topics.py ausführen.")

    base_df = load_base_df(n_docs=n_docs)

    sections: Dict[str, Dict[str, Any]] = {}
    if LDA_DOC.exists() and LDA_TERMS.exists():
        sections["LDA"] = run_model_block("LDA", LDA_DOC, LDA_TERMS, base_df)
    if NMF_DOC.exists() and NMF_TERMS.exists():
        sections["NMF"] = run_model_block("NMF", NMF_DOC, NMF_TERMS, base_df)

    index_html = render_html(sections, total_docs=len(base_df), data_source=DATA_CSV)
    print("[OK] Visuals in:", OUTDIR.resolve())
    print("     Öffne:", index_html)

if __name__ == "__main__":
    main()
