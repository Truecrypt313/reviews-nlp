import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RAW_GLOB = "data/raw/*.csv"
OUT_CSV  = "data/processed/reviews_clean.csv"
MIN_LEN  = 20  # Mindestlänge nach Cleaning

def clean_text(s: str) -> str:
    """HTML + URLs entfernen, Whitespace normalisieren."""
    if not isinstance(s, str):
        return ""
    # HTML raus
    s = BeautifulSoup(s, "lxml").get_text(" ", strip=True)
    # URLs raus
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    # Whitespaces zusammenfassen
    s = re.sub(r"\s+", " ", s).strip()
    return s

def select_text_columns(df: pd.DataFrame) -> pd.Series:
    """Nimmt bevorzugt 'Review Text' und 'Title', sonst heuristisch textähnliche Spalten."""
    preferred = [c for c in ["Review Text", "Title"] if c in df.columns]
    if preferred:
        cols = preferred
    else:
        cols = [c for c in df.columns if any(k in c.lower() for k in ["review","text","title","comment","content"])]
        if not cols:
            raise ValueError("Keine geeigneten Textspalten gefunden.")
    # NaN vermeiden
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1)

def lemmatize_or_lower(texts: list[str]) -> list[str]:
    """Versuche spaCy-Lemmatisierung; fällt bei Problemen auf einfaches Lowercasing zurück."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception as e:
        print("[WARN] spaCy/Modell nicht verfügbar – nutze Lowercasing:", e)
        return [t.lower() for t in texts]

    out = []
    for doc in nlp.pipe(texts, batch_size=500):
        toks = []
        for t in doc:
            if t.is_stop or t.is_punct or t.like_num:
                continue
            toks.append(t.lemma_.lower())
        out.append(" ".join(toks))
    return out

def main(in_glob: str = RAW_GLOB, out_csv: str = OUT_CSV) -> None:
    paths = list(Path().glob(in_glob))
    if not paths:
        raise FileNotFoundError(f"Keine CSV unter {in_glob} gefunden.")
    path = paths[0]
    print(f"[*] Lese {path} ...")
    df = pd.read_csv(path, low_memory=False)

    print("[*] Wähle Textspalten und führe zusammen ...")
    df["text_raw"] = select_text_columns(df)

    print("[*] Clean Texte (HTML/URLs/Whitespace) ...")
    df["text_clean"] = df["text_raw"].map(clean_text)

    print(f"[*] Filtere sehr kurze Texte (<={MIN_LEN}) und Duplikate ...")
    before = len(df)
    df = df[df["text_clean"].str.len() > MIN_LEN]
    df = df.drop_duplicates(subset=["text_clean"]).copy()
    print(f"    behalten: {len(df)}/{before}")

    print("[*] Lemmatisiere...")
    df["text_lemma"] = lemmatize_or_lower(df["text_clean"].tolist())

    keep_cols = ["text_clean", "text_lemma"]
    for c in ["Rating", "Title"]:
        if c in df.columns:
            keep_cols.append(c)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep_cols].to_csv(out_path, index=False)
    print(f"[OK] Gespeichert: {out_path} | Zeilen: {len(df)}")

if __name__ == "__main__":
    main()
