# NLP Projekt: Data Analysis V2

### Projekt Download
Den Projektordner herunterladen.<br>
Ordner „reviews-nlp“ in einer Umgebung deiner Wahl öffnen/importieren.

Die virtuelle Umgebung wird ggf. über settings.json im Projekt-Root automatisch aktiviert.<br>
Falls nicht, 
### .venv manuell aktivieren:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```
Den zu analysierenden Datensatz im CSV-Format unter data/raw/ ablegen (Der Datensatz liegt bereits in data/raw/ im .CSV-Format).

## Variante 1: Ausführung des Codes in Version 2 vom 23.11.2025

### Preprocessing ausführen
```bash
python nlp_pipeline.py --preprocess
```
### Coherence-Tuning starten
```bash
python nlp_pipeline.py --tune --k-grid 6,8,10,12,14
```
### Mit den gewählten k-Werten trainieren + visualisieren
```bash
python nlp_pipeline.py --all --k-lda <dein_k_lda> --k-nmf <dein_k_nmf>
```

## Variante 2: 
### Preprocessing ausführen, um einen bereinigten Datensatz zu erzeugen:
```bash
python src/preprocess.py
```
Der bereinigte Datensatz liegt danach unter data/processed/ als "reviews_clean.csv".

Zur Abschätzung, wie „stimmig“ (interpretierbar) die Top-Wörter eines Themas sind, bestimmen wir den Topic-Coherence-Score.<br> 
Er prüft, ob die wichtigsten Wörter eines Topics häufig gemeinsam in denselben Dokumenten auftreten.<br>
### Dafür zunächst die c_v-Coherence über mehrere k-Werte für LDA und NMF berechnen:
```bash
python src/topics.py --tune
```
Im Terminal erscheint c_v je k und Modell. Zusätzlich wird reports/coherence_grid.csv erstellt. 
Anhand dessen geeignete k-Werte wählen (je eins für LDA und NMF). Diese k-Werte im nächsten Schritt verwenden (optimales k für LDA und NMF wird im Terminal angezeigt).

### src/topics.py öffnen und k-Werte einsetzen:
```bash
N_TOPICS_LDA = <dein_k>
N_TOPICS_NMF = <dein_k>
(Code-Zeile 32 und 33).
```
### Anschließend Modelle trainieren:
```bash
python src/topics.py
```
Die Ergebnisse werden unter reports/ abgelegt.

### Visualisierung erzeugen:
```bash
python src/visualize.py
```
Dies erstellt unter reports/pretty/ Tabellen, Balkendiagramme und eine index.html.<br>
Die index.html im Browser öffnen (Rechtsklick > Öffnen mit > Firefox/Edge/Chrome).
