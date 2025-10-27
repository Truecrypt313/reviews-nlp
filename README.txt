Ordner „reviews-nlp“ in einer Umgebung deiner Wahl öffnen (empfohlen: VS Code).

Die virtuelle Umgebung wird ggf. über settings.json im Projekt-Root automatisch aktiviert. Falls nicht, manuell aktivieren:

> > python3 -m venv .venv
> > source .venv/bin/activate

Abhängigkeiten installieren:

> > pip install -r requirements.txt

Den zu analysierenden Datensatz im CSV-Format unter data/raw/ ablegen (Der Datensatz liegt bereits in data/raw/ im .CSV-Format).

Preprocessing ausführen, um einen bereinigten Datensatz zu erzeugen:

> > python src/preprocess.py

Der bereinigte Datensatz liegt danach unter data/processed/ als "reviews_clean.csv".

Zur Abschätzung, wie „stimmig“ (interpretierbar) die Top-Wörter eines Themas sind, bestimmen wir den Topic-Coherence-Score. 
Er prüft, ob die wichtigsten Wörter eines Topics häufig gemeinsam in denselben Dokumenten auftreten. 
Dafür zunächst die c_v-Coherence über mehrere k-Werte für LDA und NMF berechnen:

> > python src/topics.py --tune

Im Terminal erscheint c_v je k und Modell. Zusätzlich wird reports/coherence_grid.csv erstellt. 
Anhand dessen geeignete k-Werte wählen (je eins für LDA und NMF). Diese k-Werte im nächsten Schritt verwenden (optimales k für LDA und NMF wird im Terminal angezeigt).

src/topics.py öffnen und oben N_TOPICS_LDA = <dein_k> sowie N_TOPICS_NMF = <dein_k> setzen (Code-Zeile 32 und 33).
Anschließend Modelle trainieren:

> > python src/topics.py

Die Ergebnisse werden unter reports/ abgelegt.

Visualisierung erzeugen:

> > python src/visualize.py

Dies erstellt unter reports/pretty/ Tabellen, Balkendiagramme und eine index.html. 
Die index.html im Browser öffnen (Rechtsklick > Öffnen mit > Firefox/Edge/Chrome).