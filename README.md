# PlumeSense

Eine Drohne sucht ein Gasleck in einer 24×24-Simulation. Einziger Gassensor ist auf der Drohne. Gas tritt kontinuierlich aus und wird bei jedem Schritt durch Wind und Diffusion weitertransportiert; Wände blockieren den Transport.

**Aktuell kommen sowohl Quellenbewertung als auch Gefahrenprognose von TabPFN.** Das frühere physikalische Quellenmodell liegt ausschließlich als Vergleich in `source_search_physics.py` und wird von der Live-Demo nicht verwendet.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py generate
python source_learning.py
python launch_demo.py
```

Demo: http://localhost:8501. Standardbackend `tabpfn-api` verwendet die RTX 4090 auf `my4090`. `launch_demo.py` öffnet den SSH-Tunnel und startet Streamlit. Modellzugriff und passende Lizenz müssen auf dem GPU-Server eingerichtet sein. Der laufende Dienst ist bereits konfiguriert.

```bash
ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ConnectTimeout=8 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18000:127.0.0.1:8000 my4090
```

`my4090` nutzt die vorhandene SSH-Config samt ProxyJump. Einstellungen: `PLUMESENSE_API_URL`, `PLUMESENSE_SSH_HOST`, `PLUMESENSE_REMOTE_PORT`, `PLUMESENSE_AUTO_TUNNEL=0`. Vorhandene Tunnel werden wiederverwendet; selbst gestartete Tunnel beim normalen Prozessende beendet.

## Was TabPFN erhält

**Quellenmodell:** Jeder freien Kartenzelle wird eine Tabellenzeile zugeordnet. Features sind Kandidatenposition, Wind, Zeit, Wandabstand, Messanzahl, Mittelwert/Maximum, Konzentrationsschwerpunkt, Besuchszahlen und Messwerte in 4×4 räumlichen Bins sowie die vier stärksten und die vier jüngsten Messungen mit Position/Zeit. Alle bisherigen mobilen Messungen gehen in die Zusammenfassungen ein. Die rohe, beliebig lange Zeitreihe kann nicht als variable Spaltenzahl an TabPFN übergeben werden.

Offline erzeugte Simulationen liefern das Label `target_source_near`: liegt die Kandidatenzelle höchstens eine Manhattan-Zelle von der tatsächlichen Quelle entfernt? Die tatsächliche Quelle wird ausschließlich für Trainingslabels und Evaluation gelesen, nicht für Live-Features oder die Suchplanung. `TabPFNClassifier.predict_proba()` bewertet neue Kandidaten. Die Quelle mit dem höchsten Score ist die aktuelle Schätzung.

100 Episoden ergeben 26.871 Kandidatenzeilen. Episoden 0–74 bilden den Trainingspool, 75–99 den Testpool. Maximal 8.000 zufällig ausgewählte Trainingszeilen werden mit festem Seed als Kontext verwendet. Positive Kandidaten sind beim Erzeugen überrepräsentiert. Die Heatmap zeigt deshalb **normierte Rangfolgescores, keine kalibrierten Quellenwahrscheinlichkeiten**.

**Gefahrenmodell:** Separate TabPFN-Klassifikation für `target_hazard_t4`, basierend auf der aktuellen/vorherigen mobilen Messung, Position, Wind, Karte und Messabdeckung. Ohne feste Sensoren sind die einfachen IDW-Felder räumlich konstant. Dieses Modell ist derzeit noch schwach.

**Planner:** Verwendet die TabPFN-Quellenscores, Besuchszähler und A* zum nächsten Messziel. Der Planner hält Routen für mindestens sechs Simulationsschritte und überprüft sie dann erneut. Ohne klares Gassignal priorisiert er ungemessene Flächen; mit Signal regionale TabPFN-Quellenscores. Tatsächliche Wegkosten, Besuchszähler und die Abdeckung eines Messpunkts werden berücksichtigt. Randzellen bleiben möglich, werden aber nicht pauschal bevorzugt. Gefahr erhält nur ein kleines Gewicht, da die Mission Quellenortung ist. Keine inverse Ausbreitungsrechnung und kein Zugriff auf die tatsächliche Quelle. Eine stabile Schätzung mit positiven Messungen vor Ort wird nur als vermuteter Fund gemeldet, nicht als Beweis.

## API auf der RTX 4090

Servercode: `api/server.py`. Remote: `~/projects/countercoach/server.py`, Log `plumesense-api.log`. Der Dienst läuft mit nohup und startet nach einem Server-Reboot nicht automatisch. Frühere Versionen wurden dort gesichert.

POST `/predict` erhält `X_train`, `y_train`, `X_test` und optional `context` (`hazard` oder `source`). Antwort: `probabilities`, `classes`, `predictions`, `fit_reused`, `elapsed_seconds`. Zwei getrennte Fit-Kontexte verhindern ständiges Wechseln zwischen Quellen- und Gefahrenmodell; ein Lock schützt gemeinsame Modelle. TabPFN v3.5-fast, CUDA, ein Ensemblemitglied. `/health` bestätigt CUDA und NVIDIA GeForce RTX 4090.

Daten werden an den konfigurierten API-Dienst übertragen. Jede Anfrage enthält derzeit die vollständigen Kontextdaten. `fit_seconds` im API-Evaluationsbericht bezeichnet nur die lokale Vorbereitung; echtes Fit, Netzwerk und Inferenz sind in der ersten Request-Latenz enthalten. Lokales Backend `tabpfn` benötigt eigene Modellgewichte/Lizenz und nutzt `device=auto`.

## Evaluation und aktueller Stand

```bash
python -m pytest -q
python run.py audit
python run.py evaluate --backend tabpfn-api --navigation-episodes 0
python evaluate_source_model.py
python evaluate_search.py --episodes 1 --steps 100 --backend tabpfn-api
```

`evaluate_source_model.py` bewertet alle freien Zellen in fünf ungesehenen Episoden zu je drei Zeitpunkten. Nicht nur vorselektierte Kandidaten werden geprüft. Ergebnis des ersten echten 4090-Laufs (`reports/tabpfn_source_full_grid_v1.json`): mittlerer Ortungsfehler **9,87 Zellen**, **2/15** Momentaufnahmen innerhalb von zwei Zellen. Auf ungesehenen Episoden ist die Quellenprognose damit **noch nicht zuverlässig**.

Vollständiger Suchlauf Seed 100 (`reports/tabpfn_source_search_tabpfn-api_v1.json`): vermuteter Fund nach 88 Schritten, tatsächlicher Schätzfehler **3,16 Zellen**; das Erfolgskriterium von höchstens zwei Zellen wurde **nicht erreicht**. Stabile Modellschätzungen plus positive Gaswerte reichen nicht aus, einen Fund sicher zu bestätigen. Das wird in der Oberfläche ausdrücklich als Vermutung bezeichnet.

TabPFN-Gefahrenmodell nur mit mobilem Sensor: AUROC **0,552**, F1 **0** bei Schwelle 0,5, Brier **0,0359** auf 3.888 ungesehenen Rasterabfragen. Wenige Messungen ohne Kontakt mit der Wolke bestimmen die Quelle oft nicht eindeutig. Weitere unabhängige Simulationen, bessere Historienmerkmale, breitere Erkundung und eine validierte Abbruchregel bleiben erforderlich.

Die früheren **9/9** erfolgreichen Ortungen gehören zum archivierten **physikalischen Quellenmodell**. Sie sind kein Ergebnis der jetzigen TabPFN-Quellensuche. Ältere Berichte bleiben als Vergleich erhalten und dürfen nicht als aktuelle Modellleistung präsentiert werden.

## Originaldaten

Unverändert unter `data/original/`: 6.696 Trainings- und 2.232 Testzeilen, acht feste Messpositionen und ca. 6,3 % positive Gefahrenlabels. Die reduzierte Trainingsdatei stimmt mit den entsprechenden Spalten der vollständigen Datei überein. Vollständige Testmetadaten fehlen im Anhang, daher lässt sich die ursprüngliche Episodentrennung nicht unabhängig bestätigen. `predictions.csv` sagt Quellen-Y voraus und ist keine Gefahrenprognose.

Originaldaten werden nicht für die Live-Drohne verwendet, da sie von festen Sensoren stammen. Die neuen Dateien `synthetic_drone_only.csv` und `source_histories.csv` werden reproduzierbar erzeugt und sind nicht versioniert. Ground Truth wird in der Demo ausschließlich separat angezeigt und zur Auswertung gelesen.


### Erkundung und Historienmerkmale überarbeitet

Die Trainingsflugwege starten nun zur Hälfte an zufälligen freien Positionen, zur Hälfte bei (0,0). Besuchskosten verringern wiederholte Korridore. Neue Features: Messdistanzen, Messwerte und Alter der vier nächsten historischen Messungen relativ zum Quellenkandidaten; zeitgewichtete historische IDW, lokale Messanzahl, Mittelwert/Maximum und Windprojektionen zu den stärksten Messorten. Nur reale mobile Beobachtungen gehen ein.

Die Suchplanung verwendet regionale Quellenscores, den Abstand zu bisherigen Messorten und noch ungemessene Nachbarschaften. Ohne deutliches Gas überwiegt Erkundung; nach Gaskontakt überwiegen TabPFN-Quellenregionen. Wegkosten werden über erreichbare freie Zellen berechnet. Zieländerungen erfolgen frühestens nach sechs Schritten oder bei Ankunft. Geometrische Abdeckung ist eine Heuristik für informative Messungen, keine berechnete Informationsentropie.

UI-Test auf der echten API: erste acht Schritte von (0,0) über (0,1), (0,2), (1,2), (1,3), (2,3), (3,3), (4,3) nach (5,3). Kein langes Folgen der äußeren Kante mehr. 19 Tests bestehen.

Neue Modellprüfung: Episoden 80–84, 7/15 Schätzungen innerhalb von zwei Zellen, mittlerer Fehler 5,35 Zellen (`reports/tabpfn_source_grid_v2_80_84.json`). Vergleich auf Episoden 75–79: 0/15, mittlerer Fehler 10,26 Zellen (`reports/tabpfn_source_grid_v2_75_79.json`), gegenüber zuvor 2/15 und 9,87. Die Modellqualität ist also gemischt; eine allgemeine Verbesserung der TabPFN-Ortung ist nicht nachgewiesen. Randverhalten der Planung und Genauigkeit des Modells werden getrennt bewertet.

Aktueller vollständiger 4090-Suchlauf (Seed 100, maximal 100 Schritte): kein bestätigter Fund, abschließender Ortungsfehler 2,0 Zellen, Exposition 1,955, Laufzeit 163,24 s. Bericht: `reports/tabpfn_source_search_tabpfn-api.json`. Der frühere Lauf liegt als `_v1.json` vor. Der Planner erkundet das Innere, aber eine zuverlässige Quellenbestätigung ist weiterhin nicht nachgewiesen. Latenz und Modellqualität sind zusätzliche Einschränkungen dieser Konfiguration.
