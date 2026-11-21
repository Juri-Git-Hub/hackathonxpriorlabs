# PlumeSense Backend + Three.js aus main

Die Oberfläche ist die Three.js-/Vite-Simulation aus dem aktuellen `main`. Unser Beitrag ist das Backend: zwei TabPFN-Modelle auf der RTX 4090, mobiler Gassensor, Historienmerkmale, 3D-Flugplanung, Live-Prognosen und WebSocket-Verträge. Streamlit, Plotly und die separaten Python-Oberflächen wurden entfernt.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m integration.generate --episodes 24 --seconds 40
python launch_demo.py
```

**http://localhost:8501** öffnen. Der Launcher baut das vorhandene Three.js-Frontend, startet bei Bedarf den SSH-Tunnel und serviert UI, `/ws` und `/health` auf demselben Port. Node/npm sind für den Build erforderlich; bei fehlenden node_modules wird `npm ci` ausgeführt. Port 8000 wird lokal nicht verwendet, da dort bereits ein anderer Dienst läuft.

```bash
# Ohne GPU/Trainingsdaten, expliziter Mock-Modus:
python launch_demo.py --mock-agents
# Anderer Port / Seed:
python launch_demo.py --port 8502 --seed 99
```

Unser bestehender API-Server `api/server.py` auf `my4090:8000` bleibt unverändert. `tunnel.py` verbindet localhost:18000 mit diesem Remote-Port. Konfiguration: `PLUMESENSE_API_URL`, `PLUMESENSE_SSH_HOST`, `PLUMESENSE_REMOTE_PORT`, `PLUMESENSE_AUTO_TUNNEL=0`. Der SSH-Alias benötigt den vorhandenen Schlüssel/Agenten und ProxyJump.

## Frontend-Vertrag

`shared/schemas.py` bleibt die Quelle der Pydantic-Verträge. Vorhandene main-Felder bleiben erhalten; neue Felder besitzen Defaults, damit ältere Frontends weiter lesen können.

- `world_init`: Gebäude, Anlagen und 788 XYZ-Kandidaten auf Höhen 3, 8, 13, 18, 23, 28 m.
- `sim_tick`: 20-Hz-Telemetrie, Drohnen, Wolke, Wind und neu `paused`.
- `belief_update`: Quellen-/Messzielposition sowie bisherige `q50`/`uncertainty`. Erweiterungen: `source_scores`, `hazard_probabilities`, `route`, `inference_seconds`, `measurement_count`, `exposure_ppm_seconds`, `score_kind`.
- `backend_status`: Modellmodus, `pending`/`ready`/`error`/`mock`, Fehlermeldung, Latenz, Messanzahl, Exposition und Pausenstatus; etwa einmal pro Sekunde sowie bei Verbindung/Kommandos.
- `judge_cmd`: vorhandene Wind-, Leck-, Dropout-, Pause- und Reset-Kommandos; zusätzlich `step` (0,5 Simulationssekunden, nur im pausierten Zustand) und optional `seed` für `reset_race`.

Der kleine Block oben links ergänzt die vorhandenen main-HUDs: Quellen-/Gefahrenansicht, 3D-Route, Seed, Einzelschritt und Live-Modellstatus. Die vorhandenen Judge Controls bleiben erhalten. Eine konfigurierbare Leckhöhe ergänzt das Platzieren per Bodenklick. Die Szene, Kamera, Raffinerie, Drohnen- und Wolkendarstellung stammen von main.

Python bleibt Z-up; nur beim Rendern wird in Three.js-Y-up konvertiert. Der Browser verwendet denselben Origin für `/ws` (inklusive wss unter HTTPS), alternativ `?ws=...`. Frontend-Entwicklung:

```bash
python -m server.app --port 8501
cd frontend
npm ci
npm run dev
```

Vite auf Port 5173 leitet `/ws` und `/health` zum Backend auf 8501 weiter. Das Backend liefert gebaute Frontend-Dateien aus `frontend/dist`; der Build wird nicht versioniert. `?mock=true` bleibt für die eigenständige Three.js-Entwicklung erhalten.

## Backend und Bewegung

Die neueste main-Physik (6-DOF, Rotor-/Motordynamik, ABL-Strömung und RK2-Wolke) ist integriert. Gas wird kontinuierlich emittiert. Jede Drohne besitzt einen eigenen mobilen Gassensor; der TabPFN-Agent erhält nur seine eigene Messhistorie, Position und gemessenen Wind.

`agents/plumesense_agent.py` verwendet die unveränderte 4090-API über unseren Predictor mit getrennten Kontexten `source` und `hazard`. Beobachtungen werden in `integration/features.py` zusammengefasst. Keine tatsächliche Quelle und keine zukünftigen Messwerte fließen in X ein. Nur Offline-Labels und die Simulator-Auswertung dürfen Ground Truth lesen.

`integration/navigation.py` plant echte 3D-A*-Flugwege mit geprüften Segmenten gegen Gebäude, horizontalen/vertikalen Flugzeitkosten und Zwischenhöhen bis 28 m. Routenplanung und GPU-Inferenz laufen in Hintergrund-Workern; der Physikthread folgt nur Wegpunkten. In Live-Demo und WebSocket-Server fliegt ausschließlich die TabPFN-Drohne. Raster- und Gradient-Strategien bleiben nur für Offline-Datengenerierung und Tests erhalten. Quellenmodell und Prognose beeinflussen die Suche; die Flugplanung meidet nicht pauschal Gas, da das Ziel Leckortung ist.

Trainingsdaten entstehen mit `integration.generate` aus separaten 3D-Episoden: Quellenlabel innerhalb von 6 m, Gefahrenlabel ≥10 ppm vier Sekunden später. 18 von 24 Episoden dienen als Modellkontext, sechs als Test. Die CSV ist reproduzierbar und nicht versioniert; nach Änderung der Physik erneut erzeugen. Die alten 2D-Backendmodule/Daten bleiben als Vergleich erhalten und werden nicht in die Live-3D-Inferenz eingespeist.

Quellenkarten sind normierte Rangfolgescores, keine kalibrierten Quellenwahrscheinlichkeiten. Für ältere Frontends enthält `q50` diese Scores; `uncertainty` ist Klassifikationsmehrdeutigkeit, keine epistemische Unsicherheit. `score_kind` macht den Unterschied explizit. Der Simulator bewertet `locked` mit Ground Truth; der Agent erhält diese Bewertungsdaten nicht. Eine zuverlässige Leckortung ist noch nicht nachgewiesen.

## Prüfung

```bash
npm --prefix frontend run build
python -m pytest -q
python -m integration.evaluate
```

Build sowie Live-Browserprüfung mit echter 4090 bestanden: Modellstatus ready, 3D-Szene und WebSocket aktiv, Pause/Einzelschritt (+0,5 s) und Layerwechsel geprüft. 57 funktionale Tests bestehen. Der geerbte Benchmark `test_headless_engine_speed_over_500_steps_per_sec` scheitert hier; er scheitert auch auf unverändertem main (ca. 386 Schritte/s im Vergleichslauf). Die Live-Physik benötigt 20 Hz. Der Benchmark wurde nicht abgeschwächt.

`reports/main3d_evaluation.json` enthält die aktuelle episodisch getrennte Modellprüfung und den echten API-Smoke-Test. Kandidatenmetriken sind keine Erfolgsquote der Lecksuche; insbesondere ist das Quellenmodell bislang noch schwach. Ältere Berichte gehören zu früheren Physik-/2D-Versionen.


Das HUD meldet `mission_state` (`searching`, `paused`, `no_leak`, `depleted`), `active_leaks` und `battery_pct` aus `backend_status`. Diese Simulator-/Betriebsdiagnose dient nur der Anzeige, nicht als Eingabe an TabPFN. Bei leerem Akku werden keine neuen Messungen mehr gesammelt; ohne aktive Leckquelle gibt es nichts zu lokalisieren. Ein Neustart stellt Standardleck und vollen Akku wieder her. Die Quellenmodellqualität ist davon unabhängig weiterhin unzureichend für zuverlässige Ortung.

## Fundabschluss und Bericht

Die Simulation beendet die TabPFN-Suche und pausiert automatisch, wenn **zwei Sekunden durchgehend** folgende Bedingungen erfüllt sind: Drohne innerhalb von 5 m um ein tatsächliches Leck, frische TabPFN-Quellenschätzung (höchstens 5 s alt) ebenfalls innerhalb 5 m und lokaler Sensorwert >15 ppm. Ein hoher Gaswert ohne Quellenschätzung reicht nicht. Die echte Leckposition dient ausschließlich diesem separat gekennzeichneten Simulator-Judge und geht nicht in die Agentenfeatures ein.

`mission_report` wird in `sim_tick` und `backend_status` übertragen; `mission_state` wird `found`. Der Bericht enthält Simulationssuchdauer, Echtzeit seit Reset (inklusive Pausen), XYZ-Schätzung, Modellklassifikationsscore, Messanzahl, Gaswert, Alter der Schätzung und den Simulator-Ortungsfehler. Der Score ist **unkalibriert und keine Wahrscheinlichkeit einer korrekten Ortung**. Ohne Score, etwa im Mock-Modus, wird keine scheinbare Sicherheit ausgegeben. Der Fund ist vom Simulator bestätigt, nicht autonom statistisch verifiziert.

Nach Abschluss kann die Mission nicht versehentlich über Pause/Einzelschritt fortgesetzt werden. Bericht und Simulation bleiben bis `reset_race` eingefroren; ein erneut verbundener Browser erhält den Bericht ebenfalls. „Neue Suche starten“ setzt ihn zurück. Die Modellqualität und die Wahrscheinlichkeit, überhaupt einen Fund zu erreichen, werden durch diese Abschlussfunktion nicht verbessert.

Tests: `python -m pytest tests/test_mission_completion.py tests/test_ws_contract.py -q` und `node --test frontend/test/hud.test.js` prüfen Stopp, Bericht, Reset, Reconnect und ehrliche Scoreanzeige.

## Großer V3-Datensatz: erzeugen, prüfen, benchmarken

```bash
# 512 unabhängige 120-s-Episoden, parallel; benötigt pyarrow.
python -m integration.dataset --episodes 512 --seconds 120 --workers 12
# Tiefenprüfung einschließlich Feature-Neuberechnung aus vergangenen Rohmessungen:
python -m integration.verify_dataset data/datasets/main3d-v3
# Vollständige 3D-Raster statt nur vorausgewählter Kandidaten:
python -m integration.benchmark_dataset data/datasets/main3d-v3 --context-rows 8192 --episodes 16 --tabpfn
```

Ausgabe: `data/datasets/main3d-v3/` mit `manifest.json`, `config.json`, `verification.json`, `benchmark.json` und einzelnen Episodenordnern. Pro Episode: komprimierte Parquet-Kandidaten, rohe Sensorhistorie als JSONL.gz und Metadaten mit Quellenparametern, Dateihashes und Codefingerprint. Gelegentliche vollständige Raster liegen separat unter `full_grid.parquet`. Datendateien sind nicht versioniert; kleine Auswertungsberichte liegen unter `reports/`.

Der Generator variiert Quellenposition/-höhe, Emission, Auftrieb, Wind und Windwechsel, Sensorträgheit, Rauschen, Dropouts und die Suchstrategie. Raster, Gradient, geometrische Flächenabdeckung und gemischte Strategien verwenden nur eigene Beobachtungen. Die reale 3D-Physik und der verrauschte mobile Sensor bleiben maßgeblich. 60 % der Trainingsepisoden dürfen als **Offline-Curriculum** gezielt in einer vorhandenen Wolke starten. Dafür wird ausschließlich beim Datenerzeugen Ground Truth benutzt. **Validierung und Test starten unabhängig von Quelle und Gaskonzentration.** Das Curriculum wird als Metadatum dokumentiert und nicht als Feature übergeben.

Die Aufteilung beträgt 60 % Training, 20 % Validierung, 20 % Test auf Episodenebene. Die Anlagengeometrie ist weiterhin dieselbe: Dies ist kein Nachweis für unbekannte reale Anlagen oder vollständig unbekannte Karten. Der Generator nimmt positive, nahe negative und zufällige entfernte Kandidaten für den Trainingspool auf; deren Klassenverteilung entspricht deshalb nicht der gesamten Karte. Der Benchmark benutzt vollständige Raster aus vorausbestimmten Testepisoden, ohne anhand von Gaskontakt oder Modellfehlern Episoden auszusortieren.

V3 hat 159 Beobachtungsmerkmale. Zusätzlich zur bisherigen Historie sind gemessener Wind und Fehlwertindikatoren an starken/jüngsten Messungen, Temperaturmerkmale, 5-/15-/30-s-Fenster, vertikale/räumliche Abdeckung, lokale historische Messwerte und Geschwindigkeit enthalten. Alle Gasbeobachtungen tragen zu den Zusammenfassungen bei. Es handelt sich weiterhin um eine feste tabellarische Zusammenfassung, nicht um eine verlustfreie Rohzeitreihe.

Die Verifikation prüft vollständige Episoden, Hashes, Featureschema, endliche Werte, verbotene Features, Zeitreihenfolge, eindeutige Rasterabfragen, 4-s-Horizont und Quellen-/Gefahrenlabels. Features werden aus dem jeweiligen **vergangenen** Rohmessungspräfix erneut berechnet. Dies kann spätere oder erfundene Featurewerte entdecken. Die Gefahrenlabels werden gegen gespeicherte zukünftige Simulatorwerte geprüft; kein unabhängiges CFD-Modell wird als Referenz behauptet. Separate Abdeckungsgates prüfen Episodenzahl, Gaskontakt, Suchphasen, Höhenvariation und positive Labels. Bestandene Datenprüfungen garantieren keine erfolgreiche Leckortung.

Fortsetzen: denselben Generatorbefehl mit demselben Ausgabeordner erneut aufrufen. Fertige Episoden werden wiederverwendet; die Workerzahl darf geändert werden. Andere Parameter oder ein anderer relevanter Codefingerprint werden im selben Ordner abgelehnt. Parquet-Inhalte und dekomprimierte Rohhistorien sind deterministisch; gzip-Container können unterschiedliche Header-Zeitstempel haben.

Der Benchmark vergleicht konstanten Prior, ExtraTrees und optional echte TabPFN-Inferenz über die 4090. Beide lernenden Modelle bekommen denselben begrenzten, über Trainingsepisoden verteilten Kontext. Eine große Datenbibliothek wird nicht vollständig in einen einzelnen TabPFN-Fit gepresst. F1-Schwellen werden ausschließlich auf Validierung gewählt; Testdaten werden dafür nicht verwendet. Berichtet werden F1, AUROC, Brier, Latenz sowie vollständige Quellen-Rangfolge, Ortungsfehler und Treffer innerhalb 6 m – separat mit/ohne vorherigen Gaskontakt. Konfidenzintervalle werden nach **Episoden**, nicht nach einzelnen korrelierten Rasterzeilen, gebootstrapped. Dies ist ein Prognosebenchmark und keine Flug-/Fund-Erfolgsquote.

Der bisherige Live-Kontext bleibt unverändert. Nach bestandener Prüfung und Bewertung kann V3 ausdrücklich aktiviert werden:

```bash
PLUMESENSE_DATASET="$PWD/data/datasets/main3d-v3" python launch_demo.py
```

Der Loader verlangt passende Featureversion und bestandene Integritäts-/Abdeckungsgates, verwendet ausschließlich Trainingsshards und maximal 8192 Kontextzeilen pro Modell. Validierung und Test werden nie als Modellkontext geladen. Ein neues Dataset wird nicht allein wegen seiner Größe automatisch als besseres Live-Modell eingesetzt.

### Erzeugter und geprüfter V3-Lauf

- 512 Episoden × 120 s, 12.288 Historien-Snapshots, **811.032 Kandidatenzeilen**, 159 Features, ca. 176 MB einschließlich Rohhistorien und Vollrasterdateien.
- Training: 307 Episoden, davon 253 mit Gaskontakt (82,4 %); Validierung: 102, davon 65 mit Kontakt; Test: 103, davon 70 mit Kontakt. Alle Test-/Validierungsstarts sind quellenunabhängig. Die Abdeckungs-/Integritätsgates und die Neuberechnung aus vergangenen Messpräfixen sind bestanden.
- Trainingsphasen: 2.115 kalte Such-Snapshots, 1.114 mit aktuellem Gaskontakt, 4.139 nach Verlust des Gaskontakts.
- Benchmark: je 16 Validierungs-/Testepisoden mit je drei vollständigen Rastern; **37.824 Testabfragen** aus 48 Momentaufnahmen. Der vorab festgelegte Vollraster-Kohort umfasst gerade Episoden-IDs: Default-Eckstart, Mixed-/Gradient-Strategien. Der Bericht behauptet keine Repräsentativität für beliebige Start-/Strategieverteilungen.
- TabPFN-Gefahr: **F1 0,345, AUROC 0,808, Brier 0,0601**; ExtraTrees: F1 0,312, AUROC 0,748, Brier 0,0640. Der auf Validierung gewählte F1-Schwellenwert ist 0,44. Der reine Validierungsprior erzielt Brier 0,0466; höhere AUROC/F1 bedeuten deshalb nicht bessere Kalibrierung.
- TabPFN-Quelle: **49,1 m mittlerer Ortungsfehler und 0/48 Treffer innerhalb 6 m**. ExtraTrees: 44,0 m und 3/48. Die Quellenlokalisierung bleibt trotz der deutlich informativeren Daten unzureichend. Es wurde kein erfolgreicher autonomer Suchlauf nachgewiesen und die Live-Demo nicht auf V3 umgestellt.

Ergebnisse: `reports/dataset_v3_summary.json`, `dataset_v3_benchmark.json`, `dataset_v3_verification.json`, `dataset_v3_manifest.json`, `dataset_v3_environment.json`. Sie dokumentieren auch den begrenzten Trainingskontext, Vorhersagelatenzen und episodisches Bootstrap. Eine Datenprüfung ist kein Modellgütesiegel.

Tiefenprüfung lässt sich mit `--workers 8` parallelisieren. `--skip-recompute` prüft weiterhin Schema/Hashes/Labels, ist aber ausdrücklich keine vollständige Prüfung der Feature-Historie. Für den großen Lauf wurde die vollständige Neuberechnung durchgeführt. Zusatztests unter `tests/test_dataset_v3.py` prüfen Reproduzierbarkeit, Fortsetzen, Testtrennung, absichtliche Dateibeschädigung, gefälschte Featurewerte, parallele Prüfung und den Schutz vor Aktivierung eines Datensatzes mit fehlgeschlagenen Abdeckungsgates.
