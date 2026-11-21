# PlumeSense Backend + Three.js aus main

Die Oberfläche ist die Three.js-/Vite-Simulation aus dem aktuellen `main`. Unser Beitrag ist das Backend: zwei TabPFN-Modelle auf der RTX 4090, mobiler Gassensor, Historienmerkmale, 3D-Flugplanung, Live-Prognosen und WebSocket-Verträge. Streamlit, Plotly und die separaten Python-Oberflächen wurden entfernt.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Nur falls der geprüfte V3-Datensatz noch nicht vorhanden ist:
python -m integration.dataset --episodes 512 --seconds 120 --workers 12
python -m integration.verify_dataset data/datasets/main3d-v3 --workers 8
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
- `backend_status`: Modellmodus, `pending`/`ready`/`error`/`mock`, Fehlermeldung, Latenz, Messanzahl, Exposition und Pausenstatus; außerdem Suchphase, abgeschlossene Messstopps und aktive Featureanzahl. Etwa einmal pro Sekunde sowie bei Verbindung/Kommandos.
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

`agents/plumesense_agent.py` verwendet die unveränderte 4090-API über unseren Predictor mit getrennten Kontexten `source` und `hazard`. Der Standard ist jetzt das validierte kompakte Profil aus `integration/reduced_features.py`: **12 Eingaben für die Quelle, 20 für die Gefahr, jeweils 4096 Kontextzeilen ausschließlich aus V3-Trainingsepisoden**. Quelleninferenz erfolgt alle 3 Simulationssekunden, Gefahreninferenz alle 10 Sekunden. Gecachte Gefahrenkarten behalten ihren tatsächlichen Mess-/Prognosezeitpunkt (`hazard_observed_at`, `hazard_forecast_time`); das HUD kennzeichnet abgelaufene Prognosen. Beide Modelle bleiben echte TabPFN-Klassifikatoren. Keine tatsächliche Quelle und keine zukünftigen Messwerte fließen in X ein. Nur Offline-Labels und die Simulator-Auswertung dürfen Ground Truth lesen.

Die zwölf Quelleneingaben beschreiben Abstände zur stärksten eigenen Gasmessung entlang/quer zum gemessenen Wind, Höhenunterschied, Windgeschwindigkeit, aktuelle/stärkste Gaswerte, nächste historische Messung, zeitgewichtete Interpolation, lokale Gaswerte und die Änderung über fünf Sekunden. Absolute Koordinaten, Uhrzeit, Anlagenalter, Temperatur und redundante Historienblöcke entfallen für dieses Modell. Der schnelle Builder wurde gegen die gespeicherten 159 Trainingsfeatures auf exakte Übereinstimmung der verwendeten Werte getestet. Es werden weiterhin alle vergangenen Messungen für lokale/historische Zusammenfassungen berücksichtigt; TabPFN erhält keine verlustfreie Rohzeitreihe.

`integration/navigation.py` plant echte 3D-A*-Flugwege mit geprüften Segmenten gegen Gebäude, horizontalen/vertikalen Flugzeitkosten und Zwischenhöhen bis 28 m. Routenplanung und GPU-Inferenz laufen in Hintergrund-Workern; der Physikthread folgt nur Wegpunkten. In Live-Demo und WebSocket-Server fliegt ausschließlich die TabPFN-Drohne. Raster- und Gradient-Strategien bleiben für Offline-Datengenerierung und Tests erhalten.

`integration/search_control.py` sucht zuerst nach Gaskontakt an Messstellen im Abwind bekannter Anlagen, bei ausbleibendem Kontakt später im gesamten XYZ-Volumen. Anschließend testet die Drohne kurze lokale 3D-Messflüge, zuerst in Richtung der TabPFN-Schätzung. Sie wartet an den Messstellen auf Sensorberuhigung und vergleicht gemessene Konzentrationen. Auf-/Abstieg und Rückkehr zu bereits besuchten Positionen sind ausdrücklich erlaubt. Gas ist kein pauschal zu meidendes Hindernis. Die Quellenentscheidung kommt aus TabPFN; die kontinuierliche Bewegung und Messplanung kommen vom separaten Controller.

**Quellenhypothesen und fliegbare Wegpunkte sind getrennt.** Das alte Grobraster hatte beim Standardleck einen minimal möglichen Schätzfehler von 7,17 m und konnte die unveränderte 5-m-Fundbedingung nie erfüllen. Jetzt ergänzt das Backend das Displayraster um alle bekannten Anlagenpositionen und nach Gaskontakt feinere windorientierte XYZ-Kandidaten. Die echte Leckposition wird dabei nicht abgefragt. Eine Quelle darf an einer Anlagenwand liegen; der Drohnenmesspunkt hält weiterhin Kollisionsabstand. Die Displayarrays bleiben kompatibel zum bisherigen Grobraster.

Trainingslabels: Quelle innerhalb von 6 m, Gefahr ≥10 ppm vier Sekunden später. Der große V3-Datensatz und seine Aufteilung sind unten dokumentiert. `integration.generate` und die alten 2D-Daten bleiben für historische Vergleiche erhalten und werden nicht in die Live-3D-Inferenz eingespeist.

Quellenkarten sind normierte Rangfolgescores, keine kalibrierten Quellenwahrscheinlichkeiten. Für ältere Frontends enthält `q50` diese Scores; `uncertainty` ist Klassifikationsmehrdeutigkeit, keine epistemische Unsicherheit. `score_kind` macht den Unterschied explizit. Der Simulator bewertet `locked` mit Ground Truth; der Agent erhält diese Bewertungsdaten nicht. Erfolgreiche Funde sind unten nachgewiesen; robuste Ortung für beliebige Lecks ist weiterhin nicht erreicht.

## Prüfung

```bash
npm --prefix frontend run build
python -m pytest -q
node --test frontend/test/*.test.js
# Vorhersagemetriken des tatsächlich eingesetzten 12/20-Feature-Profils:
python -m integration.evaluate
# Echte asynchrone Live-Architektur, dauert bis zu drei Minuten:
python -m integration.closed_loop --realtime --seed 42 --output reports/realtime_search_validation.json
# Vorab festgelegte neue Testflüge; gemessene API-Latenz als Simulationsverzögerung:
python -m integration.closed_loop --suite --output reports/final_closed_loop_tests.json
# Gleicher Controller ohne TabPFN bzw. ohne feine Quellenkandidaten:
python -m integration.closed_loop --sensor-only --seed 42 --output reports/development_sensor_search.json
python -m integration.closed_loop --coarse-only --seed 42 --output reports/validation_coarse_grid_ablation.json
# Featureanzahl und gezielte Featuregruppen nur auf Validierung vergleichen:
python -m integration.feature_ablation
python -m integration.feature_ablation --groups
```

Der geerbte Benchmark `test_headless_engine_speed_over_500_steps_per_sec` scheitert auf dieser Maschine auch auf unverändertem main (ca. 386 Schritte/s im früheren Vergleichslauf). Die Live-Physik benötigt 20 Hz. Der Benchmark wurde nicht abgeschwächt; die funktionale Suite kann separat mit `-k 'not test_headless_engine_speed_over_500_steps_per_sec'` ausgeführt werden.

Aktuelle Nachweise: `reports/compact_profile_predictions.json` (episodisch getrennte Vollraster-Prognosen), `realtime_search_validation.json` (asynchroner Echtzeitfund), `final_closed_loop_tests.json` (alle sieben Testflüge) und `validation_feature*.json` (Ablationen). Ältere Berichte einschließlich `main3d_evaluation.json` gehören zu früheren Profilen.

Das kompakte Profil besteht **69 funktionale Python-Tests und vier Frontend-Tests**; der Three.js-Produktionsbuild besteht ebenfalls. `reports/search_optimization_summary.json` bündelt Konfiguration, Codehashes, Einflussanalyse und Ergebnisverweise. Tests prüfen unter anderem Train/Live-Featureparität, windrelative Invarianz, fehlende Windwerte, feinere Quellenhypothesen, Sensorberuhigung, 3D-Messflüge, unveränderten Fund-Judge, asynchronen Reset und Zeitbezug gecachter Gefahrenprognosen.


Das HUD meldet `mission_state` (`searching`, `paused`, `no_leak`, `depleted`, `found`), lokale Suchphase, `active_leaks` und `battery_pct`. Diese Simulator-/Betriebsdiagnose dient nur der Anzeige. Bei leerem Akku werden keine neuen Messungen mehr gesammelt; ohne aktive Leckquelle gibt es nichts zu lokalisieren. Ein Neustart stellt Standardleck und vollen Akku wieder her.

### Gemessener Einfluss und tatsächliche Suchflüge

- Gleicher Trainingskontext und acht Validierungsepisoden: Quelle mit 12 Features findet 4/7 Momentaufnahmen mit vorherigem Gaskontakt innerhalb von 5 m, ebenso das Modell mit 159 Features. Die gemessene mittlere Anfragezeit sank von ca. 1,80 s auf 0,25–0,28 s. Das ist ein kleiner Validierungsausschnitt, kein allgemeiner Genauigkeitsnachweis.
- Beim gezielten Entfernen von Windgeometrie fällt dies auf 3/7, ohne lokale Messevidenz auf 2/7. Mehr Features oder Temperatur/Anlagenalter zeigten hier keinen verlässlichen Mehrwert. Die 20 Gefahrenfeatures haben eine kleine Prognoseeinbuße gegenüber 159, benötigen in der Validierung aber nur ca. 0,34 statt 2,32 s je Anfrage.
- Seed 42, gleicher lokaler Controller, jeweils 180 s: ohne TabPFN kein Fund; mit TabPFN, aber nur Grobraster ebenfalls kein Fund. Mit TabPFN plus feinen Quellenkandidaten: Fund nach **134 s**, Fehler **2,38 m**. Dies sind Entwicklungs-/Validierungsläufe; gemessene API-Latenzen wurden als Verzögerung in Simulationszeit angewandt.
- Die **echte asynchrone** 20-Hz-Architektur mit beiden Modellen fand das Standardleck bei Seed 42 nach **175,5 s Simulationszeit / 177,1 s Echtzeit**, Fehler **1,33 m**. Messwert vor Ort: 726,5 ppm; die Drohne war 2,95 m vom Leck entfernt. Gleiche Fundbedingung wie zuvor, kein Mock und keine gelockerten Grenzwerte.
- Die Three.js-Demo fand das Leck zunächst nach **65 s**, Fehler **3,71 m**. Nach dem letzten Neustart mit vollständiger Prognosezeit-Anzeige: **90 s**, Fehler **2,59 m**. Sie zeigt den eingefrorenen Fundbericht. `reports/live_browser_detection.json` speichert beide echten WebSocket-Ergebnisse. Async-Suchzeiten variieren durch Thread-/Netzwerktaktung; diese Läufe sind keine Erfolgsstatistik.
- Vorab eingefrorene neue Testfälle (Seeds 101–107): **5/7 Funde innerhalb von 180 s**. Erfolge: Standardleck Seeds 101/103 (29,5/51 s), anderes Ventil (21,5 s), anderer Tankanschluss (77 s), frei platzierte Quelle (82 s). Fehlschläge werden vollständig beibehalten: Standardleck Seed 102 (Gasfahne, nächster Drohnenabstand 10,4 m) und erhöhte Quelle z=13 m (Drohne kam nahe, aber keine bestätigte frische Quellenschätzung). Das Testprofil wurde danach nicht anhand dieser Fälle optimiert.

Die Testflüge nutzen dieselbe Karte mit neuen Simulationsseeds, nicht unbekannte Anlagen. Im schnelleren Headless-Harness wird die gemessene RPC-Zeit auf die simulierte Uhr angewandt; ein solcher Lauf ist nicht identisch zur echten Thread-/Netzwerktaktung. Der separate Echtzeitnachweis enthält diese Taktung. Nur der Simulator-Judge bestätigt Funde; ein autonom kalibriertes statistisches Abbruchkriterium ist noch nicht vorhanden.

Der getrennte Vollrastertest des aktuellen Profils auf 16 Testepisoden (37.824 Abfragen, 48 Momentaufnahmen) ergibt für die Gefahr **F1 0,317, AUROC 0,745, Brier 0,0600**, ca. 0,31 s je 4096 Abfragen. Der F1-Schwellenwert 0,36 stammt nur aus Validierung. Die IDW-Beobachtungsbaseline hat F1 0,280, AUROC 0,779 und Brier 0,0916: TabPFN gewinnt hier nicht bei jeder Metrik. Die reine globale Quellenrangfolge bleibt schwach: 47,4 m mittlerer Fehler und 3/48 Treffer innerhalb von 5 m. Erfolgreiche lokale Suche ist deshalb kein Nachweis einer verlässlichen globalen Quellenkarte. Die Verbesserungen entstehen aus feiner Quellenauflösung und neuen lokalen Messungen zusammen mit der kompakten Modellprognose.

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

Der Live-Loader verwendet jetzt standardmäßig den geprüften V3-Ordner. Ein anderer geprüfter V3-Ordner kann explizit gewählt werden:

```bash
PLUMESENSE_DATASET="$PWD/data/datasets/main3d-v3" python launch_demo.py
```

Der Loader verlangt passende Featureversion und bestandene Integritäts-/Abdeckungsgates, verwendet ausschließlich Trainingsshards und im validierten Profil **4096 Kontextzeilen pro Modell**. Validierung und Test werden nie als Modellkontext geladen. Ein neues Dataset wird nicht allein wegen seiner Größe automatisch als besseres Live-Modell eingesetzt.

### Erzeugter V3-Lauf und historischer Benchmark mit 159 Features

- 512 Episoden × 120 s, 12.288 Historien-Snapshots, **811.032 Kandidatenzeilen**, 159 Features, ca. 176 MB einschließlich Rohhistorien und Vollrasterdateien.
- Training: 307 Episoden, davon 253 mit Gaskontakt (82,4 %); Validierung: 102, davon 65 mit Kontakt; Test: 103, davon 70 mit Kontakt. Alle Test-/Validierungsstarts sind quellenunabhängig. Die Abdeckungs-/Integritätsgates und die Neuberechnung aus vergangenen Messpräfixen sind bestanden.
- Trainingsphasen: 2.115 kalte Such-Snapshots, 1.114 mit aktuellem Gaskontakt, 4.139 nach Verlust des Gaskontakts.
- Benchmark: je 16 Validierungs-/Testepisoden mit je drei vollständigen Rastern; **37.824 Testabfragen** aus 48 Momentaufnahmen. Der vorab festgelegte Vollraster-Kohort umfasst gerade Episoden-IDs: Default-Eckstart, Mixed-/Gradient-Strategien. Der Bericht behauptet keine Repräsentativität für beliebige Start-/Strategieverteilungen.
- TabPFN-Gefahr: **F1 0,345, AUROC 0,808, Brier 0,0601**; ExtraTrees: F1 0,312, AUROC 0,748, Brier 0,0640. Der auf Validierung gewählte F1-Schwellenwert ist 0,44. Der reine Validierungsprior erzielt Brier 0,0466; höhere AUROC/F1 bedeuten deshalb nicht bessere Kalibrierung.
- TabPFN-Quelle im damaligen 159-Feature-/8192-Kontext-Benchmark: **49,1 m mittlerer Ortungsfehler und 0/48 Treffer innerhalb 6 m**. ExtraTrees: 44,0 m und 3/48. Diese isolierte Grobrasterprognose war unzureichend. Die oben dokumentierte kompakte Live-Suche wurde anschließend separat mit aktiven Messflügen und feinen Quellenhypothesen geprüft.

Ergebnisse: `reports/dataset_v3_summary.json`, `dataset_v3_benchmark.json`, `dataset_v3_verification.json`, `dataset_v3_manifest.json`, `dataset_v3_environment.json`. Sie dokumentieren auch den begrenzten Trainingskontext, Vorhersagelatenzen und episodisches Bootstrap. Eine Datenprüfung ist kein Modellgütesiegel.

Tiefenprüfung lässt sich mit `--workers 8` parallelisieren. `--skip-recompute` prüft weiterhin Schema/Hashes/Labels, ist aber ausdrücklich keine vollständige Prüfung der Feature-Historie. Für den großen Lauf wurde die vollständige Neuberechnung durchgeführt. Zusatztests unter `tests/test_dataset_v3.py` prüfen Reproduzierbarkeit, Fortsetzen, Testtrennung, absichtliche Dateibeschädigung, gefälschte Featurewerte, parallele Prüfung und den Schutz vor Aktivierung eines Datensatzes mit fehlgeschlagenen Abdeckungsgates.
