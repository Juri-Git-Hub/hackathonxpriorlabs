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

`integration/navigation.py` plant echte 3D-A*-Flugwege mit geprüften Segmenten gegen Gebäude, horizontalen/vertikalen Flugzeitkosten und Zwischenhöhen bis 28 m. Routenplanung und GPU-Inferenz laufen in Hintergrund-Workern; der Physikthread folgt nur Wegpunkten. Volumetrischer Raster- und Gradient-Agent bleiben Vergleichsstrategien. Quellenmodell und Prognose beeinflussen die Suche; die Flugplanung meidet nicht pauschal Gas, da das Ziel Leckortung ist.

Trainingsdaten entstehen mit `integration.generate` aus separaten 3D-Episoden: Quellenlabel innerhalb von 6 m, Gefahrenlabel ≥10 ppm vier Sekunden später. 18 von 24 Episoden dienen als Modellkontext, sechs als Test. Die CSV ist reproduzierbar und nicht versioniert; nach Änderung der Physik erneut erzeugen. Die alten 2D-Backendmodule/Daten bleiben als Vergleich erhalten und werden nicht in die Live-3D-Inferenz eingespeist.

Quellenkarten sind normierte Rangfolgescores, keine kalibrierten Quellenwahrscheinlichkeiten. Für ältere Frontends enthält `q50` diese Scores; `uncertainty` ist Klassifikationsmehrdeutigkeit, keine epistemische Unsicherheit. `score_kind` macht den Unterschied explizit. Der Simulator bewertet `locked` mit Ground Truth; der Agent erhält diese Bewertungsdaten nicht. Eine zuverlässige Leckortung ist noch nicht nachgewiesen.

## Prüfung

```bash
npm --prefix frontend run build
python -m pytest -q
python -m integration.evaluate
```

Build sowie Live-Browserprüfung mit echter 4090 bestanden: Modellstatus ready, 3D-Szene und WebSocket aktiv, Pause/Einzelschritt (+0,5 s) und Layerwechsel geprüft. 43 funktionale Tests bestehen. Der geerbte Benchmark `test_headless_engine_speed_over_500_steps_per_sec` scheitert hier; er scheitert auch auf unverändertem main (ca. 386 Schritte/s im Vergleichslauf). Die Live-Physik benötigt 20 Hz. Der Benchmark wurde nicht abgeschwächt.

`reports/main3d_evaluation.json` enthält die aktuelle episodisch getrennte Modellprüfung und den echten API-Smoke-Test. Kandidatenmetriken sind keine Erfolgsquote der Lecksuche; insbesondere ist das Quellenmodell bislang noch schwach. Ältere Berichte gehören zu früheren Physik-/2D-Versionen.
