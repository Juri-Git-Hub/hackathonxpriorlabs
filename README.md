# PlumeSense · main-Simulation mit unserem TabPFN-Backend

Dieser Branch vereint die 3D-Simulation von `main` mit unserer RTX-4090-API, automatischem SSH-Tunnel, zwei TabPFN-Klassifikatoren (Quelle und zukünftige Gefahr) und Streamlit/Plotly-Demo.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m integration.generate --episodes 24 --seconds 40
python launch_demo.py
```

http://localhost:8501 öffnen und „Live laufen“ aktivieren. Die drei Ansichten zeigen Raffinerie, TabPFN-Quellenscores und Gefahr in vier Sekunden. Ground Truth ist standardmäßig verborgen. Jede Drohne hat ausschließlich ihren eigenen mobilen Gassensor; der TabPFN-Agent erhält keine Messwerte der Vergleichsdrohnen.

SSH-Alias `my4090` muss erreichbar sein; auf dessen Port 8000 läuft unsere vorhandene API aus `api/server.py`. `launch_demo.py` und `tunnel.py` öffnen bei Bedarf den Tunnel zu lokalem Port 18000. Konfiguration: `PLUMESENSE_API_URL`, `PLUMESENSE_SSH_HOST`, `PLUMESENSE_REMOTE_PORT`. Kein erneuter Modellabruf oder Token auf dem Laptop nötig. Die API mit den getrennten Kontexten `source` und `hazard` bleibt unverändert.

## Integration

- `sim/` und `shared/` stammen unverändert von main: 120×120×30 m, Z-up, turbulente Strömung, kontinuierliche Puff-Emission, Gebäude, Sensorträgheit/-rauschen und 20-Hz-Drohnendynamik.
- `agents/plumesense_agent.py` verbindet unsere beiden Klassifikatoren über den vorhandenen Predictor/API-Client mit `AsyncInferenceAgent`. Nur Features aus `DroneObservation` gehen in die Inferenz. Die Physik wartet nicht auf Netzwerk/GPU. Resets verwerfen alte Ergebnisse.
- `integration/features.py` verwendet 3D-Kandidaten, mobile Messhistorie, Wind einschließlich Fehlwertindikatoren, jüngste/stärkste Messungen und Distanzen. Fehlende Windwerte werden als 0 plus Indikator codiert. Temperatur wird nicht als Feature verwendet. Kein Zugriff auf tatsächliche Quellen oder Gaskonzentrationen außerhalb der Drohnenmessungen.
- `integration/generate.py` erzeugt neue Episoden mit mains echten Physik- und Sensormodulen und Gradient-/Raster-Suchflugwegen. Drei Viertel der Episoden bilden den Trainingskontext, ein Viertel bleibt für Evaluation getrennt. Quellenlabels: Kandidat innerhalb 6 m der tatsächlichen Quelle; Gefahrenlabels: ≥10 ppm am Kandidaten vier Sekunden später. Nur Offline-Labels dürfen die Wahrheit lesen. Die generierte CSV wird nicht versioniert.
- `integration/runtime.py` hält Physik und UI getrennt. `app.py` visualisiert die drei Drohnen und TabPFN-Ergebnisse. `GradientAgent` und `LawnmowerAgent` bleiben die Baselines aus main.
- `server/app.py` behält mains WebSocket-/Judge-Verträge und verwendet im echten Modus unseren API-Agenten. `--mock-agents` benötigt keine Daten oder GPU. Das ursprünglich vorbereitete lokale Regressionsmodell bleibt unter `agents/tabpfn_agent.py` erhalten.

Die Quellenschätzung stammt direkt aus TabPFN-Klassifikationsscores, nicht aus inverser Simulatorphysik. Messziele verbinden regionale Quellenscores, geometrische Erkundung und Entfernung. Die Drohnendynamik samt Hindernisabstoßung stammt von main; es wird kein 2D-A* auf eine 3D-Welt projiziert. Gerade bei starkem Wind sind erfolgreiche Navigation und zuverlässige Ortung noch nicht nachgewiesen.

## Evaluation

```bash
python -m pytest -q
python -m integration.evaluate
python -m server.app --mock-agents --port 8000
# Echter WebSocket-Modus mit unserem API-Agenten:
python -m server.app --port 8000
```

`reports/main3d_evaluation.json` enthält die erste echte 4090-Prüfung: Trainingsepisoden 0–17, Testepisoden 18–23, 896 Testzeilen. Quellen-AUROC ca. 0,582, F1 0; Gefahren-AUROC ca. 0,789, F1 ca. 0,121. Das sind Metriken auf vorselektierten Kandidaten, kein Nachweis einer erfolgreichen 3D-Lecksuche. Es werden keine 2D-Erfolgsquoten auf 3D übertragen. Der kleine Kontext liefert insbesondere noch keine zuverlässige Quellenlokalisierung.

Quellenkarten sind normierte Scores, keine kalibrierten Wahrscheinlichkeiten. Der übernommene WebSocket-Feldname `q50` enthält für unseren Adapter diese Quellenscores; `uncertainty` ist lediglich Klassifikationsmehrdeutigkeit, kein Regressionsquantil und keine epistemische Unsicherheit. Die Gefahrenkarte wird in der Streamlit-Demo aus separaten Agentendiagnosen dargestellt; der eingefrorene WebSocket-Vertrag bleibt unverändert.

Main bewertet `locked` und Abstände mit der tatsächlichen Quelle ausschließlich im Evaluator. Diese Werte gehen nicht an den Agenten. Eine Statusmeldung ist daher eine Simulator-Auswertung und keine eigenständig kalibrierte Erfolgsgarantie des Modells.

## Erhaltene Bestandteile

[README-main.md](README-main.md) dokumentiert die ursprünglichen main-Verträge und geplante Three.js-Oberfläche (auf main liegt bislang nur deren Anleitung). [README-plumesense-2d.md](README-plumesense-2d.md) beschreibt den bisherigen Stand. Die frühere Demo bleibt als `app_2d.py` mit ihren Datenpipelines und Auswertungen erhalten; diese 2D-Daten dürfen nicht für 3D verwendet werden. Der ursprüngliche API-Server, der SSH-Tunnel und alle alten Backend-Dateien bleiben erhalten.
