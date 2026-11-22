import test from 'node:test';
import assert from 'node:assert/strict';
import { HudController } from '../src/hud.js';

function makeHud(mode = 'tabpfn-api') {
  const nodes = new Map();
  const get = (id) => {
    if (!nodes.has(id)) {
      nodes.set(id, {
        textContent: '',
        disabled: false,
        style: {},
        classList: {
          toggle(name, value) {
            this[name] = value;
          },
        },
      });
    }
    return nodes.get(id);
  };
  globalThis.document = { getElementById: get };
  const hud = Object.create(HudController.prototype);
  hud.modelMode = mode;
  hud.elMissionReport = get('mission-report');
  hud.elRiskForecastTime = get('risk-forecast-time');
  hud._disableOnCompleteEls = [get('btn-pause')];
  return { hud, nodes };
}

test('mission report shows measured duration and honest model score; reset hides it', () => {
  const { hud, nodes } = makeHud('tabpfn-api');
  hud.updateReport({
    elapsed_sim_seconds: 42.5,
    elapsed_wall_seconds: 55,
    measurement_count: 85,
    estimated_source: [1, 2, 3],
    model_score: 0.73,
    estimated_error_m: 2,
    ppm_at_detection: 30,
    estimate_age_seconds: 1,
  });
  assert.equal(nodes.get('mission-report').classList.hidden, false);
  assert.match(nodes.get('report-duration').textContent, /42.5 s/);
  assert.match(nodes.get('report-score').textContent, /73.0.*unkalibriert/);
  assert.equal(nodes.get('btn-pause').disabled, true);
  hud.updateReport(null);
  assert.equal(nodes.get('mission-report').classList.hidden, true);
  assert.equal(nodes.get('btn-pause').disabled, false);
});

test('missing or mock model score never becomes a fake certainty percentage', () => {
  const { hud, nodes } = makeHud('mock');
  hud.updateReport({
    elapsed_sim_seconds: 1,
    elapsed_wall_seconds: 1,
    measurement_count: 2,
    estimated_source: [0, 0, 3],
    model_score: null,
    estimated_error_m: 0,
    ppm_at_detection: 20,
    estimate_age_seconds: 0,
  });
  assert.match(nodes.get('report-score').textContent, /nicht verfügbar/);
  assert.match(nodes.get('report-evidence').textContent, /Mock-Fund/);
});

test('live status exposes compact inputs and measured local search phase', () => {
  const { hud, nodes } = makeHud('tabpfn-api');
  hud.updateBackendStatus({
    mode: 'tabpfn-api',
    state: 'ready',
    mission_state: 'searching',
    search_phase: 'track',
    source_features: 12,
    hazard_features: 20,
    completed_probes: 3,
    measurement_count: 22,
    inference_seconds: 0.25,
  });
  assert.match(nodes.get('mission-status').textContent, /Gaskontakt.*3D/);
  assert.match(nodes.get('model-status').textContent, /12 source \/ 20 risk inputs/);
  assert.match(nodes.get('model-metrics').textContent, /3 Messstopps/);
  hud.updateBackendStatus({
    mode: 'tabpfn-api',
    state: 'ready',
    mission_state: 'searching',
    search_phase: 'track',
    navigation_basis: 'tabpfn_source_prediction',
    prediction_decisions: 7,
  });
  assert.match(nodes.get('mission-status').textContent, /TabPFN bestimmt/);
  assert.match(nodes.get('model-metrics').textContent, /7 TabPFN-Ziele/);
  hud.updateBackendStatus({
    mode: 'tabpfn-api',
    state: 'ready',
    mission_state: 'searching',
    navigation_basis: 'wind_adapted_acquisition',
    adaptive_wind: true,
    wind_replans: 2,
  });
  assert.match(nodes.get('mission-status').textContent, /Wind geändert/);
  assert.match(nodes.get('model-metrics').textContent, /2 Windanpassungen/);
  hud.updateBackendStatus({
    mode: 'tabpfn-api',
    state: 'ready',
    plan_state: 'coverage_fallback',
    search_candidate_count: 578,
    plan_revision: 1,
  });
  assert.match(nodes.get('map-plan-status').textContent, /dataset pending.*578 cells/);
  hud.updateBackendStatus({
    mode: 'tabpfn-api',
    state: 'ready',
    plan_state: 'tabpfn_map',
  });
  assert.match(nodes.get('map-plan-status').textContent, /scored by TabPFN/);
});

test('cached hazard shows its actual forecast time and expiry', () => {
  const { hud, nodes } = makeHud('tabpfn-api');
  hud.updateBelief({ sim_time: 3, estimated_source: [1, 2, 3], hazard_forecast_time: 4.5 });
  assert.match(nodes.get('risk-forecast-time').textContent, /t=4.5 s$/);
  hud.updateRiskTime(5);
  assert.match(nodes.get('risk-forecast-time').textContent, /veraltet/);
});
