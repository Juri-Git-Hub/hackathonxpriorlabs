import test from 'node:test';
import assert from 'node:assert/strict';
import { HudController } from '../src/hud.js';

test('mission report shows measured duration and honest model score; reset hides it',()=>{
  const nodes=new Map();
  globalThis.document={getElementById(id){if(!nodes.has(id))nodes.set(id,{textContent:'',disabled:false,classList:{toggle(name,value){this[name]=value;}}});return nodes.get(id);}};
  const hud=Object.create(HudController.prototype);hud.modelMode='tabpfn-api';
  hud.updateReport({elapsed_sim_seconds:42.5,elapsed_wall_seconds:55,measurement_count:85,estimated_source:[1,2,3],model_score:.73,estimated_error_m:2,ppm_at_detection:30,estimate_age_seconds:1});
  assert.equal(nodes.get('mission-report').classList.hidden,false);
  assert.match(nodes.get('report-duration').textContent,/42.5 s/);
  assert.match(nodes.get('report-score').textContent,/73.0.*unkalibriert/);
  assert.equal(nodes.get('btn-pause').disabled,true);
  hud.updateReport(null);
  assert.equal(nodes.get('mission-report').classList.hidden,true);
  assert.equal(nodes.get('btn-pause').disabled,false);
});

test('missing or mock model score never becomes a fake certainty percentage',()=>{
  const nodes=new Map();
  globalThis.document={getElementById(id){if(!nodes.has(id))nodes.set(id,{classList:{toggle(){}},textContent:''});return nodes.get(id);}};
  const hud=Object.create(HudController.prototype);hud.modelMode='mock';
  hud.updateReport({elapsed_sim_seconds:1,elapsed_wall_seconds:1,measurement_count:2,estimated_source:[0,0,3],model_score:null,estimated_error_m:0,ppm_at_detection:20,estimate_age_seconds:0});
  assert.match(nodes.get('report-score').textContent,/nicht verfügbar/);
  assert.match(nodes.get('report-evidence').textContent,/Mock-Fund/);
});

test('live status exposes compact inputs and measured local search phase',()=>{
  const nodes=new Map();
  globalThis.document={getElementById(id){if(!nodes.has(id))nodes.set(id,{style:{},textContent:'',classList:{toggle(){}}});return nodes.get(id);}};
  const hud=Object.create(HudController.prototype);
  hud.updateBackendStatus({mode:'tabpfn-api',state:'ready',mission_state:'searching',search_phase:'track',source_features:12,hazard_features:20,completed_probes:3,measurement_count:22,inference_seconds:.25});
  assert.match(nodes.get('mission-status').textContent,/Gaskontakt.*3D/);
  assert.match(nodes.get('model-status').textContent,/12 source \/ 20 risk inputs/);
  assert.match(nodes.get('model-metrics').textContent,/3 Messstopps/);
});

test('cached hazard shows its actual forecast time and expiry',()=>{
  const nodes=new Map();
  globalThis.document={getElementById(id){if(!nodes.has(id))nodes.set(id,{textContent:''});return nodes.get(id);}};
  const hud=Object.create(HudController.prototype);
  hud.updateBelief({sim_time:3,estimated_source:[1,2,3],hazard_forecast_time:4.5});
  assert.match(nodes.get('risk-forecast-time').textContent,/t=4.5 s$/);
  hud.updateRiskTime(5);
  assert.match(nodes.get('risk-forecast-time').textContent,/veraltet/);
});
