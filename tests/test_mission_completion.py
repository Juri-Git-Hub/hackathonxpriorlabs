import numpy as np
from agents.mock_agent import MockTabPFNAgent
from sim.engine import SimEngine
from shared.schemas import BeliefUpdatePayload,JudgeCommand,LeakSource


def fixture():
    e=SimEngine({'tabpfn':MockTabPFNAgent('tabpfn')})
    p=e.leaks[0].pos;d=e.drones['tabpfn'];d.pos=np.array(p);d.samples_taken=20
    b=BeliefUpdatePayload(sim_time=1,drone_id='tabpfn',q50=[1],uncertainty=[0],estimated_source=p,next_waypoint=p,source_model_score=.73,regional_score_mass=.4)
    e._latest_agent_beliefs['tabpfn']=b
    return e,b

def test_completion_stops_and_freezes_report_until_reset():
    e,b=fixture()
    steps=round(e.LOCK_SUSTAIN_S/e.OBS_INTERVAL)
    for i in range(steps):e.sim_time=1+i*e.OBS_INTERVAL;e._check_completion('tabpfn',30)
    expected_t=1+(steps-1)*e.OBS_INTERVAL
    assert e.paused and e.drones['tabpfn'].status=='locked'
    r=e.mission_report
    assert np.isclose(r.elapsed_sim_seconds,expected_t) and r.model_score==.73 and r.simulator_confirmed
    p=e.drones['tabpfn'].pos.copy();e.step()
    assert np.array_equal(e.drones['tabpfn'].pos,p) and np.isclose(e.sim_time,expected_t)
    e.apply_command(JudgeCommand(action='pause'));e.apply_command(JudgeCommand(action='step'))
    assert e.paused and e.mission_report is r and np.isclose(e.sim_time,expected_t)
    e.apply_command(JudgeCommand(action='reset_race'))
    assert e.mission_report is None and not e.paused and e._latest_agent_beliefs=={}

def test_needs_fresh_model_estimate_and_tolerates_brief_gas_dips():
    e,b=fixture();e.sim_time=1;e._latest_agent_beliefs.clear();e._check_completion('tabpfn',100)
    assert e.mission_report is None and e._lock_timers['tabpfn']==0
    e._latest_agent_beliefs['tabpfn']=b
    e.sim_time=8;e._check_completion('tabpfn',100)
    assert e._lock_timers['tabpfn']==0
    e.sim_time=1;e._check_completion('tabpfn',100)
    assert e._lock_timers['tabpfn'] == e.OBS_INTERVAL
    e._check_completion('tabpfn',100)
    e._check_completion('tabpfn',.5)
    assert e._lock_timers['tabpfn'] == e.OBS_INTERVAL
    e._check_completion('tabpfn',.5)
    assert e._lock_timers['tabpfn']==0

def test_wrong_model_location_does_not_complete_at_true_source():
    e,b=fixture();e.sim_time=1
    e._latest_agent_beliefs['tabpfn']=b.model_copy(update={'estimated_source':(-50,50,28)})
    for _ in range(8):e._check_completion('tabpfn',100)
    assert e.mission_report is None



def test_estimate_and_drone_must_match_the_same_leak():
    engine, belief = fixture()
    drone_pos = engine.leaks[0].pos
    other_pos = (drone_pos[0], drone_pos[1], drone_pos[2] + 12)
    engine.leaks.append(LeakSource(id="other_leak", pos=other_pos))
    engine.sim_time = belief.sim_time
    engine._latest_agent_beliefs["tabpfn"] = belief.model_copy(
        update={"estimated_source": other_pos}
    )
    for _ in range(round(engine.LOCK_SUSTAIN_S / engine.OBS_INTERVAL)):
        engine._check_completion("tabpfn", 30)
    assert engine.mission_report is None
    assert engine._lock_timers["tabpfn"] == 0

    engine._latest_agent_beliefs["tabpfn"] = belief
    for _ in range(round(engine.LOCK_SUSTAIN_S / engine.OBS_INTERVAL)):
        engine._check_completion("tabpfn", 30)
    assert engine.mission_report is not None
