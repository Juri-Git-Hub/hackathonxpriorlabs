"""Prediction-driven policy; new predeclared holdouts, separate from v1 tests."""
PROFILE=dict(
    name='windframe12-model-driven-v2',context_rows=4096,
    source_features=12,hazard_features=20,source_interval_s=3,hazard_interval_s=10,
    sensor_dwell_s=2.5,max_model_step_m=6,physical_dt_s=.05,
    goal_selection='TabPFN source probabilities; novelty and travel cost for measurement planning',
    before_gas_contact='source-independent map coverage',
    without_fresh_prediction='hold after the current measurement stop; no directional fallback',
    success='unchanged simulator judge: estimate <=5m, drone <=5m, ppm>15 for 2s',
    selection_split='validation',validation_seeds=[41,42,43],test_seeds=[201,202,203,204,205,206,207],
)
TEST_CASES=[
    dict(name='default_seed201',seed=201),
    dict(name='default_seed202',seed=202),
    dict(name='default_seed203',seed=203),
    dict(name='other_valve',seed=204,source=[-17.5,-2.,3.5]),
    dict(name='east_tank_port',seed=205,source=[20.5,-7.,6.]),
    dict(name='off_asset',seed=206,source=[8.,-10.,4.]),
    dict(name='elevated',seed=207,source=[15.,8.,13.]),
]
