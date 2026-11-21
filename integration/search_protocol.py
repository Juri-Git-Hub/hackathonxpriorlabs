"""Frozen search configuration and predeclared final test scenarios."""
PROFILE=dict(name='windframe12-local-probes-v1',context_rows=4096,source_features=12,hazard_features=20,source_interval_s=3,hazard_interval_s=10,sensor_dwell_s=2.5,initial_probe_m=6,volumetric_expansion_after_s=60,min_probe_m=1.5,physical_dt_s=.05,success='unchanged simulator judge: estimate <=5m, drone <=5m, ppm>15 for 2s',selection_split='validation',validation_seeds=[41,42,43],test_seeds=[101,102,103])
# Set before running final trials; no scenario is discarded based on outcome.
TEST_CASES=[
 dict(name='default_seed101',seed=101),
 dict(name='default_seed102',seed=102),
 dict(name='default_seed103',seed=103),
 dict(name='other_valve',seed=104,source=[-17.5,-2.,3.5]),
 dict(name='east_tank_port',seed=105,source=[20.5,-7.,6.]),
 dict(name='off_asset',seed=106,source=[8.,-10.,4.]),
 dict(name='elevated',seed=107,source=[15.,8.,13.]),
]
