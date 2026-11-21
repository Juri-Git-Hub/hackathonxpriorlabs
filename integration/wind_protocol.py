"""Predeclared unseen wind scenarios for the measured-wind acquisition policy."""
from integration.prediction_protocol import PROFILE as PREDICTION_PROFILE

PROFILE={**PREDICTION_PROFILE,
    'name':'windframe12-reactive-acquisition-v3',
    'adaptive_wind':True,
    'before_gas_contact':'map coverage refreshed after >=60deg sustained measured wind change',
    'wind_replan_cooldown_s':8,
    'wind_history_s':4,
    'headwind_cost_penalty':False,
    'test_seeds':list(range(301,308)),
}
TEST_CASES=[
    dict(name='default',seed=301),
    dict(name='north_wind',seed=302,wind=[90,3.5]),
    dict(name='west_wind',seed=303,wind=[180,4.5]),
    dict(name='early_wind_reversal',seed=304,wind_schedule=[dict(time=5,angle=195,speed=3.5)]),
    dict(name='later_wind_change',seed=305,wind_schedule=[dict(time=35,angle=135,speed=4)]),
    dict(name='off_asset',seed=306,source=[8.,-10.,4.],wind=[225,3.]),
    dict(name='elevated',seed=307,source=[15.,8.,13.]),
]
