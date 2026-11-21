"""TabPFN is causal in target selection, not merely a heatmap side effect."""
import numpy as np
import pytest

from integration.prediction_control import PredictionController
from sim.world import World
from tests.test_search_control import reading


def settled():
    controller=PredictionController(World().to_world_init())
    for t in [.5,1,1.5,2,2.5,3]:controller.ingest(reading(t))
    assert controller.awaiting_prediction
    return controller


def test_identical_measurements_and_map_follow_opposite_model_predictions():
    points=[(-46,-40,5),(-34,-40,5),(-40,-40,11)]
    left=settled();right=settled();up=settled()
    left.set_source_prediction(points,[.99,.01,.01],3)
    right.set_source_prediction(points,[.01,.99,.01],3)
    up.set_source_prediction(points,[.01,.01,.99],3)
    assert left.goal[0]<-40 and right.goal[0]>-40 and up.goal[2]>5
    assert left.decision_basis==right.decision_basis==up.decision_basis=='tabpfn_source_prediction'
    assert left.prediction_decisions==right.prediction_decisions==up.prediction_decisions==1


def test_contact_without_model_prediction_cannot_start_a_directional_probe():
    c=settled()
    for t in [3.5,4,4.5,5]:c.ingest(reading(t,ppm=500))
    np.testing.assert_array_equal(c.goal,(-40,-40,5))
    assert c.awaiting_prediction and c.prediction_decisions==0


def test_expired_predictions_cannot_start_new_measurement_flights():
    c=settled()
    c.set_source_prediction([(-34,-40,5)],[.99],-3)
    np.testing.assert_array_equal(c.goal,(-40,-40,5))
    assert c.awaiting_prediction and c.prediction_decisions==0


@pytest.mark.parametrize('scores', [[np.nan],[1.01],[-.01],[.5,.5]])
def test_invalid_prediction_does_not_change_the_target(scores):
    c=settled()
    with pytest.raises(ValueError):c.set_source_prediction([(-34,-40,5)],scores,3)
    np.testing.assert_array_equal(c.goal,(-40,-40,5))
