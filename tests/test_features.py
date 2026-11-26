import numpy as np
import pytest

from doom_env.spaces import observation_space
from submission.model.features import FeatureBuilder


@pytest.mark.parametrize("collision_aware, expected_stuck", [(False, 0), (True, 1)])
def test_stuck_feature_under_wall_momentum(collision_aware, expected_stuck):
    obs = observation_space().sample()
    obs["rays"]["distance"].fill(0.125)
    obs["rays"]["hit_type"].fill(1)
    obs["visible_entities"]["mask"].fill(0)
    obs["vitals"].fill(0)
    obs["vitals"][14] = 4.0
    builder = FeatureBuilder(include_rgb=False, collision_aware=collision_aware)
    forward = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    builder.step(obs)
    builder.note_action(forward)
    builder.step(obs)
    builder.note_action(forward)
    builder.step(obs)
    assert builder.stuck == expected_stuck

    # After exploring a new location the wall scan changes, so motion counts again.
    obs["rays"]["distance"].fill(0.25)
    builder.note_action(forward)
    builder.step(obs)
    assert builder.stuck == 0
