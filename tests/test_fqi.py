import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from dqn_tutorial.fqi import create_model_input, get_q_values


def test_model_input():  # pragma: no cover
    obs = np.ones((10, 2))
    actions = np.zeros((10, 1))
    model_input = create_model_input(obs, actions)
    assert model_input.shape == (10, 3)
    assert np.allclose(model_input[:, 2], actions)

    # with features_extractor
    obs = np.ones((10, 1))
    model_input = create_model_input(obs, actions, PolynomialFeatures(degree=2))
    # x1^2, x1, x1*x2, x2, x2^2, bias
    assert model_input.shape == (10, 6)


def test_q_values() -> None:
    model = LinearRegression()
    obs = np.ones((10, 2))
    # Fit on dummy data
    model.fit(np.ones((10, 3)), np.ones(10))

    q_values = get_q_values(model, obs, n_actions=3)
    assert q_values.shape == (10, 3)
