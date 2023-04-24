from dqn_tutorial.collect_data import collect_data, save_data


def test_collect_data(tmp_path):
    env_id = "CartPole-v1"
    output_filename = tmp_path / f"{env_id}_data"
    # Collect data
    data = collect_data(env_id, n_steps=10_000)
    assert len(data.observations) == 10_000
    # Save collected data using numpy
    save_data(data, output_filename)
