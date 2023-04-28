from dqn_tutorial.fqi import collect_data, load_data, save_data


def test_collect_data(tmp_path):
    env_id = "CartPole-v1"
    output_filename = tmp_path / f"{env_id}_data"
    # Collect data
    data = collect_data(env_id, n_steps=10_000)
    assert len(data.observations) == 10_000
    # Save collected data using numpy
    save_data(data, output_filename)
    # load data
    data = load_data(output_filename)
    assert len(data.observations) == 10_000
