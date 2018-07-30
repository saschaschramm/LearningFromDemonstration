from common.env_wrapper import init_env
from common.model import Model, PolicyFullyConnected
from common.runner import Runner
from common.utilities import global_seed

def run():
    global_seed(0)
    env = init_env()

    model = Model(
        policy=PolicyFullyConnected,
        observation_space=(80, 80),
        action_space=3,
        learning_rate=2e-4
    )

    dir = "reinforce"

    runner = Runner(
        env = env,
        model = model,
        batch_size=128,
        timesteps=int(1e6),
        discount_rate=0.99,
        summary_frequency=20000,
        performance_num_episodes=100,
        summary_log_dir=dir
    )
    runner.run()

run()