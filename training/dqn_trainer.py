from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from maternal_care_env import MaternalCareEnv
from callbacks import TrainingProgressCallback
import yaml

def train_dqn(config_path="configs/dqn_config.yaml"):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = MaternalCareEnv(mode=config['mode'])
    
    # Create callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"models/dqn_{config['mode']}",
        log_path=f"logs/dqn_{config['mode']}",
        eval_freq=config['eval_freq'],
        deterministic=True
    )
    
    progress_callback = TrainingProgressCallback(
        check_freq=1000,
        log_dir=f"logs/dqn_{config['mode']}"
    )
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        exploration_fraction=config['exploration_fraction'],
        exploration_initial_eps=config['exploration_initial_eps'],
        exploration_final_eps=config['exploration_final_eps'],
        verbose=1,
        tensorboard_log=f"logs/dqn_{config['mode']}"
    )
    
    # Train the model
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[eval_callback, progress_callback]
    )
    
    # Save the final model
    model.save(f"models/dqn_{config['mode']}_final")
    
    return model