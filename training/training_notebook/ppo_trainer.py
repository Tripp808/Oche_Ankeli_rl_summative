from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from maternal_care_env import MaternalCareEnv
from callbacks import TrainingProgressCallback
import yaml

def train_ppo(config_path="configs/ppo_config.yaml"):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = DummyVecEnv([lambda: MaternalCareEnv(mode=config['mode'])])
    
    # Create callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"models/ppo_{config['mode']}",
        log_path=f"logs/ppo_{config['mode']}",
        eval_freq=config['eval_freq'],
        deterministic=True
    )
    
    progress_callback = TrainingProgressCallback(
        check_freq=1000,
        log_dir=f"logs/ppo_{config['mode']}"
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        verbose=1,
        tensorboard_log=f"logs/ppo_{config['mode']}"
    )
    
    # Train the model
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[eval_callback, progress_callback]
    )
    
    # Save the final model
    model.save(f"models/ppo_{config['mode']}_final")
    
    return model