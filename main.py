#!/usr/bin/env python3
"""
MamaSafe RL - Main execution script
Command-line interface for training, evaluating, and visualizing maternal care navigation agents
"""

import argparse
import os
import yaml
from datetime import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList
from maternal_care import MaternalCareEnv
from callbacks import TrainingProgressCallback, EvalCallback, CheckpointCallback

def train_dqn(config):
    """Train DQN agent with given configuration"""
    print(f"\nTraining DQN agent in {config['mode']} mode...")
    
    env = MaternalCareEnv(mode=config['mode'])
    log_dir = f"logs/dqn_{config['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = CallbackList([
        EvalCallback(
            env,
            best_model_save_path=f"{log_dir}/best_model",
            log_path=log_dir,
            eval_freq=config['eval_freq']
        ),
        TrainingProgressCallback(log_dir=log_dir),
        CheckpointCallback(
            save_freq=config['eval_freq'],
            save_path=f"{log_dir}/checkpoints",
            name_prefix="dqn"
        )
    ])
    
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
        tensorboard_log=log_dir,
        verbose=1
    )
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks
    )
    
    model.save(f"models/dqn_{config['mode']}_final")
    env.close()
    return model

def train_ppo(config):
    """Train PPO agent with given configuration"""
    print(f"\nTraining PPO agent in {config['mode']} mode...")
    
    env = MaternalCareEnv(mode=config['mode'])
    log_dir = f"logs/ppo_{config['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = CallbackList([
        EvalCallback(
            env,
            best_model_save_path=f"{log_dir}/best_model",
            log_path=log_dir,
            eval_freq=config['eval_freq']
        ),
        TrainingProgressCallback(log_dir=log_dir),
        CheckpointCallback(
            save_freq=config['eval_freq'],
            save_path=f"{log_dir}/checkpoints",
            name_prefix="ppo"
        )
    ])
    
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
        tensorboard_log=log_dir,
        verbose=1
    )
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks
    )
    
    model.save(f"models/ppo_{config['mode']}_final")
    env.close()
    return model

def visualize_environment(mode='delivery'):
    """Launch interactive environment visualization"""
    env = MaternalCareEnv(mode=mode, render_mode='human')
    print(f"\nVisualizing {mode} environment...")
    print("Controls: Close window to exit")
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action = env.action_space.sample()  # Random actions for demo
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    
    env.close()

def evaluate_model(model_path, mode, num_episodes=10):
    """Evaluate a trained model"""
    print(f"\nEvaluating {model_path} in {mode} mode...")
    
    env = MaternalCareEnv(mode=mode)
    model = DQN.load(model_path) if "dqn" in model_path.lower() else PPO.load(model_path)
    
    total_rewards = []
    successes = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        if episode_reward > 50:  # Success threshold
            successes += 1
    
    env.close()
    
    print(f"Evaluation over {num_episodes} episodes:")
    print(f"Average reward: {sum(total_rewards)/num_episodes:.2f}")
    print(f"Success rate: {(successes/num_episodes)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="MamaSafe RL Training System")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train RL agents')
    train_parser.add_argument('--algo', choices=['dqn', 'ppo', 'both'], required=True)
    train_parser.add_argument('--mode', choices=['delivery', 'emergency'], required=True)
    train_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize environment')
    viz_parser.add_argument('--mode', choices=['delivery', 'emergency'], default='delivery')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('model_path', help='Path to trained model')
    eval_parser.add_argument('--mode', choices=['delivery', 'emergency'], required=True)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        with open(args.config) as f:
            config = yaml.safe_load(f)
        config['mode'] = args.mode
        
        if args.algo in ['dqn', 'both']:
            train_dqn(config)
        if args.algo in ['ppo', 'both']:
            train_ppo(config)
    
    elif args.command == 'visualize':
        visualize_environment(args.mode)
    
    elif args.command == 'evaluate':
        evaluate_model(args.model_path, args.mode)

if __name__ == "__main__":
    main()