import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN, PPO
from maternal_care import MaternalCareEnv
from tqdm import tqdm
import argparse

def test_trained_agent(model, env_mode, num_episodes=5):
    """ testing function """
    env = MaternalCareEnv(mode=env_mode)
    
    rewards = []
    steps = []
    successes = 0

    for episode in tqdm(range(num_episodes), desc="Testing agent"):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1

        rewards.append(episode_reward)
        steps.append(episode_steps)
        if episode_reward > 50:
            successes += 1

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "success_rate": float(successes / num_episodes),
        "num_episodes": num_episodes
    }

    print("\nPerformance Metrics:")
    print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Min/Max reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
    print(f"Mean steps: {metrics['mean_steps']:.2f} ± {metrics['std_steps']:.2f}")
    print(f"Success rate: {metrics['success_rate']*100:.1f}%")

    env.close()
    return metrics

def compare_models(models_info, mode='delivery', num_episodes=20):
    """comparison function """
    all_metrics = []

    for model, model_name, model_type in models_info:
        print(f"\nEvaluating model: {model_name}")
        metrics = test_trained_agent(model, mode, num_episodes)
        metrics["model_name"] = model_name
        metrics["model_type"] = model_type
        all_metrics.append(metrics)

    # plotting 
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 10))

    model_names = [m["model_name"] for m in all_metrics]
    mean_rewards = [m["mean_reward"] for m in all_metrics]
    std_rewards = [m["std_reward"] for m in all_metrics]
    success_rates = [m["success_rate"] * 100 for m in all_metrics]
    mean_steps = [m["mean_steps"] for m in all_metrics]

    plt.subplot(2, 2, 1)
    plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=10)
    plt.title('Mean Reward by Model')
    plt.ylabel('Mean Reward')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    plt.bar(model_names, success_rates)
    plt.title('Success Rate by Model')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    plt.bar(model_names, mean_steps)
    plt.title('Mean Steps by Model')
    plt.ylabel('Mean Steps')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f'visualizations/model_comparison_{mode}.png')
    plt.show()

    return all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare trained RL models')
    parser.add_argument('--dqn_path', required=True, help='Path to DQN model')
    parser.add_argument('--ppo_path', required=True, help='Path to PPO model')
    parser.add_argument('--mode', choices=['delivery', 'emergency'], default='delivery')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()

    # Load models 
    dqn_model = DQN.load(args.dqn_path)
    ppo_model = PPO.load(args.ppo_path)

    # comparison setup
    models_info = [
        (dqn_model, "DQN Agent", "dqn"),
        (ppo_model, "PPO Agent", "ppo")
    ]

    # Run comparison
    comparison_metrics = compare_models(models_info, mode=args.mode, num_episodes=args.episodes)

    # Print results 
    for metrics in comparison_metrics:
        print(f"\nModel: {metrics['model_name']} ({metrics['model_type']})")
        print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean steps: {metrics['mean_steps']:.2f}")
        print(f"Success rate: {metrics['success_rate']*100:.1f}%")