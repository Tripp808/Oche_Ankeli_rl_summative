"""

This script evaluates trained DQN and PPO models in the hospital navigation environment
for both delivery and emergency modes.
"""

import os
import time
import pygame
import numpy as np
from stable_baselines3 import DQN, PPO
from environment.maternity_ward import MaternalCareEnv  

# Configuration
MODEL_PATHS = {
    "delivery": {
        "DQN": "./models/dqn/delivery/final_model.zip",  
        "PPO": "./models/ppo/delivery/final_model.zip"
    },
    "emergency": {
        "DQN": "./models/dqn/emergency/final_model.zip",  
        "PPO": "./models/ppo/emergency/final_model.zip"
    }
}

MODE = "emergency"  # "delivery" or "emergency"
RENDER_MODE = "human"  # "human" for visualization, None for faster evaluation
NUM_EPISODES = 3  # How many episodes to run
MODEL_TYPES = {
    "DQN": DQN,
    "PPO": PPO
}

def load_model(model_path, model_type):
    """
    Load a trained model
    
    Args:
        model_path: Path to the saved model file
        model_type: Model class (DQN or PPO)
    
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    return model_type.load(model_path)

def evaluate_model(model, env, num_episodes=3, model_name="Model"):
    """
    Run the model in the environment and collect metrics
    
    Args:
        model: Trained model to evaluate
        env: Environment to run the model in
        num_episodes: Number of episodes to evaluate
        model_name: Name of the model for display purposes
    
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    print(f"\nEvaluating {model_name} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Episode {episode + 1}/{num_episodes} started...")
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # Display current location if available
            if info and 'location' in info and info['location']:
                location = info['location'].replace('_', ' ').title()
                print(f"\rStep {steps}: In {location}, Reward: {reward:.1f}", end="")
            
            # Render environment if in human mode
            if RENDER_MODE == "human":
                env.render()
                time.sleep(0.05)  # Slow down for visualization
        
        # Record episode results
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Check if this was a successful episode (reached target)
        success = terminated and (
            (env.mode == 'delivery' and info.get('location') == 'delivery_room') or
            (env.mode == 'emergency' and info.get('location') == 'emergency_room')
        )
        
        if success:
            successes += 1
            result = "SUCCESS"
        else:
            result = "FAILURE"
        
        print(f"\nEpisode {episode + 1}: {result} - Reward = {total_reward:.1f}, Steps = {steps}")
    
    # Calculate metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_steps": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }
    
    return metrics

def print_header():
    """Print a fancy header for the evaluation script"""
    print("\n" + "="*80)
    print(f"MATERNAL CARE NAVIGATION MODEL EVALUATION".center(80))
    print(f"User: Emmanuel-Begati | Date: 2025-04-03 16:52:42".center(80))
    print("="*80)

def main():
    print_header()
    
    # Ask user which mode to evaluate
    mode_choice = input("\nChoose mode to evaluate (delivery/emergency) [default: delivery]: ").strip().lower()
    if mode_choice in ["delivery", "emergency"]:
        selected_mode = mode_choice
    else:
        selected_mode = MODE
        print(f"Using default mode: {selected_mode}")
    
    # Ask user which models to evaluate
    model_choice = input("\nChoose models to evaluate (dqn/ppo/both) [default: both]: ").strip().lower()
    if model_choice == "dqn":
        selected_models = ["DQN"]
    elif model_choice == "ppo":
        selected_models = ["PPO"]
    else:
        selected_models = ["DQN", "PPO"]
        print("Evaluating both DQN and PPO models")
    
    # Initialize pygame for rendering if needed
    if RENDER_MODE == "human":
        pygame.init()
    
    # Create environment with selected mode
    env = MaternalCareEnv(render_mode=RENDER_MODE, mode=selected_mode)
    
    print(f"\nInitialized environment in {selected_mode.upper()} mode")
    print(f"Target location: {'Delivery Room' if selected_mode == 'delivery' else 'Emergency Room'}")
    
    # Store results for comparison
    all_results = {}
    
    # Evaluate each selected model
    for model_name in selected_models:
        model_path = MODEL_PATHS[selected_mode][model_name]
        
        print(f"\n{'-'*40}")
        print(f"Evaluating {model_name} model for {selected_mode} mode...")
        print(f"{'-'*40}")
        
        try:
            # Load model
            model = load_model(model_path, MODEL_TYPES[model_name])
            
            # Run evaluation
            metrics = evaluate_model(model, env, NUM_EPISODES, model_name)
            all_results[model_name] = metrics
            
            # Print results
            print(f"\n{model_name} Results ({selected_mode} mode):")
            print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"Mean Steps: {metrics['mean_steps']:.2f}")
            print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
    
    # Compare models if both were evaluated
    if len(all_results) > 1:
        print("\n" + "="*40)
        print("MODEL COMPARISON".center(40))
        print("="*40)
        
        comparison_metrics = ["mean_reward", "mean_steps", "success_rate"]
        header_format = "{:<15} " + " ".join(["{:<15}" for _ in selected_models])
        print(header_format.format("Metric", *selected_models))
        print("-" * 15 * (len(selected_models) + 1))
        
        for metric in comparison_metrics:
            values = []
            for model in selected_models:
                if model in all_results:
                    if metric == "success_rate":
                        value = f"{all_results[model][metric]*100:.1f}%"
                    else:
                        value = f"{all_results[model][metric]:.2f}"
                    values.append(value)
                else:
                    values.append("N/A")
            
            display_metric = metric.replace("_", " ").title()
            row_format = "{:<15} " + " ".join(["{:<15}" for _ in values])
            print(row_format.format(display_metric, *values))
        
        # Determine the winner
        if all(model in all_results for model in selected_models):
            # Compare by mean reward
            best_model = max(selected_models, key=lambda m: all_results[m]["mean_reward"])
            print(f"\nBest performing model: {best_model} (based on mean reward)")
    
    # Close environment
    env.close()
    if RENDER_MODE == "human":
        pygame.quit()
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        pygame.quit()
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        pygame.quit()