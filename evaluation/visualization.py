import pygame
import argparse
from stable_baselines3 import DQN, PPO
from maternal_care import MaternalCareEnv

def visualize_agent(model_path=None, mode='delivery'):
    """rendering logic for the Maternal Care environment"""
    env = MaternalCareEnv(mode=mode, render_mode='human')
    
    if model_path:
        if 'dqn' in model_path.lower():
            model = DQN.load(model_path)
        else:
            model = PPO.load(model_path)
    
    obs, _ = env.reset()
    done = False
    
    print(f"Visualizing {mode} mode")
    print("Controls:" + (" Auto-pilot" if model_path else " Arrow keys to move, SPACE to wait"))
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if model_path:
            action, _ = model.predict(obs, deterministic=True)
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action = 0
            elif keys[pygame.K_DOWN]: action = 1
            elif keys[pygame.K_LEFT]: action = 2
            elif keys[pygame.K_RIGHT]: action = 3
            elif keys[pygame.K_SPACE]: action = 4
            else: action = 4  # Default to wait
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        env.render()
        clock.tick(10)  # 10 FPS
        
        if done:
            obs, _ = env.reset()
            done = False
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Maternal Care Environment')
    parser.add_argument('--mode', choices=['delivery', 'emergency'], default='delivery')
    parser.add_argument('--model', help='Path to trained model for auto-pilot')
    args = parser.parse_args()
    
    visualize_agent(model_path=args.model, mode=args.mode)