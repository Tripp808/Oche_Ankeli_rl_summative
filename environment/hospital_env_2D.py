import pygame
import numpy as np
import time
from maternal_care import MaternalCareEnv

def visualize_environment():
    # Create both environment modes for visualization
    env_delivery = MaternalCareEnv(render_mode="human", mode="delivery")
    env_emergency = MaternalCareEnv(render_mode="human", mode="emergency")
    
    current_env = env_delivery
    current_mode = "delivery"
    
    # Initialize pygame for user input
    pygame.init()
    font = pygame.font.SysFont('Arial', 20)
    
    running = True
    manual_control = True  # Start with manual control enabled
    
    print("Environment Visualization")
    print("-------------------------")
    print("Press SPACE to switch between delivery and emergency modes")
    print("Press M to toggle manual control")
    print("Press R to reset the environment")
    print("Press ESC or Q to quit")
    print("Use arrow keys for manual control when enabled")
    
    # Show initial control status
    status_text = font.render("Manual Control: ENABLED", True, (0, 100, 0))
    current_env.window.blit(status_text, (300, 570))
    pygame.display.update()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Switch modes
                if event.key == pygame.K_SPACE:
                    if current_mode == "delivery":
                        current_env = env_emergency
                        current_mode = "emergency"
                    else:
                        current_env = env_delivery
                        current_mode = "delivery"
                    current_env.reset()
                
                # Toggle manual control
                elif event.key == pygame.K_m:
                    manual_control = not manual_control
                    if manual_control:
                        status_text = font.render("Manual Control: ENABLED", True, (0, 100, 0))
                    else:
                        status_text = font.render("Manual Control: DISABLED", True, (100, 0, 0))
                    
                    current_env.window.blit(status_text, (300, 570))
                    pygame.display.update()
                    time.sleep(0.5)
                
                # Reset environment
                elif event.key == pygame.K_r:
                    current_env.reset()
                
                # Quit
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                
                # Manual control
                elif manual_control:
                    action = None
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_LEFT:
                        action = 3
                    
                    if action is not None:
                        obs, reward, terminated, truncated, info = current_env.step(action)
                        if terminated:
                            time.sleep(1)
                            current_env.reset()
        
        # If not in manual control, take random actions at a slower pace
        if not manual_control:
            action = current_env.action_space.sample()
            obs, reward, terminated, truncated, info = current_env.step(action)
            if terminated:
                time.sleep(1)
                current_env.reset()
        
        # Render the current environment
        current_env.render()
        
        # Display control instructions
        control_bg = pygame.Surface((700, 30))
        control_bg.fill((240, 240, 240))
        control_bg.set_alpha(200)
        current_env.window.blit(control_bg, (50, 565))
        
        controls_text = font.render("SPACE: Switch Mode | M: Manual/Random | R: Reset | Arrow Keys: Move | ESC: Quit", True, (0, 0, 0))
        current_env.window.blit(controls_text, (60, 570))
        
        # Show manual control status
        if manual_control:
            status_text = font.render("Manual Control", True, (0, 100, 0))
        else:
            status_text = font.render("Random Actions", True, (100, 0, 0))
        current_env.window.blit(status_text, (650, 570))
        
        # Update the display
        pygame.display.update()
        
        # Slow down the visualization when using random actions
        if not manual_control:
            time.sleep(0.2)
    
    # Close environments
    env_delivery.close()
    env_emergency.close()
    pygame.quit()

if __name__ == "__main__":
    visualize_environment()