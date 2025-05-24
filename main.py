# Copyright (c) 2025, Bongani. All rights reserved.
# This file is part of the Space War Shoot RL Version project.
# Author: Bongani Jele <jelebongani43@gmail.com>

########################################################################################
# This project implements a reinforcement learning agent using the Bellman Equation    #
# with PyTorch and NumPy in a Pygame-based space shooter game.                         #
#                                                                                      #
# Pygame doc: https://www.pygame.org/docs/ref/rect.html                                #
#                                                                                      #
# For just in case if you experience issues, email me or                               #
# contribute on GitHub — I'll appreciate your support!                                 #
########################################################################################

import pygame
import random
from constants import WIDTH, HEIGHT, FPS, WHITE, RED, ENEMY_VEL, MAX_BULLETS


class Player(pygame.sprite.Sprite):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (100, HEIGHT // 2)
        self.health = 3
        self.bullets = pygame.sprite.Group()
        self.last_shot_time = 0
        self.shoot_delay = 250

    def move(self, action):
        if action == 0 and self.rect.left > 0:
            self.rect.x -= 5
        elif action == 1 and self.rect.right < WIDTH:
            self.rect.x += 5
        elif action == 2 and self.rect.top > 10:
            self.rect.y -= 5
        elif action == 3 and self.rect.bottom < HEIGHT:
            self.rect.y += 5
        elif action == 4:
            self.shoot()

        self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))
        self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if len(self.bullets) < MAX_BULLETS and current_time - self.last_shot_time >= self.shoot_delay:
            bullet = Bullet(self.rect.centerx, self.rect.top)
            self.bullets.add(bullet)
            self.last_shot_time = current_time

    def update_bullets(self, enemies):
        reward = 0
        for bullet in list(self.bullets):
            bullet.update()
            if bullet.rect.bottom < 0:
                self.bullets.remove(bullet)
                continue

            collided_enemies = pygame.sprite.spritecollide(bullet, enemies, False)
            if collided_enemies:
                self.bullets.remove(bullet)
                for enemy in collided_enemies:
                    enemy.kill()
                    reward += 1
        return reward
    
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill((98, 190, 193))
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.rect.y -= 7


class Enemy(pygame.sprite.Sprite):
    def __init__(self, image):
        super().__init__()
        self.image = image
        x = random.randint(0, WIDTH - self.image.get_width())
        y = random.randint(-300, 0)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.bullets = pygame.sprite.Group()
        self.last_shot_time = 0
        self.shoot_delay = random.randint(1000, 2000)  # Random delay for variety

    def update(self):
        self.rect.y += ENEMY_VEL
        if self.rect.top > HEIGHT:
            self.kill()

    def shoot(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot_time >= self.shoot_delay:
            bullet = EnemyBullet(self.rect.centerx, self.rect.bottom)
            self.bullets.add(bullet)
            self.last_shot_time = current_time

    def update_bullets(self):
        self.bullets.update()

            
            
class EnemyBullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill((255, 0, 0))  # Proper fill syntax
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.rect.y += 5  # Enemies shoot downward





class SpaceShooterRL:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Space Shooter RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)

        # Load images after pygame display initialized
        self.player_image = pygame.image.load('assets/spaceship/ship1.png').convert_alpha()
        self.player_image = pygame.transform.scale(self.player_image, (60, 60))
        self.enemy_image = pygame.image.load('assets/enemies/alien-ship.png').convert_alpha()
        self.enemy_image = pygame.transform.scale(self.enemy_image, (50, 50))


        self.bg_image = pygame.image.load('assets/kurt/space_ft.png').convert_alpha()
        self.bg_image = pygame.transform.scale(self.bg_image, (WIDTH, HEIGHT))

        self.agent = Agent()
        self.player = Player(self.player_image)
        self.enemies = pygame.sprite.Group([Enemy(self.enemy_image) for _ in range(5)])

        self.score = 0
        self.game_over = False

    def reset(self):
        self.player = Player(self.player_image)
        self.enemies = pygame.sprite.Group([Enemy(self.enemy_image) for _ in range(5)])
        self.score = 0
        self.episode_reward = 0
        self.game_over = False

    def step(self, action):
        if self.game_over:
            return 0

        self.player.move(action)
        self.enemies.update()

        while len(self.enemies) < 3:
            self.enemies.add(Enemy(self.enemy_image))

        # Enemy shooting and bullet updates
        for enemy in self.enemies:
            if random.random() < 0.01:  # 1% chance to shoot per frame
                bullet = EnemyBullet(enemy.rect.centerx, enemy.rect.bottom)
                enemy.bullets.add(bullet)

            enemy.bullets.update()

            # Check for bullet collision with player
            for bullet in enemy.bullets:
                if bullet.rect.colliderect(self.player.rect):
                    self.player.health -= 1
                    enemy.bullets.remove(bullet)
                    if self.player.health <= 0:
                        self.game_over = True

        # Player bullet logic
        reward = self.player.update_bullets(self.enemies)
        self.score += reward
        
        self.episode_reward += reward

        # Handle direct collisions
        for enemy in self.enemies.sprites():
            if enemy.rect.colliderect(self.player.rect):
                self.player.health -= 1
                enemy.kill()
                self.enemies.add(Enemy(self.enemy_image))
                if self.player.health <= 0:
                    self.game_over = True

        return reward


    def run_episode(self):
        self.reset()
        
        done = False
        while not done:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return True

            state = self.agent.get_state(self.player, self.enemies)
            action = self.agent.act(state)
            reward = self.step(action)
          
            done = self.game_over
            
            
            self.agent.record_reward(self.episode_reward)

            next_state = self.agent.get_state(self.player, self.enemies)
            self.agent.memory.push((state, action, reward, next_state, done))
            self.agent.train(64)

            self.render()

        print(f"Episode finished. Score: {self.score}, Epsilon: {self.agent.epsilon:.2f}, Total Reward: {self.episode_reward}")
        return False
    
    def render(self):
        self.screen.blit(self.bg_image, (0, 0))

        # Draw player and their bullets
        self.screen.blit(self.player.image, self.player.rect)
        self.player.bullets.draw(self.screen)

        # Draw enemies
        self.enemies.draw(self.screen)

        # Draw each enemy's bullets
        for enemy in self.enemies:
            enemy.bullets.draw(self.screen)

        # UI - Score and Health
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        health_text = self.font.render(f"Health: {self.player.health}", True, WHITE)
        self.screen.blit(health_text, (10, 40))

        # Game Over message
        if self.game_over:
            over_text = self.font.render("Game Over!", True, RED)
            self.screen.blit(over_text, (WIDTH // 2 - 70, HEIGHT // 2))

        pygame.display.flip()


    def main_loop(self):
        running = True
        while running:
            running = not self.run_episode()
        pygame.quit()


if __name__ == "__main__":
    from agent import Agent  # Import here to avoid circular issues if any
    # from agent_no_matlib import Agent  # Import here to avoid circular issues if any
    game = SpaceShooterRL()
    game.main_loop()
