import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import sys
from pathlib import Path

# ------------------ Paths & Imports ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from envs.icu_env import HackathonICUEnv, ICUAction

# ------------------ Native RL Agent ------------------
class ActorCritic(nn.Module):
    def __init__(self, input_dim=3, action_dim=2):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor head: Outputs mean and log_std for continuous actions
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head: Outputs state value
        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.shared(state)
        
        action_mean = torch.tanh(self.actor_mean(features)) # Bound between -1 and 1
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        state_value = self.critic(features)
        return action_mean, action_std, state_value

# ------------------ Training Loop ------------------
def train_native_rl():
    env = HackathonICUEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="hackathon-rl-ptz",
        name="native_rl_run",
        config={
            "learning_rate": 1e-3,
            "epochs": 1000,
            "gamma": 0.99
        }
    )
    
    agent = ActorCritic().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    gamma = 0.99
    
    epochs = 1000
    print("Starting Native OpenEnv RL Training...")
    
    for epoch in range(epochs):
        # 1. Reset OpenEnv
        obs = env.reset()
        done = False
        epoch_reward = 0
        
        while not done:
            # 2. Extract state from OpenEnv Observation
            state_tensor = torch.tensor(
                [obs.current_pan, obs.current_tilt, obs.monitor_distance], 
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # 3. Agent predicts action
            mean, std, state_value = agent(state_tensor)
            dist = Normal(mean, std)
            action_tensor = dist.sample()
            
            # 4. Step OpenEnv natively
            action = ICUAction(
                pan_target=float(action_tensor[0][0]), 
                tilt_target=float(action_tensor[0][1])
            )
            next_obs = env.step(action)
            
            # 5. Extract next state & compute target
            next_state_tensor = torch.tensor(
                [next_obs.current_pan, next_obs.current_tilt, next_obs.monitor_distance], 
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            _, _, next_state_value = agent(next_state_tensor)
            
            # 6. Calculate 1-step Actor-Critic Loss
            reward = torch.tensor([next_obs.reward], dtype=torch.float32).to(device)
            td_target = reward + gamma * next_state_value * (1 - int(next_obs.done))
            td_error = td_target - state_value
            
            critic_loss = td_error.pow(2).mean()
            actor_loss = -dist.log_prob(action_tensor).sum() * td_error.detach()
            loss = actor_loss + critic_loss
            
            # 7. Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            obs = next_obs
            epoch_reward += float(reward)
            done = next_obs.done

        best_reward = max(best_reward, epoch_reward)
        wandb.log({
            "epoch": epoch + 1,
            "total_reward": epoch_reward,
            "avg_reward_per_step": epoch_reward / max(step_count, 1),
            "best_reward": best_reward
        })
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Reward: {epoch_reward:.2f}")

    # Save weights
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), model_dir / "native_rl_policy.pth")
    print("RL Model saved to models/native_rl_policy.pth")

if __name__ == "__main__":
    train_native_rl()