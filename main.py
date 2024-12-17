import minari
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import random
import torch.nn.functional as F  # Import functional API for activation functions
import os
from datetime import datetime


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download(dataset_id='D4RL/pen/expert-v2'):
    dataset = minari.load_dataset(dataset_id, True)
    observations, actions, rewards, terminations, truncations, next_observations, steps = [], [], [], [], [], [], []
    for episode in dataset.iterate_episodes():
        episode_length = len(episode.observations)
        for i in range(min(100, episode_length - 1)):
            observations.append(episode.observations[i])
            actions.append(episode.actions[i])
            rewards.append(episode.rewards[i])
            terminations.append(episode.terminations[i])
            truncations.append(episode.truncations[i])
            next_obs = episode.observations[i + 1]
            next_observations.append(next_obs)
            steps.append(i + 1)  # Number of step for each transition

    return dataset, {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminations': np.array(terminations),
        'truncations': np.array(truncations),
        'next_observations': np.array(next_observations),
        'dones': np.logical_or(terminations, truncations).astype(int),
        'steps': np.array(steps)  # New key for step count for each transition
    }


class StepPredictionModel(nn.Module):
    def __init__(self, state_dim):
        super(StepPredictionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.network(state)


def train_step_prediction_model(dataset, epochs=20, lr=1e-3, batch_size=128):
    save_path = "c:/users/armin/step_aware/"
    model_filename = os.path.join(save_path, "step_aware.pth")
    log_file = os.path.join(save_path, "step_aware_log.txt")
    os.makedirs(save_path, exist_ok=True)

    state_dim = dataset['observations'].shape[1]
    model = StepPredictionModel(state_dim)

    # Check if pre-trained model exists
    if os.path.exists(model_filename):
        print("Pre-trained model found. Loading model...")
        model.load_state_dict(torch.load(model_filename))
        model.eval()
        print("Model loaded. Skipping training.")
        return model

    # Training process
    observations = torch.tensor(dataset['observations'], dtype=torch.float32)
    steps = torch.tensor(dataset['steps'], dtype=torch.float32).unsqueeze(-1)

    full_dataset = TensorDataset(observations, steps)
    train_size = int(len(full_dataset) * 0.8)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_states, batch_steps in train_loader:
            optimizer.zero_grad()
            predicted_steps = model(batch_states)
            loss = loss_fn(predicted_steps, batch_steps)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_epoch_loss:.4f}")

        # Save model checkpoint for the epoch
        torch.save(model.state_dict(), model_filename)
        
        # Append epoch information to log file
        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch+1}, Training Loss: {avg_epoch_loss:.4f}, Model: step_aware.pth\n")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_states, batch_steps in test_loader:
            predicted_steps = model(batch_states)
            loss = loss_fn(predicted_steps, batch_steps)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    return model


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log standard deviation

    def forward(self, state):
        mean = self.network(state)
        std = F.softplus(self.log_std)  # Correctly apply Softplus activation
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action, step_predicted):
        x = torch.cat([state, action, step_predicted], dim=1)
        return self.network(x)


def evaluate_policy(env, actor, num_episodes=10, video_folder="videos"):
    total_rewards = []
    actor.eval()
    os.makedirs(video_folder, exist_ok=True)

    for episode_index in range(num_episodes):
        frames = []
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        while not terminated and not truncated:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std = actor(state_tensor)
            dist = Normal(mean, std)
            action = dist.mean.cpu().numpy()[0]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            # Render the current frame and add it to the list
            frames.append(env.render())

        total_rewards.append(episode_reward)

        # Save video for the episode
        save_video(
            frames=frames,
            video_folder=video_folder,
            fps=env.metadata.get("render_fps", 30),
            step_starting_index=0,
            episode_index=episode_index
        )
    
    avg_reward = np.mean(total_rewards)
    print(f"Saved evaluation videos in '{video_folder}'")
    return avg_reward


def actor_critic_training(dataset, step_model, epochs=20, lr=1e-4, gamma=0.99, batch_size=256, env_name="D4RL/pen/expert-v2"):
    observations = torch.tensor(dataset['observations'], dtype=torch.float32)
    actions = torch.tensor(dataset['actions'], dtype=torch.float32)
    rewards = torch.tensor(dataset['rewards'], dtype=torch.float32).unsqueeze(-1)
    steps = torch.tensor(dataset['steps'], dtype=torch.float32).unsqueeze(-1)

    full_dataset = TensorDataset(observations, actions, rewards, steps)
    train_size = int(len(full_dataset) * 0.8)
    test_size = len(full_dataset) - train_size
    train_dataset, _ = random_split(full_dataset, [train_size, test_size])
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    state_dim = observations.shape[1]
    action_dim = actions.shape[1]

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    env = minari.load_dataset(env_name).recover_environment(render_mode="rgb_array")

    save_path = "c:/users/armin/step_aware/"
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, "actor_critic_log.txt")

    for epoch in range(epochs):
        actor.train()
        critic.train()
        epoch_actor_loss, epoch_critic_loss = 0, 0
        for batch_states, batch_actions, batch_rewards, batch_steps in data_loader:
            # Predict steps using the pre-trained StepPredictionModel
            step_predicted = step_model(batch_states)

            # Compute values
            predicted_values = critic(batch_states, batch_actions, step_predicted)
            advantages = batch_rewards - predicted_values.detach()

            # Update Actor
            mean, std = actor(batch_states)
            dist = Normal(mean, std)
            action_sample = dist.rsample()
            log_probs = dist.log_prob(action_sample).sum(dim=1, keepdim=True)
            entropy = dist.entropy().sum(dim=1, keepdim=True)
            actor_loss = -(log_probs * advantages + 0.01 * entropy).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update Critic
            target_values = batch_rewards + gamma * predicted_values.detach()
            critic_loss = mse_loss(predicted_values, target_values)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()

        # Evaluate Policy and save video
        avg_reward = evaluate_policy(env, actor)
        print(f"Epoch {epoch+1}/{epochs}, Actor Loss: {epoch_actor_loss/len(data_loader):.4f}, "
              f"Critic Loss: {epoch_critic_loss/len(data_loader):.4f}, Avg Reward: {avg_reward:.2f}")
        
        # Append epoch information to log file
        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch+1}, Actor Loss: {epoch_actor_loss/len(data_loader):.4f}, "
                      f"Critic Loss: {epoch_critic_loss/len(data_loader):.4f}, Avg Reward: {avg_reward:.2f}\n")
        
        # Save model checkpoints for each epoch
        torch.save(actor.state_dict(), os.path.join(save_path, f"actor.pth"))
        torch.save(critic.state_dict(), os.path.join(save_path, f"critic.pth"))

    return actor, critic


# Example usage:
if __name__ == "__main__":
    set_seed()
    dataset = download()[1]
    step_model = train_step_prediction_model(dataset, epochs=100)
    actor, critic = actor_critic_training(dataset, step_model, env_name='D4RL/pen/expert-v2', epochs=30)
