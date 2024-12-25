import os
import random
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import minari

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-4
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 256
DEFAULT_ENV_NAME = "D4RL/door/expert-v2"
SAVE_PATH_WINDOWS = "C:/users/armin/step_aware"
SAVE_PATH_UNIX = "/home/armin/step_aware"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_rewards(rewards: np.ndarray) -> np.ndarray:
    """Normalize rewards to have zero mean and unit variance."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def preprocess_dataset(dataset) -> dict:
    """Preprocess the Minari dataset into a structured format."""
    keys = [
        'observations', 'actions', 'rewards', 'terminations', 'truncations',
        'next_observations', 'prev_observations', 'prev_actions',
        'prev_rewards', 'prev_dones', 'next_actions', 'step_numbers'
    ]
    data = {key: [] for key in keys}

    for episode_idx, episode in enumerate(dataset.iterate_episodes()):
        # Verify that all necessary attributes exist
        required_attrs = ['observations', 'actions', 'rewards', 'terminations', 'truncations']
        for attr in required_attrs:
            if not hasattr(episode, attr):
                raise AttributeError(f"Episode {episode_idx} is missing required attribute '{attr}'.")

        obs = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        truncations = episode.truncations

        # Ensure all lists have the same length
        lengths = [len(obs), len(actions), len(rewards), len(terminations), len(truncations)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Episode {episode_idx} has inconsistent lengths: "
                f"observations={len(obs)}, actions={len(actions)}, rewards={len(rewards)}, "
                f"terminations={len(terminations)}, truncations={len(truncations)}"
            )
        
        episode_length = len(obs)
        logger.info(f"Processing Episode {episode_idx}: Length={episode_length}")

        for i in range(episode_length):
            # Append current step data
            data['observations'].append(obs[i])
            data['actions'].append(actions[i])
            data['rewards'].append(rewards[i])
            data['terminations'].append(terminations[i])
            data['truncations'].append(truncations[i])

            # Handle next_observations
            if i + 1 < episode_length:
                next_obs = obs[i + 1]
            else:
                next_obs = np.zeros_like(obs[i])
            data['next_observations'].append(next_obs)

            # Handle next_actions
            if i + 1 < episode_length:
                next_act = actions[i + 1]
            else:
                next_act = np.zeros_like(actions[i])
            data['next_actions'].append(next_act)

            # Handle prev_observations
            if i > 0:
                prev_obs = obs[i - 1]
            else:
                prev_obs = np.zeros_like(obs[i])
            data['prev_observations'].append(prev_obs)

            # Handle prev_actions
            if i > 0:
                prev_act = actions[i - 1]
            else:
                prev_act = np.zeros_like(actions[i])
            data['prev_actions'].append(prev_act)

            # Handle prev_rewards
            if i > 0:
                prev_rew = rewards[i - 1]
            else:
                prev_rew = 0.0
            data['prev_rewards'].append(prev_rew)

            # Handle prev_dones
            if i > 0 and (terminations[i - 1] or truncations[i - 1]):
                prev_done = 1
            else:
                prev_done = 0
            data['prev_dones'].append(prev_done)

            # Step number
            data['step_numbers'].append(i + 1)  # Step number starting from 1

    # Convert lists to numpy arrays
    data['rewards'] = normalize_rewards(np.array(data['rewards']))
    data['prev_rewards'] = normalize_rewards(np.array(data['prev_rewards']))
    data['dones'] = np.logical_or(data['terminations'], data['truncations']).astype(int)

    # Ensure all data arrays have consistent shapes
    for key in keys:
        data[key] = np.array(data[key])
    data['dones'] = np.array(data['dones'])

    logger.info("Preprocessing completed successfully.")
    return data


def download_dataset(dataset_id: str) -> tuple:
    """Download and preprocess the Minari dataset."""
    logger.info(f"Loading dataset '{dataset_id}' from Minari...")
    dataset = minari.load_dataset(dataset_id, True)
    logger.info("Dataset loaded successfully.")
    processed_data = preprocess_dataset(dataset)
    return dataset, processed_data


class Actor(nn.Module):
    """Actor network for policy representation."""

    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, state: torch.Tensor) -> Normal:
        shared_output = self.shared_layers(state)
        mean = torch.tanh(self.mean_layer(shared_output))
        log_std = self.log_std_layer(shared_output).clamp(min=-20, max=2)
        return Normal(mean, torch.exp(log_std))

    def sample_action(self, state: torch.Tensor) -> tuple:
        dist = self(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """Critic network for value estimation."""

    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.reward_transform = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim + 4, 256),  # Include transformed reward
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        reward_features = self.reward_transform(reward).max(dim=-1, keepdim=True)[0]
        return self.network(torch.cat([state, action, reward_features], dim=1))


class StepPredictor(nn.Module):
    """Neural network to predict step number from state."""

    def __init__(self, state_dim: int):
        super(StepPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output step number as a scalar
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def evaluate_policy(actor: Actor, device: torch.device, env: gym.Env, save_path: str) -> float:
    """Evaluate the current policy and record a video."""
    seeds = [42]  # Define desired seeds
    total_rewards = []

    for seed in seeds:
        env = RecordVideo(env, video_folder=save_path, episode_trigger=lambda episode_id: True)
        actor.eval()
        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        with torch.no_grad():
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action, _ = actor.sample_action(state_tensor)
                action_np = action.cpu().numpy().flatten()
                state, reward, terminated, truncated, _ = env.step(action_np)
                total_reward += reward
                done = terminated or truncated

        total_rewards.append(total_reward)
        env.close()

    average_reward = sum(total_rewards) / len(total_rewards)
    logger.info(f"Average Reward over {len(seeds)} seeds: {average_reward}")
    return average_reward


def load_latest_models(save_path: str, actor: Actor, critic: Critic, device: torch.device) -> None:
    """Load the latest saved models from the specified path."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created save directory at {save_path}")
        return

    folders = [
        f for f in os.listdir(save_path)
        if os.path.isdir(os.path.join(save_path, f)) and "_" in f
    ]
    if not folders:
        logger.info("No saved models found. Starting from scratch.")
        return

    try:
        latest_folder = max(
            folders,
            key=lambda f: datetime.strptime(f.split('_')[-1], "%d%m%y")
        )
        actor_path = os.path.join(save_path, latest_folder, "actor.pth")
        critic_path = os.path.join(save_path, latest_folder, "critic.pth")

        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location=device))
            logger.info(f"Loaded actor model from {actor_path}")

        if os.path.exists(critic_path):
            critic.load_state_dict(torch.load(critic_path, map_location=device))
            logger.info(f"Loaded critic model from {critic_path}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Starting from scratch.")


def train_actor_critic(
    minari_dataset,
    dataset: dict,
    epochs: int,
    lr: float,
    gamma: float,
    batch_size: int,
    save_path: str,
    device: torch.device
) -> None:
    """Train the Actor-Critic models using the provided dataset."""
    state_dim = dataset['observations'].shape[1]
    action_dim = dataset['actions'].shape[1]

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    load_latest_models(save_path, actor, critic, device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    observations = torch.tensor(dataset['observations'], dtype=torch.float32)
    actions = torch.tensor(dataset['actions'], dtype=torch.float32)
    rewards = torch.tensor(dataset['rewards'], dtype=torch.float32).unsqueeze(-1)
    next_observations = torch.tensor(dataset['next_observations'], dtype=torch.float32)
    dones = torch.tensor(dataset['dones'], dtype=torch.float32).unsqueeze(-1)
    next_actions = torch.tensor(dataset['next_actions'], dtype=torch.float32)
    prev_observations = torch.tensor(dataset['prev_observations'], dtype=torch.float32)
    prev_actions = torch.tensor(dataset['prev_actions'], dtype=torch.float32)
    step_numbers = torch.tensor(dataset['step_numbers'], dtype=torch.float32).unsqueeze(-1)

    dataset_tensor = TensorDataset(
        observations, actions, rewards, next_observations,
        dones, prev_observations, prev_actions, next_actions, step_numbers
    )
    train_loader = DataLoader(
        dataset_tensor, batch_size=batch_size, shuffle=True, drop_last=True
    )

    log_file = os.path.join(save_path, "training_log.txt")
    with open(log_file, "a") as f:
        f.write("Starting training...\n")

    alpha = 0.001
    step = 0.001

    for epoch in range(1, epochs + 1):
        actor.train()
        critic.train()

        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        alpha += step  # Unused variable, consider removing if not needed

        for batch in train_loader:
            (
                states, actions_batch, rewards_batch, next_states,
                dones_batch, prev_states, prev_actions_batch,
                true_next_actions, step_numbers_batch
            ) = [x.to(device) for x in batch]

            # Critic Update
            with torch.no_grad():
                next_actions_pred, _ = actor.sample_action(next_states)
                target_q = rewards_batch + gamma * critic(next_states, next_actions_pred, rewards_batch) * (1 - dones_batch)

            current_q = critic(states, actions_batch, rewards_batch)
            critic_loss = loss_fn(current_q, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_optimizer.step()

            epoch_critic_loss += critic_loss.item()

            # Actor Update
            actions_new, _ = actor.sample_action(states)
            actor_loss = -critic(states, actions_new, rewards_batch).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_optimizer.step()

            epoch_actor_loss += actor_loss.item()

        avg_actor_loss = epoch_actor_loss / len(train_loader)
        avg_critic_loss = epoch_critic_loss / len(train_loader)

        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        epoch_save_folder = os.path.join(save_path, f"{timestamp}_epoch_{epoch}")
        os.makedirs(epoch_save_folder, exist_ok=True)

        env = minari_dataset.recover_environment(render_mode='rgb_array')
        eval_reward = evaluate_policy(actor, device, env, epoch_save_folder)
        env.close()

        with open(log_file, "a") as f:
            f.write(
                f"Epoch {epoch}/{epochs}, "
                f"Actor Loss: {avg_actor_loss:.4f}, "
                f"Critic Loss: {avg_critic_loss:.4f}, "
                f"Eval Reward: {eval_reward:.4f}\n"
            )

        torch.save(actor.state_dict(), os.path.join(epoch_save_folder, "actor.pth"))
        torch.save(critic.state_dict(), os.path.join(epoch_save_folder, "critic.pth"))

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Actor Loss: {avg_actor_loss:.4f} | "
            f"Critic Loss: {avg_critic_loss:.4f} | "
            f"Eval Reward: {eval_reward:.4f}"
        )


def get_save_path() -> str:
    """Determine the save path based on the operating system."""
    return SAVE_PATH_WINDOWS if os.name == "nt" else SAVE_PATH_UNIX


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Actor-Critic Models")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME,
                        help="Environment ID for Minari dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--save_path", type=str, default=get_save_path(),
        help="Path to save models and logs"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    set_seed(args.seed)

    minari_dataset, data = download_dataset(args.env_name)

    train_actor_critic(
        minari_dataset=minari_dataset,
        dataset=data,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        save_path=args.save_path,
        device=DEVICE
    )


if __name__ == "__main__":
    main()
