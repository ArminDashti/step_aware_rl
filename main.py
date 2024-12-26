import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo
from ila.datasets.minari_env import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-4
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 256
DEFAULT_ENV_NAME = "D4RL/door/expert-v2"
SAVE_PATH_WINDOWS = "C:/users/armin/step_aware"
SAVE_PATH_UNIX = "/home/armin/step_aware"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
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
    def forward(self, state):
        shared = self.shared_layers(state)
        mean = torch.tanh(self.mean_layer(shared))
        log_std = self.log_std_layer(shared).clamp(-20, 2)
        return Normal(mean, torch.exp(log_std))
    def sample_action(self, state):
        dist = self(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # self.steps_transform = nn.Sequential(
        #     nn.Linear(1, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 4),
        #     nn.ReLU(),
        #     nn.Softmax(dim=-1)
        # )
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action, steps):
        # steps_feat = self.steps_transform(steps)
        # max_steps_feat = steps_feat.argmax(dim=-1).unsqueeze(1)
        # print(max_steps_feat)
        return self.network(torch.cat([state, action], dim=1))

def evaluate_policy(actor, device, env, save_path):
    seeds = [42]
    total_rewards = []
    for seed in seeds:
        env = RecordVideo(env, video_folder=save_path, episode_trigger=lambda _: True)
        actor.eval()
        state, _ = env.reset(seed=seed)
        total_reward, done = 0.0, False
        with torch.no_grad():
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action, _ = actor.sample_action(state_tensor)
                state, reward, rermins, truncs, _ = env.step(action.cpu().numpy().flatten())
                total_reward += reward
                done = rermins or truncs
        total_rewards.append(total_reward)
        env.close()
    avg_reward = sum(total_rewards) / len(total_rewards)
    logger.info(f"Average Reward: {avg_reward}")
    return avg_reward

def load_latest_models(save_path, actor, critic, device):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created save directory at {save_path}")
        return
    folders = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f)) and "_" in f]
    if not folders:
        logger.info("No saved models found. Starting from scratch.")
        return
    try:
        latest_folder = max(folders, key=lambda f: datetime.strptime(f.split('_')[-1], "%d%m%y"))
        actor_path = os.path.join(save_path, latest_folder, "actor.pth")
        critic_path = os.path.join(save_path, latest_folder, "critic.pth")
        if os.path.exists(actor_path):
            actor.load_state_dict(torch.load(actor_path, map_location=device))
            logger.info(f"Loaded actor from {actor_path}")
        if os.path.exists(critic_path):
            critic.load_state_dict(torch.load(critic_path, map_location=device))
            logger.info(f"Loaded critic from {critic_path}")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Starting from scratch.")

def train_actor_critic(minari_dataset, dataset, epochs, lr, gamma, batch_size, save_path, device):
    state_dim = dataset['states'].shape[1]
    action_dim = dataset['actions'].shape[1]
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    load_latest_models(save_path, actor, critic, device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(
        torch.tensor(dataset['states'], dtype=torch.float32),
        torch.tensor(dataset['actions'], dtype=torch.float32),
        torch.tensor(dataset['rewards'], dtype=torch.float32).unsqueeze(-1),
        torch.tensor(dataset['next_states'], dtype=torch.float32),
        torch.tensor(dataset['dones'], dtype=torch.float32).unsqueeze(-1),
        torch.tensor(dataset['prev_states'], dtype=torch.float32),
        torch.tensor(dataset['prev_actions'], dtype=torch.float32),
        torch.tensor(dataset['next_actions'], dtype=torch.float32),
        torch.tensor(dataset['steps'], dtype=torch.float32).unsqueeze(-1)
    )
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    log_file = os.path.join(save_path, "training_log.txt")
    with open(log_file, "a") as f:
        f.write("Starting training...\n")
    for epoch in range(1, epochs + 1):
        actor.train()
        critic.train()
        epoch_actor_loss, epoch_critic_loss = 0.0, 0.0
        for batch in train_loader:
            states, actions_batch, rewards_batch, next_states, dones_batch, prev_states, prev_actions, next_actions, steps_batch = [x.to(device) for x in batch]
            with torch.no_grad():
                next_actions_pred, _ = actor.sample_action(next_states)
                target_q = rewards_batch + gamma * critic(next_states, next_actions_pred, steps_batch+1) * (1 - dones_batch)
            current_q = critic(states, actions_batch, steps_batch)
            critic_loss = loss_fn(current_q, target_q)
            # print('critic_loss', critic_loss.mean())

            prev_q = critic(prev_states, prev_actions, None)
            # print('prev_q', prev_q.mean())
            next_q = critic(next_states, next_actions, None)
            # print('next_q', next_q.mean())

            prev_q_safe = prev_q + 1e-6
            next_q_safe = next_q + 6
            ratio_prev = current_q / prev_q_safe
            ratio_next = current_q / next_q_safe
            deviation_lower = 0.5            # Lower bound of acceptable ratio
            deviation_upper = 1.5 
            penalty_prev = torch.relu(ratio_prev - deviation_upper) + torch.relu(deviation_lower - ratio_prev)
            penalty_next = torch.relu(ratio_next - deviation_upper) + torch.relu(deviation_lower - ratio_next)
            deviation_penalty = (penalty_prev + penalty_next) * (1 - dones_batch)
            critic_deviation_loss = deviation_penalty.mean()
            critic_loss += 1 * critic_deviation_loss


            critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            epoch_critic_loss += critic_loss.item()
            actions_new, _ = actor.sample_action(states)

            actor_loss = -critic(states, actions_new, steps_batch).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
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
            f.write(f"Epoch {epoch}/{epochs}, Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}, Eval Reward: {eval_reward:.4f}\n")
        torch.save(actor.state_dict(), os.path.join(epoch_save_folder, "actor.pth"))
        torch.save(critic.state_dict(), os.path.join(epoch_save_folder, "critic.pth"))
        logger.info(f"Epoch {epoch}/{epochs} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f} | Eval Reward: {eval_reward:.4f}")

def get_save_path():
    return SAVE_PATH_WINDOWS if os.name == "nt" else SAVE_PATH_UNIX

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Actor-Critic Models")
    parser.add_argument("--env_name", type=str, default=DEFAULT_ENV_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--save_path", type=str, default=get_save_path())
    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)
    minari = Dataset(DEFAULT_ENV_NAME,normalize_rewards=True)
    minari_dataset = minari.download()
    data = minari.download_processed()
    train_actor_critic(minari_dataset, data, args.epochs, args.lr, args.gamma, args.batch_size, args.save_path, DEVICE)

if __name__ == "__main__":
    main()
