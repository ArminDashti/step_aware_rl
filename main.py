import minari
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal
import gymnasium as gym


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
        for i in range(100):
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

def train_step_prediction_model(dataset, epochs=10, lr=1e-3, batch_size=64):
    observations = torch.tensor(dataset['observations'], dtype=torch.float32)
    steps = torch.tensor(dataset['steps'], dtype=torch.float32).unsqueeze(-1)

    full_dataset = TensorDataset(observations, steps)
    train_size = int(len(full_dataset) * 0.8)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    state_dim = observations.shape[1]
    model = StepPredictionModel(state_dim)
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

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_states, batch_steps in test_loader:
            predicted_steps = model(batch_states)
            loss = loss_fn(predicted_steps, batch_steps)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    return model

class StepPredictionModel(nn.Module):
    def __init__(self, state_dim):
        super(StepPredictionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 128),  # state, action, step_predicted
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action, step_predicted):
        x = torch.cat([state, action, step_predicted], dim=1)
        return self.network(x)

def evaluate_policy(env, actor, num_episodes=10):
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        while not terminated and not truncated:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std = actor(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()[0].detach().numpy()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    avg_reward = np.mean(total_rewards)
    return avg_reward

def actor_critic_training(dataset, step_model, epochs=10, lr=1e-3, gamma=0.99, batch_size=64, env_name="CartPole-v1"):
    observations = torch.tensor(dataset['observations'], dtype=torch.float32)
    actions = torch.tensor(dataset['actions'], dtype=torch.float32)
    rewards = torch.tensor(dataset['rewards'], dtype=torch.float32).unsqueeze(-1)
    steps = torch.tensor(dataset['steps'], dtype=torch.float32).unsqueeze(-1)

    full_dataset = TensorDataset(observations, actions, rewards, steps)
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    state_dim = observations.shape[1]
    action_dim = actions.shape[1]

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    env = minari.load_dataset(env_name).recover_environment()

    for epoch in range(epochs):
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
            actor_loss = -(log_probs * advantages).mean()

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
        
        # Evaluate Policy
        avg_reward = evaluate_policy(env, actor)
        print(f"Epoch {epoch+1}/{epochs}, Actor Loss: {epoch_actor_loss/len(data_loader):.4f}, Critic Loss: {epoch_critic_loss/len(data_loader):.4f}, Avg Reward: {avg_reward:.2f}")

    return actor, critic

# Example usage:
dataset = download()[1]
step_model = train_step_prediction_model(dataset)
actor, critic = actor_critic_training(dataset, step_model, env_name='D4RL/pen/expert-v2')
