import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lime.lime_tabular import LimeTabularExplainer
import multiprocessing
import time
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# ----------------------------
# Grid World Environment Functions
# ----------------------------
grid_size = 10

def pos_to_state(x, y):
    """Convert (x, y) position to state index."""
    return x * grid_size + y

def state_to_pos(state):
    """Convert state index to (x, y) position."""
    x = state // grid_size
    y = state % grid_size
    return x, y

def get_next_state(state, action):
    """
    Determine next state based on current state and action.
    Actions: 0 (right), 1 (left), 2 (up), 3 (down).
    """
    x, y = state_to_pos(state)
    next_x, next_y = x, y
    if action == 0:  # Right
        next_y = y + 1 if y + 1 < grid_size else y
    elif action == 1:  # Left
        next_y = y - 1 if y - 1 >= 0 else y
    elif action == 2:  # Up
        next_x = x - 1 if x - 1 >= 0 else x
    elif action == 3:  # Down
        next_x = x + 1 if x + 1 < grid_size else x
    return pos_to_state(next_x, next_y)

# ----------------------------
# Q-learning Agent
# ----------------------------
class Agent:
    def __init__(self, state_space, action_space):
        """Initialize the agent with state and action spaces."""
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((len(state_space), len(action_space)))
        self.visited_states = set()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        # ENHANCEMENT: Track total reward for benchmarking and distribution
        self.total_reward = 0

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (
            reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action]
        )

    def reward_function(self, current_state, previous_state):
        """Default reward function for abstract environment."""
        reward = -0.1
        if current_state not in self.visited_states:
            reward = 1
            self.visited_states.add(current_state)
        return reward

    def train(self, episodes, env_type='abstract', **kwargs):
        """
        Train the agent in specified environment.
        - env_type='grid': Grid world environment.
        - env_type='abstract': Original abstract environment.
        """
        if env_type == 'grid':
            max_steps = kwargs.get('max_steps', 100)
            start_state = kwargs.get('start_state', 0)
            goal_state = kwargs.get('goal_state', 99)
            total_rewards = []
            # ENHANCEMENT: Track unique states per episode for diversity plot
            unique_states_per_episode = []
            visited_in_episode = set()
            for episode in range(episodes):
                state = start_state
                done = False
                episode_reward = 0
                step = 0
                visited_in_episode.clear()
                while not done and step < max_steps:
                    action = self.choose_action(state)
                    next_state = get_next_state(state, action)
                    reward = 1 if next_state == goal_state else -0.01
                    episode_reward += reward
                    self.update_q_table(state, action, reward, next_state)
                    visited_in_episode.add(next_state)
                    state = next_state
                    if state == goal_state:
                        done = True
                    step += 1
                total_rewards.append(episode_reward)
                unique_states_per_episode.append(len(visited_in_episode))
                self.epsilon = max(self.min_epsilon, self.epsilon_decay * self.epsilon)
                # ENHANCEMENT: Accumulate total reward
                self.total_reward += episode_reward
            return total_rewards, unique_states_per_episode
        elif env_type == 'abstract':
            for episode in range(episodes):
                state = random.choice(range(len(self.state_space)))
                done = False
                while not done:
                    action = self.choose_action(state)
                    next_state = random.choice(range(len(self.state_space)))
                    reward = self.reward_function(next_state, state)
                    self.update_q_table(state, action, reward, next_state)
                    state = next_state
                    if state == len(self.state_space) - 1:
                        done = True
                self.epsilon = max(self.min_epsilon, self.epsilon_decay * self.epsilon)
            return None, None

# ----------------------------
# RLNeuronLayer with Exploration-Exploitation
# ----------------------------
class RLNeuronLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=1.0, beta=0.1):
        super(RLNeuronLayer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        output = torch.matmul(x, self.weights) + self.bias
        output = torch.sigmoid(output)
        return output

    def compute_task_reward(self, output, target):
        return -torch.mean((output - target) ** 2)

    def compute_exploration_reward(self, epsilon, random_action):
        return epsilon if random_action else 0

    def compute_meta_loss(self, output):
        return self.beta * torch.sum(torch.abs(self.weights))

# ----------------------------
# Attention Layer for Global Explanations
# ----------------------------
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        attention_weights = torch.softmax(self.attention_weights, dim=0)
        weighted_input = x * attention_weights
        return weighted_input

# ----------------------------
# RLFeedForwardNetworkWithAttention
# ----------------------------
class RLFeedForwardNetworkWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=1.0, beta=0.1):
        super(RLFeedForwardNetworkWithAttention, self).__init__()
        self.attention_layer = AttentionLayer(input_dim)
        self.hidden_layer = RLNeuronLayer(input_dim, hidden_dim, alpha, beta)
        self.output_layer = RLNeuronLayer(hidden_dim, output_dim, alpha, beta)

    def forward(self, x):
        x = self.attention_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

    def compute_task_reward(self, output, target):
        return self.output_layer.compute_task_reward(output, target)

    def compute_exploration_reward(self, epsilon, random_action):
        return self.output_layer.compute_exploration_reward(epsilon, random_action)

    def compute_meta_loss(self, output):
        return self.output_layer.compute_meta_loss(output)

# ----------------------------
# LIME Integration for Local Explanations
# ----------------------------
def explain_with_lime(model, data_point, training_data, feature_names):
    explainer = LimeTabularExplainer(training_data, training_labels=None, feature_names=feature_names, class_names=['0', '1'])
    explanation = explainer.explain_instance(data_point, model.predict, num_features=5)
    explanation.show_in_notebook()

# ----------------------------
# Training Function for RL-based Network
# ----------------------------
def rl_train_network(agent, model, optimizer, data_loader, episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # ENHANCEMENT: Track meta-loss and task error for visualization
    meta_losses = []
    task_errors = []
    # ENHANCEMENT: Track attention weights for dynamics visualization
    attention_weights_over_time = []
    for episode in range(episodes):
        model.train()
        total_loss = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            task_reward = model.compute_task_reward(output, target)
            epsilon = agent.epsilon
            random_action = random.uniform(0, 1) < epsilon
            exploration_reward = model.compute_exploration_reward(epsilon, random_action)
            meta_loss = model.compute_meta_loss(output)
            total_reward = task_reward + exploration_reward - meta_loss
            loss = -total_reward
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_state = random.choice(range(len(agent.state_space)))
            action = agent.choose_action(current_state)
            agent.update_q_table(current_state, action, total_reward, current_state)
            # ENHANCEMENT: Log metrics
            meta_losses.append(meta_loss.item())
            task_errors.append(-task_reward.item())
            attention_weights = model.attention_layer.attention_weights.detach().cpu().numpy()
            attention_weights_over_time.append(attention_weights.copy())
        agent.epsilon = max(agent.min_epsilon, agent.epsilon_decay * agent.epsilon)
        if episode % 10 == 0:
            print(f"Episode {episode}/{episodes}, Loss: {total_loss / len(data_loader)}")
    return meta_losses, task_errors, attention_weights_over_time

# ----------------------------
# Parallel Simulation of Multiple Agents
# ----------------------------
def run_agent_simulation(agent, episodes):
    total_rewards, unique_states_per_episode = agent.train(episodes, env_type='grid', start_state=0, goal_state=99)
    return agent.q_table, total_rewards, agent.visited_states, unique_states_per_episode, agent.total_reward

def run_multiple_simulations(num_agents, state_space, action_space, episodes=1000):
    agents = [Agent(state_space, action_space) for _ in range(num_agents)]
    with multiprocessing.Pool() as pool:
        results = pool.starmap(run_agent_simulation, [(agent, episodes) for agent in agents])
    q_tables, total_rewards_list, visited_states_sets, unique_states_lists, total_rewards = zip(*results)
    aggregated_q_table = np.mean(q_tables, axis=0)
    average_total_rewards = np.mean(total_rewards_list, axis=0)
    all_visited_states = set().union(*visited_states_sets)
    average_unique_states_per_episode = np.mean(unique_states_lists, axis=0)
    agent_total_rewards = total_rewards
    return aggregated_q_table, average_total_rewards, all_visited_states, average_unique_states_per_episode, agent_total_rewards

# ----------------------------
# Visualization Functions
# ----------------------------
def visualize_q_table(q_table, grid_size=10):
    """Visualize maximum Q-value for each state as a heatmap."""
    max_q = np.max(q_table, axis=1).reshape((grid_size, grid_size))
    plt.figure(figsize=(6, 6))
    sns.heatmap(max_q, cmap='coolwarm')
    plt.title("Max Q-Value Heatmap")
    plt.show()

def plot_rewards(average_total_rewards):
    """Plot average total rewards per episode."""
    plt.figure(figsize=(6, 6))
    plt.plot(average_total_rewards)
    plt.title("Average Total Rewards Per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.show()

# ENHANCEMENT: Exploration Diversity Plot
def plot_exploration_diversity(unique_states_per_episode):
    """Plot the number of unique states visited over episodes."""
    plt.figure(figsize=(6, 6))
    plt.plot(unique_states_per_episode)
    plt.title("Exploration Diversity Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Unique States Visited")
    plt.show()

# ENHANCEMENT: Reward Distribution Histogram
def plot_reward_distribution(agent_total_rewards):
    """Plot the distribution of total rewards across all agents."""
    plt.figure(figsize=(6, 6))
    plt.hist(agent_total_rewards, bins=20)
    plt.title("Reward Distribution Across Agents")
    plt.xlabel("Total Reward")
    plt.ylabel("Number of Agents")
    plt.show()

# ENHANCEMENT: Meta-Loss Impact Plot
def plot_meta_loss_impact(meta_losses, task_errors):
    """Plot meta-loss and task error over training steps."""
    plt.figure(figsize=(6, 6))
    plt.plot(meta_losses, label="Meta-Loss")
    plt.plot(task_errors, label="Task Error")
    plt.title("Meta-Loss vs. Task Error")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# ENHANCEMENT: Policy Entropy Heatmap
def plot_policy_entropy(q_table, grid_size=10):
    """Plot policy entropy across the state space."""
    entropy_grid = np.zeros((grid_size, grid_size))
    for state in range(len(q_table)):
        q_values = torch.tensor(q_table[state], dtype=torch.float32)
        probs = F.softmax(q_values, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        x, y = state_to_pos(state)
        entropy_grid[x, y] = entropy
    plt.figure(figsize=(6, 6))
    sns.heatmap(entropy_grid, cmap='viridis')
    plt.title("Policy Entropy Heatmap")
    plt.show()

# ENHANCEMENT: Attention Dynamics Animation (Simplified)
def visualize_attention_dynamics(attention_weights_over_time):
    """Visualize attention weights over training episodes."""
    final_weights = attention_weights_over_time[-1]
    plt.figure(figsize=(6, 6))
    plt.bar(range(len(final_weights)), final_weights)
    plt.title("Final Attention Weights")
    plt.xlabel("Feature Index")
    plt.ylabel("Attention Weight")
    plt.show()

# ENHANCEMENT: Benchmarking Against Vanilla Q-Learning
def benchmark_vanilla_q_learning(state_space, action_space, episodes=1000):
    """Train a single vanilla Q-learning agent for benchmarking."""
    vanilla_agent = Agent(state_space, action_space)
    total_rewards, _ = vanilla_agent.train(episodes, env_type='grid', start_state=0, goal_state=99)
    return vanilla_agent.total_reward

class MovingAgent:
    def __init__(self, x, y, label):
        self.position = np.array([x, y], dtype=float)
        self.label = label

    def update_position(self, q_table):
        """Update position based on learned Q-table policy."""
        i = int(self.position[0] / 10)
        j = int(self.position[1] / 10)
        state = pos_to_state(i, j)
        action = np.argmax(q_table[state])
        next_state = get_next_state(state, action)
        next_i, next_j = state_to_pos(next_state)
        self.position = np.array([next_i * 10 + 5, next_j * 10 + 5])

def visualize_agents(num_agents=100, steps=200, interval=50, q_table=None):
    """Visualize agents moving according to the learned Q-table policy."""
    agents = []
    for i in range(num_agents):
        x_idx = np.random.randint(0, grid_size)
        y_idx = np.random.randint(0, grid_size)
        x = x_idx * 10 + 5
        y = y_idx * 10 + 5
        agents.append(MovingAgent(x, y, f"A{i}"))
    positions = np.array([agent.position for agent in agents])

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue')
    labels = [ax.text(pos[0], pos[1], agent.label, fontsize=8, color='red') for agent, pos in zip(agents[:5], positions[:5])]
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Agent Movements Following Learned Policy")
    ax.set_facecolor('lightgreen')

    def update(frame):
        for agent in agents:
            agent.update_position(q_table)
        positions = np.array([agent.position for agent in agents])
        scatter.set_offsets(positions)
        for i, txt in enumerate(labels):
            txt.set_position((positions[i, 0], positions[i, 1]))
        step_text.set_text(f"Step: {frame}")
        return scatter, step_text, *labels

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=True)
    plt.show()

# ----------------------------
# Main Execution Block
# ----------------------------
if __name__ == '__main__':
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Define state and action spaces for grid world
    grid_state_space = range(grid_size * grid_size)
    grid_action_space = range(4)

    # Run parallel simulations for grid world Q-learning agents
    print("Running parallel simulations for grid world Q-learning agents...")
    num_agents = 1000
    episodes = 1000
    aggregated_q_table, average_total_rewards, all_visited_states, average_unique_states_per_episode, agent_total_rewards = run_multiple_simulations(
        num_agents=num_agents,
        state_space=grid_state_space,
        action_space=grid_action_space,
        episodes=episodes
    )
    print("Aggregated Q-Table from parallel simulations:")
    print(aggregated_q_table)

    # Visualizations from original code
    visualize_q_table(aggregated_q_table)
    plot_rewards(average_total_rewards)

    # ENHANCEMENT: Exploration Diversity Plot
    plot_exploration_diversity(average_unique_states_per_episode)

    # ENHANCEMENT: Reward Distribution Histogram
    plot_reward_distribution(agent_total_rewards)

    # ENHANCEMENT: Policy Entropy Heatmap
    plot_policy_entropy(aggregated_q_table)

    # Define state and action spaces for RL network training
    rl_state_space = range(10)
    rl_action_space = range(4)

    # Prepare dummy data for training the RL-based network
    X_data = torch.randn(100, 2)
    y_data = torch.randint(0, 2, (100, 1)).float()
    dataset = TensorDataset(X_data, y_data)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Initialize a Q-learning agent for network training
    rl_agent = Agent(rl_state_space, rl_action_space)

    # Initialize the RL-based network with attention
    input_dim = 2
    hidden_dim = 4
    output_dim = 1
    model_with_attention = RLFeedForwardNetworkWithAttention(input_dim, hidden_dim, output_dim)
    optimizer = optim.SGD(model_with_attention.parameters(), lr=0.01)

    # Train the network with enhanced logging
    print("Training RL-based network with attention...")
    meta_losses, task_errors, attention_weights_over_time = rl_train_network(rl_agent, model_with_attention, optimizer, data_loader, episodes=100)

    # ENHANCEMENT: Meta-Loss Impact Plot
    plot_meta_loss_impact(meta_losses, task_errors)

    # ENHANCEMENT: Visualize Attention Dynamics
    visualize_attention_dynamics(attention_weights_over_time)

    # LIME Explanation Example
    X_train_np = X_data.numpy()
    def predict_fn(X):
        model_with_attention.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            output = model_with_attention(X_tensor)
            p = output.numpy()
            probs = np.hstack([1 - p, p])
            return probs
    feature_names = ['Feature1', 'Feature2']
    explainer = LimeTabularExplainer(X_train_np, training_labels=None, feature_names=feature_names, class_names=['0', '1'])
    instance_to_explain = X_train_np[0]
    explanation = explainer.explain_instance(instance_to_explain, predict_fn, num_features=2)
    explanation.show_in_notebook()  # In a Jupyter Notebook environment

    # Visualize agent movements following learned policy
    print("Visualizing agent movements following learned policy...")
    visualize_agents(num_agents=100, steps=200, interval=50, q_table=aggregated_q_table)

    # ENHANCEMENT: Benchmarking Against Vanilla Q-Learning
    print("Running benchmark with vanilla Q-learning...")
    vanilla_total_reward = benchmark_vanilla_q_learning(grid_state_space, grid_action_space, episodes=episodes)
    print(f"Total reward with enhancements: {sum(agent_total_rewards) / num_agents}, Vanilla Q-learning: {vanilla_total_reward}")