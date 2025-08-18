import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.live import Live
from rich.table import Table

hyperparams = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "steps_per_rollout": 2048,
    "update_epochs": 10,
    "minibatch_size": 64
}

total_timesteps = int(input("Enter total number of timesteps for training: ").strip())
env = gym.make("HalfCheetah-v5", render_mode="rgb_array")

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        hidden_size = 64
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.policy_mean = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor):
        features = self.shared_layers(state)
        mean = self.policy_mean(features)
        value = self.value_head(features).squeeze(-1)
        return mean, value

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
model = ActorCritic(obs_dim, act_dim)
optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_steps = 0
iteration = 0
hyperparam_change_log = []
recording = False
frames = []
video_count = 1

obs, info = env.reset()
obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

def create_table(iteration, total_steps, mean_reward, std_reward, policy_loss, value_loss, entropy, params):
    table = Table(title="PPO Training Progress", title_style="bold magenta")
    table.add_column("Iteration", justify="right")
    table.add_column("Total Steps", justify="right")
    table.add_column("Mean Reward", justify="right")
    table.add_column("Std Reward", justify="right")
    table.add_column("Policy Loss", justify="right")
    table.add_column("Value Loss", justify="right")
    table.add_column("Entropy", justify="right")
    table.add_column("Learning Rate", justify="right")
    table.add_column("Clip Range", justify="right")
    table.add_column("GAE Lambda", justify="right")
    table.add_column("Entropy Coef", justify="right")
    table.add_row(
        str(iteration),
        str(total_steps),
        f"{mean_reward:.2f}",
        f"{std_reward:.2f}",
        f"{policy_loss:.4f}",
        f"{value_loss:.4f}",
        f"{entropy:.4f}",
        f"{params['learning_rate']:.2e}",
        f"{params['clip_range']:.2f}",
        f"{params['gae_lambda']:.2f}",
        f"{params['entropy_coef']:.2f}"
    )
    return table

with Live(create_table(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, hyperparams), refresh_per_second=4) as live:
    while total_steps < total_timesteps:
        iteration += 1
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        episode_rewards = []
        ep_reward = 0.0

        for step in range(hyperparams["steps_per_rollout"]):
            obs_tensor = obs_tensor.to(device)
            with torch.no_grad():
                mean, value = model(obs_tensor)
                std = torch.exp(model.log_std)
                dist = torch.distributions.Normal(mean, std)
                action_tensor = dist.sample()
                action = action_tensor.cpu().numpy()
                log_prob = dist.log_prob(action_tensor).sum().item()
                value = value.item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if recording:
                frame = env.render()
                frames.append(frame)
            ep_reward += reward
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(log_prob)
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, info = env.reset()
            else:
                obs = next_obs
            total_steps += 1
            if total_steps >= total_timesteps:
                break
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        advantages = []
        gae = 0.0
        last_done = dones[-1] if len(dones) > 0 else False
        if not last_done:
            with torch.no_grad():
                _, last_val = model(obs_tensor)
                next_value = last_val.item()
        else:
            next_value = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                next_value = 0.0
            else:
                delta = rewards[t] + hyperparams["gamma"] * next_value - values[t]
            gae = delta + hyperparams["gamma"] * hyperparams["gae_lambda"] * gae
            advantages.insert(0, gae)
            next_value = values[t]
        returns = [adv + val for adv, val in zip(advantages, values)]

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        old_log_probs_tensor = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(np.array(returns), dtype=torch.float32, device=device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        for epoch in range(hyperparams["update_epochs"]):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), hyperparams["minibatch_size"]):
                end = start + hyperparams["minibatch_size"]
                batch_idx = indices[start:end]
                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_log = old_log_probs_tensor[batch_idx]
                batch_adv = advantages_tensor[batch_idx]
                batch_ret = returns_tensor[batch_idx]
                mean, value_pred = model(batch_states)
                std = torch.exp(model.log_std)
                dist = torch.distributions.Normal(mean, std)
                new_log_prob = dist.log_prob(batch_actions).sum(axis=-1)
                ratio = torch.exp(new_log_prob - batch_old_log)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - hyperparams["clip_range"], 1 + hyperparams["clip_range"]) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((value_pred.squeeze(-1) - batch_ret) ** 2).mean()
                entropy = dist.entropy().sum(axis=-1).mean()
                loss = policy_loss + hyperparams["value_coef"] * value_loss - hyperparams["entropy_coef"] * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            mean, value_pred = model(states_tensor)
            std = torch.exp(model.log_std)
            dist = torch.distributions.Normal(mean, std)
            new_log_prob = dist.log_prob(actions_tensor).sum(axis=-1)
            ratio = torch.exp(new_log_prob - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - hyperparams["clip_range"], 1 + hyperparams["clip_range"]) * advantages_tensor
            policy_loss_val = -torch.min(surr1, surr2).mean().item()
            value_loss_val = ((model(states_tensor)[1].squeeze(-1) - returns_tensor) ** 2).mean().item()
            entropy_val = dist.entropy().sum(axis=-1).mean().item()

        if episode_rewards:
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
        else:
            mean_reward = float(ep_reward)
            std_reward = 0.0

        table = create_table(iteration, total_steps, mean_reward, std_reward,
                             policy_loss_val, value_loss_val, entropy_val, hyperparams)
        live.update(table)

        live.stop()
        command = input("Enter command (key=value, 'record', 'save', 'quit' or Enter to continue): ").strip()
        live.start()

        if not command:
            continue
        
        tokens = [tok for tok in command.replace(",", " ").split() if tok]
        exit_flag = False
        for token in tokens:
            token_lower = token.lower()
            if token_lower == "record":
                if not recording:
                    recording = True
                    frames = []
                    live.console.print("[yellow]Recording started.[/yellow]")
                else:
                    recording = False
                    video_filename = f"ppo_demo_{video_count}.mp4"
                    try:
                        import imageio
                        imageio.mimwrite(video_filename, frames, fps=30, macro_block_size=None)
                        live.console.print(f"[green]Recording saved to {video_filename}[/green]")
                    except Exception as e:
                        live.console.print(f"[red]Failed to save video: {e}[/red]")
                    frames = []
                    video_count += 1
                continue
            if token_lower in ["quit", "exit", "stop"]:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "hyperparams": hyperparams,
                    "total_steps": total_steps,
                    "iteration": iteration
                }, "ppo_checkpoint_final.pth")
                live.console.print("[red]Training interrupted by user. Checkpoint saved as ppo_checkpoint_final.pth[/red]")
                exit_flag = True
                break
            if token_lower == "save":
                filename = f"ppo_checkpoint_iter{iteration}.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "hyperparams": hyperparams,
                    "total_steps": total_steps,
                    "iteration": iteration
                }, filename)
                live.console.print(f"[green]Checkpoint saved: {filename}[/green]")
                continue
            if "=" in token:
                key, value_str = token.split("=", 1)
                key = key.strip()
                value_str = value_str.strip()
                key_map = {
                    "lr": "learning_rate",
                    "clip": "clip_range",
                    "lambda": "gae_lambda",
                    "lam": "gae_lambda",
                    "gae_lambda": "gae_lambda",
                    "ent": "entropy_coef",
                    "ent_coef": "entropy_coef",
                    "entropy": "entropy_coef",
                    "entropy_coef": "entropy_coef",
                    "steps": "steps_per_rollout",
                    "batch_size": "steps_per_rollout",
                    "epochs": "update_epochs",
                    "gamma": "gamma",
                    "timesteps": "total_timesteps",
                    "total_timesteps": "total_timesteps"
                }
                actual_key = key_map.get(key.lower(), key)
                if actual_key in hyperparams or actual_key == "total_timesteps":
                    if actual_key in hyperparams:
                        current_val = hyperparams[actual_key]
                    else:
                        current_val = total_timesteps
                    try:
                        if isinstance(current_val, bool):
                            new_val = True if value_str.lower() in ["true", "1", "yes"] else False
                        elif isinstance(current_val, int) and ("." not in value_str):
                            new_val = int(value_str)
                        else:
                            new_val = float(value_str)
                    except:
                        live.console.print(f"[red]Invalid value for {actual_key}: {value_str}[/red]")
                        continue
                    old_val = current_val
                    if actual_key in hyperparams:
                        hyperparams[actual_key] = new_val
                        if actual_key == "learning_rate":
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_val
                    if actual_key == "total_timesteps":
                        total_timesteps = int(new_val)
                    hyperparam_change_log.append(f"Iteration {iteration}: {actual_key} changed from {old_val} to {new_val}")
                    live.console.print(f"[cyan]{actual_key} updated to {new_val}[/cyan]")
                else:
                    live.console.print(f"[red]Unknown hyperparameter: {key}[/red]")

        if exit_flag:
            break

torch.save({
    "model_state_dict": model.state_dict(),
    "hyperparams": hyperparams,
    "total_steps": total_steps,
    "iteration": iteration
}, "ppo_checkpoint_final.pth")

print(f"Training finished after {total_steps} timesteps (iterations: {iteration}). Final model saved to ppo_checkpoint_final.pth")
if hyperparam_change_log:
    print("\nHyperparameter changes during training:")
    for log_entry in hyperparam_change_log:
        print(log_entry)
env.close()
