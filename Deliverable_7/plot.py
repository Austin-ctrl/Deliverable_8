import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def smooth(values, window):
    arr = np.array(values)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def plot_results(episodes, rewards, save_path="outputs/training_results.png"):
    steps = np.array(episodes)
    rews  = np.array(rewards)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("A3C on CartPole-v1", fontsize=14, fontweight='bold')

    # left plot - raw rewards + smoothed line
    ax = axes[0]
    ax.plot(steps, rews, alpha=0.2, color='steelblue', linewidth=0.6, label='Raw reward')
    s = smooth(rews, 30)
    ax.plot(steps[len(steps) - len(s):], s, color='steelblue', linewidth=2, label='30-ep avg')
    ax.fill_between(steps, rews, alpha=0.05, color='steelblue')
    ax.axhline(500, color='#27ae60', ls='--', alpha=0.8, label='Max (500)')
    ax.axhline(195, color='#e67e22', ls='--', alpha=0.8, label='Solved (195)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward over time")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # right plot - rolling averages to see convergence
    ax = axes[1]
    for window, color, label in [
        (20,  '#e74c3c', '20-ep avg'),
        (50,  '#2ecc71', '50-ep avg'),
        (100, '#3498db', '100-ep avg'),
    ]:
        if len(rews) >= window:
            s = smooth(rews, window)
            ax.plot(steps[len(steps) - len(s):], s, color=color, linewidth=1.8, label=label)
    ax.axhline(500, color='#27ae60', ls='--', alpha=0.8, label='Max (500)')
    ax.axhline(195, color='#e67e22', ls='--', alpha=0.8, label='Solved (195)')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Rolling averages")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"plot saved to {save_path}")