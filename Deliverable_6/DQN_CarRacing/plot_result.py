import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless – works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--log",     default="logs/training_log.csv")
parser.add_argument("--out_dir", default="plots")
parser.add_argument("--smooth",  type=int, default=20,
                    help="Rolling-average window size (episodes)")
args = parser.parse_args()
 
os.makedirs(args.out_dir, exist_ok=True)
 
# ── Load CSV ──────────────────────────────────────────────────────────────────
episodes, rewards, losses, epsilons, steps = [], [], [], [], []
 
with open(args.log, newline="") as f:
    reader = csv.DictReader(f)
    print(reader.fieldnames)  
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["reward"]))
        epsilons.append(float(row["epsilon"]))
        steps.append(int(row["total_steps"]))
 
episodes  = np.array(episodes)
rewards   = np.array(rewards)
losses    = np.array(losses)
epsilons  = np.array(epsilons)
steps     = np.array(steps)
 
 
def smooth(x, w):
    """Simple rolling mean (same length as x, edges use available data)."""
    return np.convolve(x, np.ones(w) / w, mode="same")
 
 
# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
    "font.size":        11,
})
 
BLUE  = "#58a6ff"
GREEN = "#3fb950"
AMBER = "#d29922"
RED   = "#f85149"
 
 

# Figure 1  –  2×2 dashboard  (mirrors paper Figure 2)

fig = plt.figure(figsize=(14, 9))
fig.suptitle("DQN Training Dashboard – CarRacing-v2", fontsize=15, weight="bold",
             color="#e6edf3", y=0.98)
 
gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                       left=0.08, right=0.97, top=0.93, bottom=0.08)
 
#1. Reward per episode
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(episodes, rewards,          alpha=0.25, color=BLUE,  lw=0.8, label="raw")
ax1.plot(episodes, smooth(rewards, args.smooth), color=BLUE, lw=2,
         label=f"{args.smooth}-ep avg")
ax1.set_title("Reward per Episode", weight="bold")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.legend(framealpha=0.3, fontsize=9)
ax1.grid(True)
 
 
#4. Epsilon decay
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(episodes, epsilons, color=AMBER, lw=2)
ax4.axhline(0.1, color="#8b949e", lw=1, linestyle="--", label="ε floor = 0.1")
ax4.set_title("Exploration Rate (ε)", weight="bold")
ax4.set_xlabel("Episode")
ax4.set_ylabel("Epsilon")
ax4.set_ylim(0, 1.05)
ax4.legend(framealpha=0.3, fontsize=9)
ax4.grid(True)
 
out1 = os.path.join(args.out_dir, "training_dashboard.png")
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved: {out1}")
 
 

# Figure 2  –  Reward vs Total Environment Steps  (x-axis = steps, like paper)

fig2, ax = plt.subplots(figsize=(10, 4.5),
                        facecolor="#0d1117")
ax.set_facecolor("#161b22")
 
ax.plot(steps, rewards,           alpha=0.2, color=BLUE, lw=0.8)
ax.plot(steps, smooth(rewards, args.smooth), color=BLUE, lw=2.5,
        label=f"{args.smooth}-ep rolling avg")
 
ax.set_title("Reward vs Environment Steps", weight="bold", fontsize=13)
ax.set_xlabel("Total Steps")
ax.set_ylabel("Episode Reward")
ax.legend(framealpha=0.3)
ax.grid(True, color="#21262d", linestyle="--", linewidth=0.6)
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
 
fig2.tight_layout()
out2 = os.path.join(args.out_dir, "reward_vs_steps.png")
fig2.savefig(out2, dpi=150)
plt.close(fig2)
print(f"Saved: {out2}")
 

 
print("\nAll plots saved to:", args.out_dir)