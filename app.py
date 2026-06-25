import os
import tempfile
import numpy as np
import gradio as gr
import gymnasium as gym
from stable_baselines3 import SAC
import imageio

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/reacher_trained.zip"
ENV_ID     = "Reacher-v5"
N_EPISODES = 3
FPS        = 30

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def run_policy(n_episodes: int = N_EPISODES) -> tuple[str, str]:
    """Load trained SAC model, roll out n episodes, return (video_path, stats_md)."""
    model = SAC.load(MODEL_PATH)

    env = gym.make(ENV_ID, render_mode="rgb_array")
    frames, episode_rewards = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            frame = env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        episode_rewards.append(ep_reward)

    env.close()

    # Writing video
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    imageio.mimwrite(tmp.name, frames, fps=FPS, quality=8)

    # Stats markdown
    mean_r = np.mean(episode_rewards)
    std_r  = np.std(episode_rewards)
    rows   = "\n".join(
        f"| Episode {i+1} | `{r:.2f}` |"
        for i, r in enumerate(episode_rewards)
    )
    stats = f"""
### Policy performance ({n_episodes} episodes)

| | Reward |
|---|---|
{rows}
| **Mean** | **`{mean_r:.2f}`** |
| **Std** | `±{std_r:.2f}` |

> Trained with SAC · 200 000 steps · MuJoCo Reacher-v5  
> Near-optimal benchmark reward: **−3.33 ± 1.37**
"""
    return tmp.name, stats


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'Space Grotesk', sans-serif; }
#title { text-align: center; margin-bottom: 0.25rem; }
#subtitle { text-align: center; color: #888; margin-top: 0; }
#run-btn { background: #c0572b !important; border: none !important; }
"""

with gr.Blocks(css=CSS, title="Robotic Arm RL Demo") as demo:

    gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">
        <h1 id="title">🤖 Robotic Arm RL</h1>
        <p id="subtitle">SAC agent solving MuJoCo Reacher-v5 · trained by <a href="https://github.com/av0422" target="_blank">av0422</a></p>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            video_out = gr.Video(label="Policy rollout", autoplay=True)
        with gr.Column(scale=1):
            stats_out = gr.Markdown(label="Episode stats")

    with gr.Row():
        n_eps_slider = gr.Slider(
            minimum=1, maximum=5, value=3, step=1,
            label="Number of episodes to render"
        )
        run_btn = gr.Button("▶  Run policy", elem_id="run-btn", variant="primary")

    gr.Markdown("""
---
**How it works**  
A Soft Actor-Critic (SAC) agent learns to move a 2-DOF arm to a randomly placed
target using continuous joint torques. SAC maximises both reward *and* entropy,
making it sample-efficient and robust to environment noise.

**Key results**
- Mean reward **−3.33 ± 1.37** after 200k steps  
- Beats the random-policy baseline (≈ −20) by **~6×**

**Stack:** Python · MuJoCo · Gymnasium · Stable-Baselines3
""")

    run_btn.click(
        fn=run_policy,
        inputs=[n_eps_slider],
        outputs=[video_out, stats_out],
    )

    # Auto-run on load
    demo.load(fn=run_policy, inputs=[n_eps_slider], outputs=[video_out, stats_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
