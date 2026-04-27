import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.vision_encoder import DINOv2Encoder
from server.icu_env import ICUAction, HackathonICUEnv


def load_hf_auth_from_env() -> None:
    """
    Loads .env and maps HF token to the env vars used by huggingface_hub/transformers.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    except Exception:
        print("Warning: python-dotenv not available. Skipping .env loading.")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
    else:
        print("Warning: HF_TOKEN not found in .env. HF downloads will be unauthenticated.")


@dataclass
class PTZState:
    similarity_vector: torch.Tensor
    pan: float
    tilt: float
    score: float


class SimilarityProjector(nn.Module):
    """Projects Agent-1 similarity vector into virtual LLM token embeddings."""

    def __init__(self, sim_dim: int, hidden_size: int, num_virtual_tokens: int) -> None:
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(sim_dim, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, num_virtual_tokens * hidden_size),
        )

    def forward(self, similarity_vector: torch.Tensor) -> torch.Tensor:
        batch = similarity_vector.shape[0]
        out = self.net(similarity_vector)
        return out.view(batch, self.num_virtual_tokens, self.hidden_size)


class PTZPolicy(nn.Module):
    """
    Qwen policy that conditions on:
    1) similarity vector via virtual tokens
    2) text prompt with current pan/tilt
    """

    def __init__(
        self,
        model_name: str,
        sim_dim: int,
        num_virtual_tokens: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_size = self.model.config.hidden_size
        self.projector = SimilarityProjector(sim_dim, hidden_size, num_virtual_tokens)
        self.num_virtual_tokens = num_virtual_tokens

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _build_input_embeddings(
        self, similarity_vector: torch.Tensor, prompt_text: str, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.dtype]:
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        input_embed_layer = self.model.get_input_embeddings()
        model_embed_dtype = input_embed_layer.weight.dtype
        text_embeds = input_embed_layer(prompt_ids).to(dtype=model_embed_dtype)

        soft_prompt = self.projector(similarity_vector.unsqueeze(0).to(device)).to(dtype=model_embed_dtype)
        input_embeds = torch.cat([soft_prompt, text_embeds], dim=1)
        input_embeds = input_embeds.to(dtype=model_embed_dtype)
        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=device)
        prompt_len = input_embeds.shape[1]
        return input_embeds, attention_mask, prompt_len, model_embed_dtype

    def sample_action(
        self,
        state: PTZState,
        device: torch.device,
        max_new_tokens: int = 36,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> Dict[str, object]:
        prompt = (
            "You control a PTZ camera. Output ONLY JSON with keys "
            '"pan_delta" and "tilt_delta".\n'
            f"Current pan: {state.pan:.4f}\n"
            f"Current tilt: {state.tilt:.4f}\n"
            f"Current alignment score: {state.score:.6f}\n"
            "Goal: maximize alignment score to match the reference preset."
        )

        input_embeds, attention_mask, prompt_len, model_embed_dtype = self._build_input_embeddings(
            state.similarity_vector, prompt, device
        )
        with torch.no_grad():
            gen = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

        generated_ids = gen.sequences[:, 0:]
        # sequences do not include inputs_embeds tokens. We decode only generated output.
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # Recompute log-probs with grad enabled so policy loss can backpropagate.
        input_embed_layer = self.model.get_input_embeddings()
        generated_embeds = input_embed_layer(generated_ids.to(device)).to(dtype=model_embed_dtype)
        full_embeds = torch.cat([input_embeds, generated_embeds], dim=1)
        full_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=device)
        outputs = self.model(inputs_embeds=full_embeds, attention_mask=full_mask, return_dict=True)
        logits = outputs.logits[:, :-1, :]

        step_log_probs: List[torch.Tensor] = []
        gen_len = generated_ids.shape[1]
        for step in range(gen_len):
            # token at `step` is predicted from previous position => prompt_len-1+step.
            token_logits = logits[0, prompt_len - 1 + step, :]
            token_id = generated_ids[0, step].to(device)
            token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
            step_log_probs.append(token_log_prob)
        log_prob_sum = torch.stack(step_log_probs).sum() if step_log_probs else torch.tensor(0.0, device=device)

        action = parse_action(text)
        return {
            "prompt_len": prompt_len,
            "text": text,
            "action": action,
            "log_prob_sum": log_prob_sum,
            "token_count": max(1, len(step_log_probs)),
        }


class MuJoCoPTZAdapter:
    """Adapter that exposes PTZState from the real ICU MuJoCo env."""

    def __init__(
        self,
        sim_dim: int,
        reference_pan: float,
        reference_tilt: float,
        pan_delta_limit: float,
        tilt_delta_limit: float,
    ) -> None:
        self.env = HackathonICUEnv()
        self.encoder = DINOv2Encoder()
        self.sim_dim = sim_dim
        self.pan_delta_limit = pan_delta_limit
        self.tilt_delta_limit = tilt_delta_limit
        self.reference_pan = reference_pan
        self.reference_tilt = reference_tilt
        self.reference_image = self._capture_reference_image()

    def _render_current_image(self):
        self.env.renderer.update_scene(self.env.data, camera="robot_camera")
        return self.env.renderer.render()

    def _capture_reference_image(self):
        self.env.reset()
        self.env.data.ctrl[0] = self.reference_pan
        self.env.data.ctrl[1] = self.reference_tilt
        for _ in range(20):
            import mujoco as mj

            mj.mj_step(self.env.model, self.env.data)
        return self._render_current_image()

    def reset(self) -> PTZState:
        obs = self.env.reset()
        cur_img = self._render_current_image()
        v_delta, score = self.encoder.compute_delta_and_score(self.reference_image, cur_img)
        return PTZState(
            similarity_vector=self._to_sim_dim(v_delta),
            pan=float(obs.current_pan),
            tilt=float(obs.current_tilt),
            score=float(score),
        )

    def step(self, state: PTZState, pan_delta: float, tilt_delta: float) -> Tuple[PTZState, float]:
        pan_delta = float(max(-self.pan_delta_limit, min(self.pan_delta_limit, pan_delta)))
        tilt_delta = float(max(-self.tilt_delta_limit, min(self.tilt_delta_limit, tilt_delta)))

        target_pan = state.pan + pan_delta
        target_tilt = state.tilt + tilt_delta
        obs = self.env.step(ICUAction(pan_target=target_pan, tilt_target=target_tilt))
        cur_img = self._render_current_image()
        v_delta, score = self.encoder.compute_delta_and_score(self.reference_image, cur_img)

        next_state = PTZState(
            similarity_vector=self._to_sim_dim(v_delta),
            pan=float(obs.current_pan),
            tilt=float(obs.current_tilt),
            score=float(score),
        )
        # Use env reward (task objective) + small shaping from similarity improvement.
        reward = float(obs.reward) + 0.5 * float(score - state.score)
        return next_state, reward

    def _to_sim_dim(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec.detach().float().cpu()
        if vec.shape[0] == self.sim_dim:
            return vec
        if vec.shape[0] > self.sim_dim:
            return vec[: self.sim_dim]
        out = torch.zeros(self.sim_dim, dtype=torch.float32)
        out[: vec.shape[0]] = vec
        return out


def parse_action(text: str) -> Dict[str, float]:
    # Best case: model follows the requested JSON format.
    try:
        data = json.loads(text)
        pan_delta = float(data.get("pan_delta", 0.0))
        tilt_delta = float(data.get("tilt_delta", 0.0))
        return {"pan_delta": pan_delta, "tilt_delta": tilt_delta}
    except Exception:
        pass

    # Fallback: try to recover numbers from free text.
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    pan_delta = float(nums[0]) if len(nums) > 0 else 0.0
    tilt_delta = float(nums[1]) if len(nums) > 1 else 0.0
    return {"pan_delta": pan_delta, "tilt_delta": tilt_delta}


def grpo_update(
    policy: PTZPolicy,
    optimizer: torch.optim.Optimizer,
    samples: List[Dict[str, object]],
    kl_beta: float,
) -> Dict[str, float]:
    """
    GRPO-style objective:
    - sample a group of completions for same state
    - compute group-relative advantage: reward - group_mean(reward)
    - optimize logprob * advantage with simple KL anchor approximation
    """
    rewards = torch.tensor([float(s["reward"]) for s in samples], device=next(policy.parameters()).device)
    advantages = rewards - rewards.mean()
    if rewards.std() > 1e-6:
        advantages = advantages / (rewards.std() + 1e-6)

    log_probs = torch.stack([s["log_prob_sum"] / float(s["token_count"]) for s in samples])

    # Placeholder KL regularizer term; replace with reference-policy KL if needed.
    kl_term = torch.zeros((), device=log_probs.device)
    policy_loss = -(advantages.detach() * log_probs).mean() + kl_beta * kl_term

    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(policy_loss.detach().cpu()),
        "reward_mean": float(rewards.mean().cpu()),
        "reward_max": float(rewards.max().cpu()),
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    policy = PTZPolicy(
        model_name=args.model_name,
        sim_dim=args.sim_dim,
        num_virtual_tokens=args.num_virtual_tokens,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found. Disable --freeze-backbone if needed.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    env = MuJoCoPTZAdapter(
        sim_dim=args.sim_dim,
        reference_pan=args.reference_pan,
        reference_tilt=args.reference_tilt,
        pan_delta_limit=args.pan_delta_limit,
        tilt_delta_limit=args.tilt_delta_limit,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.train_steps + 1):
        state = env.reset()
        group_samples: List[Dict[str, object]] = []

        for _ in range(args.group_size):
            out = policy.sample_action(
                state=state,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            action = out["action"]
            _, reward = env.step(state, action["pan_delta"], action["tilt_delta"])
            out["reward"] = reward
            group_samples.append(out)

        metrics = grpo_update(policy, optimizer, group_samples, args.kl_beta)

        if step % args.log_every == 0:
            example = group_samples[0]
            print(
                f"step={step:05d} loss={metrics['loss']:.5f} "
                f"reward_mean={metrics['reward_mean']:.5f} reward_max={metrics['reward_max']:.5f}"
            )
            print(f"example action text: {example['text'][:180]}")

        if step % args.save_every == 0 or step == args.train_steps:
            ckpt = save_dir / f"ptz_qwen_grpo_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "projector": policy.projector.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt,
            )
            print(f"Saved checkpoint: {ckpt}")


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PTZ policy with Qwen2.5 + GRPO-style RL")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sim-dim", type=int, default=384, help="Dimension of Agent-1 similarity vector")
    parser.add_argument("--num-virtual-tokens", type=int, default=8)
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze Qwen backbone and train projector only (recommended first).",
    )
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--group-size", type=int, default=4, help="Number of completions per state for GRPO")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--max-new-tokens", type=int, default=36)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--reference-pan", type=float, default=0.0, help="Reference preset pan value")
    parser.add_argument("--reference-tilt", type=float, default=0.0, help="Reference preset tilt value")
    parser.add_argument("--pan-delta-limit", type=float, default=0.2, help="Clamp LLM pan delta action")
    parser.add_argument("--tilt-delta-limit", type=float, default=0.2, help="Clamp LLM tilt delta action")
    return parser.parse_args()


if __name__ == "__main__":
    load_hf_auth_from_env()
    args = build_args()
    train(args)
