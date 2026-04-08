# """
# Inference Script Example
# ===================================
# MANDATORY
# - Before submitting, ensure the following variables are defined in your environment configuration:
#     API_BASE_URL   The API endpoint for the LLM.
#     MODEL_NAME     The model identifier to use for inference.
#     HF_TOKEN       Your Hugging Face / API key.
#     LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
#                      method

# - Defaults are set only for API_BASE_URL and MODEL_NAME 
#     (and should reflect your active inference setup):
#     API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
#     MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
# - The inference script must be named `inference.py` and placed in the root directory of the project
# - Participants must use OpenAI Client for all LLM calls using above variables

# STDOUT FORMAT
# - The script must emit exactly three line types to stdout, in this order:

#     [START] task=<task_name> env=<benchmark> model=<model_name>
#     [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#     [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

#   Rules:
#     - One [START] line at episode begin.
#     - One [STEP] line per step, immediately after env.step() returns.
#     - One [END] line after env.close(), always emitted (even on exception).
#     - reward and rewards are formatted to 2 decimal places.
#     - done and success are lowercase booleans: true or false.
#     - error is the raw last_action_error string, or null if none.
#     - All fields on a single line with no newlines within a line.
#     - Each tasks should return score in [0, 1]

#   Example:
#     [START] task=click-test env=miniwob model=Qwen3-VL-30B
#     [STEP] step=1 action=click('123') reward=0.00 done=false error=null
#     [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
#     [STEP] step=3 action=click('789') reward=1.00 done=true error=null
#     [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
# """

# import asyncio
# import os
# import textwrap
# from typing import List, Optional

# from openai import OpenAI
# from dotenv import load_dotenv  # 1. Add this line
# load_dotenv()  # 2. Add this line to load your .env file
# # from my_env_v4 import MyEnvV4Action, MyEnvV4Env
# from models import FirstRlDemoAction
# from client import FirstRlDemoEnv
# IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
# API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
# MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
# TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
# BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
# MAX_STEPS = 8
# TEMPERATURE = 0.7
# MAX_TOKENS = 150
# SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# # Max possible reward: each token contributes 0.1, across all steps
# _MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
# MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

# SYSTEM_PROMPT = textwrap.dedent(
#     """
#     You are interacting with a simple echo environment.
#     Each turn you must send a message. The environment will echo it back.
#     Reward is proportional to message length: reward = len(message) * 0.1
#     Your goal is to maximize total reward by sending meaningful, substantive messages.
#     Reply with exactly one message string — no quotes, no prefixes, just the message text.
#     """
# ).strip()


# def log_start(task: str, env: str, model: str) -> None:
#     print(f"[START] task={task} env={env} model={model}", flush=True)


# def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
#     error_val = error if error else "null"
#     done_val = str(done).lower()
#     print(
#         f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
#         flush=True,
#     )


# def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
#     rewards_str = ",".join(f"{r:.2f}" for r in rewards)
#     print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
#     history_block = "\n".join(history[-4:]) if history else "None"
#     return textwrap.dedent(
#         f"""
#         Step: {step}
#         Last echoed message: {last_echoed!r}
#         Last reward: {last_reward:.2f}
#         Previous steps:
#         {history_block}
#         Send your next message.
#         """
#     ).strip()


# def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
#     user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
#     try:
#         completion = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user_prompt},
#             ],
#             temperature=TEMPERATURE,
#             max_tokens=MAX_TOKENS,
#             stream=False,
#         )
#         text = (completion.choices[0].message.content or "").strip()
#         return text if text else "hello"
#     except Exception as exc:
#         print(f"[DEBUG] Model request failed: {exc}", flush=True)
#         return "hello"


# async def main() -> None:
#     client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

#     env = await FirstRlDemoEnv.from_docker_image(IMAGE_NAME)

#     history: List[str] = []
#     rewards: List[float] = []
#     steps_taken = 0
#     score = 0.0
#     success = False

#     log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

#     try:
#         result = await env.reset() # OpenENV.reset()
#         last_echoed = result.observation.echoed_message
#         last_reward = 0.0

#         for step in range(1, MAX_STEPS + 1):
#             if result.done:
#                 break

#             message = get_model_message(client, step, last_echoed, last_reward, history)

#             result = await env.step(FirstRlDemoAction(message=message))
#             obs = result.observation

#             reward = result.reward or 0.0
#             done = result.done
#             error = None

#             rewards.append(reward)
#             steps_taken = step
#             last_echoed = obs.echoed_message
#             last_reward = reward

#             log_step(step=step, action=message, reward=reward, done=done, error=error)

#             history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

#             if done:
#                 break

#         score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
#         score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
#         success = score >= SUCCESS_SCORE_THRESHOLD

#     finally:
#         try:
#             await env.close()
#         except Exception as e:
#             print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
#         log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

from models import CameraAction, CameraObservation
from client import CameraPresetEnv

# Environment & Model Configuration
IMAGE_NAME = os.getenv("IMAGE_NAME", "camera_preset_demo-env:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "camera_ptz_alignment_v1"

# Evaluation Constants
MAX_STEPS = 20
TEMPERATURE = 0.0  # Precise control requires zero randomness
MAX_TOKENS = 150
SUCCESS_THRESHOLD = 0.05  # Distance error below which task is "Success"

# SYSTEM_PROMPT = textwrap.dedent(
#     """
#     You are an AI Camera Operator for a Hospital Security System. 
#     Your goal is to align a drifting PTZ (Pan-Tilt-Zoom) camera to its target preset.

#     INPUT:
#     - current_ptz: [pan, tilt, zoom] (Current camera position)
#     - target_ptz: [pan, tilt, zoom] (Desired preset position)
#     - distance: Euclidean distance to target.
    
#     TASK:
#     Output exactly three delta values to move the camera closer to the target.
    
#     FORMAT:
#     You MUST respond with a raw JSON object only:
#     {"delta_pan": float, "delta_tilt": float, "delta_zoom": float}
#     """
# ).strip()


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Camera Operator for a Hospital Security System. 
    Your goal is to align a drifting PTZ (Pan-Tilt-Zoom) camera to its target preset.

    INPUT:
    - current_ptz: [pan, tilt, zoom] (Current camera position)
    - target_ptz: [pan, tilt, zoom] (Desired preset position)
    - distance: Current Euclidean distance to target.
    
    STRATEGY:
    1. If distance > 0.2: Use AGGRESSIVE corrections. Deltas should be between 0.1 and 0.5 to close the gap quickly.
    2. If distance <= 0.2: Use PRECISION corrections. Deltas should be between 0.01 and 0.05 to avoid overshooting the target.
    3. Look at the difference between current and target for each axis (Pan, Tilt, Zoom) and move in the direction that reduces that difference.

    FORMAT:
    You MUST respond with a raw JSON object ONLY. No prose, no markdown blocks.
    {"delta_pan": float, "delta_tilt": float, "delta_zoom": float}
    """
).strip()

# --- LOGGING UTILITIES (Mandatory STDOUT Format) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- INFERENCE LOGIC ---

def parse_action(model_output: str) -> CameraAction:
    """Safely extracts JSON from model text."""
    try:
        clean_json = model_output.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return CameraAction(
            delta_pan=float(data.get("delta_pan", 0.0)),
            delta_tilt=float(data.get("delta_tilt", 0.0)),
            delta_zoom=float(data.get("delta_zoom", 0.0))
        )
    except Exception:
        return CameraAction(delta_pan=0.0, delta_tilt=0.0, delta_zoom=0.0)

async def run_eval_task(client: OpenAI, task_id: str) -> None:
    """Executes a single PTZ alignment task."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    # Initialize env via Docker image
    async with await CameraPresetEnv.from_docker_image(IMAGE_NAME) as env:
        # Reset and get initial drift
        result = await env.reset(task_id=task_id)
        obs = result.observation
        initial_dist = obs.distance_to_target

        for step in range(1, MAX_STEPS + 1):
            # 1. Model Call
            user_prompt = f"Current: {obs.current_ptz}\nTarget: {obs.target_ptz}\nDistance: {obs.distance_to_target:.4f}"
            
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw_output = completion.choices[0].message.content
                action = parse_action(raw_output)
                error_msg = None
            except Exception as e:
                action = CameraAction(delta_pan=0.0, delta_tilt=0.0, delta_zoom=0.0)
                error_msg = str(e)

            # 2. Step Environment
            result = await env.step(action)
            obs = result.observation
            
            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            # 3. Log Step
            action_str = f"move({action.delta_pan:.2f},{action.delta_tilt:.2f},{action.delta_zoom:.2f})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                success = obs.distance_to_target < SUCCESS_THRESHOLD
                break

        # Final Scoring (Efficiency Ratio)
        final_dist = obs.distance_to_target
        # Score is 0 to 1, based on how much distance was reduced relative to initial drift
        improvement = max(0, initial_dist - final_dist)
        score = min(1.0, improvement / initial_dist) if initial_dist > 0 else 1.0
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    # Global OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Run all three mandatory difficulty levels
    tasks = ["glitch_easy", "human_touch_medium", "hardware_drift_hard"]
    
    for task_name in tasks:
        try:
            await run_eval_task(client, task_name)
        except Exception as e:
            print(f"[DEBUG] Failed task {task_name}: {e}")

if __name__ == "__main__":
    asyncio.run(main())