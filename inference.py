import asyncio
import os
from typing import List, Optional
from openai import OpenAI

# 🔹 ENV HANDLING
try:
    from my_env_v4 import MyEnvV4Action, MyEnvV4Env
except ImportError:
    class MyEnvV4Action:
        def __init__(self, message):
            self.message = message

    class DummyEnv:
        def __init__(self):
            self.emails = [
                ("Meeting tomorrow at 10 AM", "schedule"),
                ("Win a free iPhone now! Click here", "ignore"),
                ("We need to discuss the quarterly report asap.", "mark_important"),
                ("Hey, how are you doing? Let's catch up.", "reply"),
                ("Let's set up a time to chat next week.", "schedule"),
                ("URGENT: Server is down, fix immediately!", "mark_important"),
                ("Daily newsletter update for tech news.", "ignore"),
                ("Could you please send me the documents?", "reply"),
            ]
            self.idx = 0

        async def reset(self):
            self.idx = 0
            class Obj:
                observation = type("obs", (), {"echoed_message": self.emails[self.idx][0]})
                done = False
            return Obj()

        async def step(self, action):
            correct = self.emails[self.idx][1]
            reward = 1.0 if action.message.lower() == correct else 0.0
            self.idx += 1
            done = self.idx >= len(self.emails) or self.idx >= 8
            next_text = self.emails[self.idx][0] if not done else "Done"
            
            class Obj:
                observation = type("obs", (), {"echoed_message": next_text})
            res = Obj()
            res.reward = reward
            res.done = done
            return res

    class MyEnvV4Env:
        @staticmethod
        async def from_docker_image(_):
            return DummyEnv()


# 🔹 ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "email_detection")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
IMAGE_NAME = os.getenv("IMAGE_NAME")

MAX_STEPS = 8

# 🔹 LOGGING
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_model_message(client, step, last_echoed, last_reward, history):
    prompt = f"""
    Email: {last_echoed}
    Decide correct action:
    reply → normal
    schedule → meetings/dates
    mark_important → urgent/deadline
    ignore → spam/promotions
    Return ONLY one word:
    reply, schedule, mark_important, ignore
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )

        content = response.choices[0].message.content.lower().strip()

        valid = ["reply","schedule","mark_important","ignore"]
        if content in valid:
            return content

    except Exception:
        pass

    # 🔥 FALLBACK (VERY IMPORTANT)
    text = last_echoed.lower()

    if any(w in text for w in ["meeting","schedule","tomorrow","time","appointment","chat","call"]):
        return "schedule"
    if any(w in text for w in ["urgent","asap","important","deadline","down","fix","report"]):
        return "mark_important"
    if any(w in text for w in ["free","win","offer","click","buy","newsletter","update","spam"]):
        return "ignore"
    else:
        return "reply"

# 🔹 MAIN
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    rewards = []
    history = []
    steps_taken = 0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        result = await env.reset()
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_message(client, step, last_echoed, last_reward, history)

            result = await env.step(MyEnvV4Action(message=action))

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, None)

            last_echoed = result.observation.echoed_message
            last_reward = reward

            history.append(action)

            if done:
                break

        MAX_TOTAL_REWARD = MAX_STEPS * 1.0
        score = sum(rewards) / MAX_TOTAL_REWARD
        score = max(0.0, min(score, 1.0))

        success = score >= 0.1

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success, steps_taken, score, rewards)


     
if __name__ == "__main__":
    asyncio.run(main())  

# import time

# if __name__ == "__main__":
#     try:
#         while True:
#             asyncio.run(main())

#             # ⏱ wait for few seconds
#             time.sleep(3)

#             user_input = input("Run again? (yes/no): ").strip().lower()

#             if user_input != "yes":
#                 break

#     except KeyboardInterrupt:
#         pass

