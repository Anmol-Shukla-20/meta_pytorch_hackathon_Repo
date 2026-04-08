from tasks.easy import run as easy_task
from tasks.medium import run as medium_task
from tasks.hard import run as hard_task

def evaluate(env):
    total_reward = 0
    state = env.reset()

    while True:
        action = "ignore"  # baseline dummy
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


def grade_all():
    scores = []

    for task_fn in [easy_task, medium_task, hard_task]:
        env = task_fn()
        score = evaluate(env)

        # normalize to 0-10
        score = max(0, min(10, score))
        scores.append(score)

    return scores


if __name__ == "__main__":
    print("Scores:", grade_all())