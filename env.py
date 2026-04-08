import random
from models import Observation

class EmailEnv:
    def __init__(self, task="easy"):
        self.task = task
        self.max_steps = 5
        self.current_step = 0
        self.data = self.load_data()

    def load_data(self):
        if self.task == "easy":
            return [
                {"text": "Win a free iPhone now!", "sender": "spam@mail.com", "urgency": 0.1, "label": "ignore"},
                {"text": "Meeting at 3 PM", "sender": "boss@company.com", "urgency": 0.9, "label": "schedule"},
            ]
        elif self.task == "medium":
            return [
                {"text": "Project deadline tomorrow", "sender": "manager@company.com", "urgency": 0.95, "label": "reply"},
                {"text": "Lunch plans?", "sender": "friend@mail.com", "urgency": 0.3, "label": "ignore"},
            ]
        else:  # hard
            return [
                {"text": "Client escalation: urgent fix needed", "sender": "client@company.com", "urgency": 0.98, "label": "reply"},
                {"text": "Weekly newsletter", "sender": "news@mail.com", "urgency": 0.2, "label": "ignore"},
            ]

    def reset(self):
        self.current_step = 0
        self.sample = random.choice(self.data)
        return self.state()

    def state(self):
        return {
            "email_text": self.sample["text"],
            "sender": self.sample["sender"],
            "urgency": self.sample["urgency"],
            "step": self.current_step
        }

    def compute_reward(self, action):
        correct = self.sample["label"]

        reward = 0

        if action == correct:
            reward += 8
        else:
            reward -= 3

        # partial reward logic
        if self.sample["urgency"] > 0.8 and action in ["reply", "schedule"]:
            reward += 2

        return reward

    def step(self, action):
        reward = self.compute_reward(action)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self.state(), reward, done, {}