# 📧 OpenEnv Email Response Environment

## 🚀 Overview
This project implements a real-world OpenEnv environment where an AI agent processes incoming emails and decides the appropriate action. The system simulates practical email management tasks such as prioritization, scheduling, and filtering spam.

---

## 🎯 Objective
The goal is to train/evaluate an agent that can:
- Understand email content
- Assess urgency and intent
- Select the correct action from a predefined set

---

## ⚙️ Action Space
The agent must choose **exactly one** action:

- `reply` → Respond to normal emails  
- `schedule` → Meetings, events, appointments  
- `mark_important` → Urgent or critical emails  
- `ignore` → Spam, promotions, irrelevant emails  

---

## 👁️ Observation Space
Each step provides:
- `email content` → Text of the email  
- `context/history` → Previous interactions (if any)  

---

## 🧠 Environment Details
- Built using OpenEnv API (`reset()`, `step()`, `state()`)
- Simulates realistic email workflows
- Supports containerized execution via Docker
- Compatible with Hugging Face Spaces deployment

---

## 🧪 Tasks & Difficulty Levels

### 🟢 Easy
- Spam detection  
- Simple replies  

### 🟡 Medium
- Meeting scheduling  
- General classification  

### 🔴 Hard
- Urgency detection  
- Context-aware decision making  

---

## 🎯 Reward Function
- Correct action → Positive reward  
- Incorrect action → Low/zero reward  
- Partial correctness → Intermediate reward  

Rewards are normalized between **0 and 1**.

---

## 📊 Scoring
Final score is computed as:
   Total Reward / Maximum Possible Reward


---

## 🧪 Inference
The agent uses an LLM (via OpenAI-compatible API) to:
1. Analyze email content  
2. Predict the correct action  
3. Interact with the environment  

---

## 🐳 Docker Setup

### Build Image
```bash
docker build -t openenv-project .


## 🟢🐳 Run Container

```bash
docker run --env-file .env openenv-project

🟡 Environment Variables Setup 
# Create .env file (for adding local varibales )
touch .env
and then later add to sercet variables.

🔵Install Dependencies (local)
pip install -r requirements.txt

🟣 Run Inference locally
python inference.py