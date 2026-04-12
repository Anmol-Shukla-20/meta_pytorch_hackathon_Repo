import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Helper function copied from inference.py for standalone server compliance
def get_model_message(client, step, last_echoed, last_reward, history):
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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

    # Fallback
    text = last_echoed.lower()
    if any(w in text for w in ["meeting","schedule","tomorrow","time","appointment","chat","call"]):
        return "schedule"
    if any(w in text for w in ["urgent","asap","important","deadline","down","fix","report"]):
        return "mark_important"
    if any(w in text for w in ["free","win","offer","click","buy","newsletter","update","spam"]):
        return "ignore"
    return "reply"

app = FastAPI()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")

try:
    from env import EmailEnv
    active_env = EmailEnv(task="easy")

    class StepRequest(BaseModel):
        action: str
        
    class ChatRequest(BaseModel):
        email_text: str

    @app.post("/reset")
    def api_reset():
        obs = active_env.reset()
        return {"observation": obs, "done": False}

    @app.post("/step")
    def api_step(req: StepRequest):
        obs, rew, done, info = active_env.step(req.action)
        return {"observation": obs, "reward": rew, "done": done, "info": info}
        
    @app.post("/api/analyze")
    def api_analyze(req: ChatRequest):
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        action = get_model_message(client, 1, req.email_text, 0.0, [])
        return {"action": action}
        
    @app.get("/privacy", response_class=HTMLResponse)
    def api_privacy():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Privacy Policy - AI Email Inbox Agent</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
            <style>
                body { 
                    margin: 0; padding: 40px; font-family: 'Outfit', sans-serif; 
                    background: #0B0E14; color: #ffffff; line-height: 1.6;
                    display: flex; flex-direction: column; align-items: center; min-height: 100vh;
                }
                .orb {
                    position: fixed; border-radius: 50%; filter: blur(80px); opacity: 0.3; z-index: -1;
                }
                .orb-1 { width: 400px; height: 400px; background: #4f46e5; top: -100px; left: -100px; }
                .orb-2 { width: 500px; height: 500px; background: #ec4899; bottom: -150px; right: -150px; }
                
                .glass-card {
                    background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px;
                    padding: 40px; width: 90%; max-width: 800px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                }
                h1 { font-size: 2.2rem; color: #818cf8; margin-bottom: 5px; }
                h2 { font-size: 1.4rem; color: #ec4899; margin-top: 30px; }
                p, li { color: #9ca3af; font-size: 1.05rem; }
                code { background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; color: #fff; }
                .btn {
                    display: inline-block; margin-top: 40px; text-decoration: none;
                    background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
                    color: white; padding: 12px 30px; border-radius: 12px; font-weight: 600;
                    transition: 0.2s;
                }
                .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 15px -5px #ec4899; }
            </style>
        </head>
        <body>
            <div class="orb orb-1"></div>
            <div class="orb orb-2"></div>
            <div class="glass-card">
                <h1>Privacy Policy</h1>
                <p>Last updated: April 12, 2026</p>
                <p><b>AI Email Inbox Agent</b> ("the App") values your privacy. This policy explains how we handle your data when using our AI-powered triage tool.</p>
                
                <h2>1. Data Collection</h2>
                <p>We use the Google OAuth scope <code>https://www.googleapis.com/auth/gmail.readonly</code> to read your unread emails. This access is limited to the purpose of automated triage and categorization.</p>
                
                <h2>2. Real-Time Processing</h2>
                <p>All email content is processed in real-time. We <b>do not store</b> your email content, subject lines, or access tokens on our servers. All data is handled within your active session.</p>
                
                <h2>3. Third-Party Services</h2>
                <p>Email snippets are sent to our AI providers (Hugging Face / OpenAI) solely for the purpose of classification. These snippets are not used for training models or for any secondary purposes.</p>
                
                <h2>4. Your Rights</h2>
                <p>You can revoke access to your data at any time by disconnecting your Google account or revoking permissions in your Google Security settings.</p>
                
                <a href="/" class="btn">Return to Dashboard</a>
            </div>
        </body>
        </html>
        """

    @app.get("/terms", response_class=HTMLResponse)
    def api_terms():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Terms of Service - AI Email Inbox Agent</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
            <style>
                body { 
                    margin: 0; padding: 40px; font-family: 'Outfit', sans-serif; 
                    background: #0B0E14; color: #ffffff; line-height: 1.6;
                    display: flex; flex-direction: column; align-items: center; min-height: 100vh;
                }
                .orb {
                    position: fixed; border-radius: 50%; filter: blur(80px); opacity: 0.3; z-index: -1;
                }
                .orb-1 { width: 400px; height: 400px; background: #4f46e5; top: -100px; left: -100px; }
                .orb-2 { width: 500px; height: 500px; background: #ec4899; bottom: -150px; right: -150px; }
                
                .glass-card {
                    background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px;
                    padding: 40px; width: 90%; max-width: 800px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                }
                h1 { font-size: 2.2rem; color: #818cf8; margin-bottom: 5px; }
                h2 { font-size: 1.4rem; color: #ec4899; margin-top: 30px; }
                p { color: #9ca3af; font-size: 1.05rem; }
                .btn {
                    display: inline-block; margin-top: 40px; text-decoration: none;
                    background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
                    color: white; padding: 12px 30px; border-radius: 12px; font-weight: 600;
                    transition: 0.2s;
                }
                .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 15px -5px #ec4899; }
            </style>
        </head>
        <body>
            <div class="orb orb-1"></div>
            <div class="orb orb-2"></div>
            <div class="glass-card">
                <h1>Terms of Service</h1>
                <p>Last updated: April 12, 2026</p>
                <p>By using <b>AI Email Inbox Agent</b>, you agree to the following terms and conditions.</p>
                
                <h2>1. Use of Service</h2>
                <p>The service is provided to categorize and triage your Gmail inbox using AI. You grant the app read-only access to your Gmail data for this specific purpose.</p>
                
                <h2>2. Data Handling</h2>
                <p>You acknowledge that the app processes your data in real-time. We do not guarantee 100% accuracy in AI categorization and we are not liable for any misclassification of emails.</p>
                
                <h2>3. Responsibility</h2>
                <p>The app is provided "as is" without warranties of any kind. You are responsible for maintaining the security of your own Google account.</p>
                
                <a href="/" class="btn">Return to Dashboard</a>
            </div>
        </body>
        </html>
        """

    @app.get("/googlebcf113f3cc578b16.html", response_class=HTMLResponse)
    def google_file_verification():
        return "google-site-verification: googlebcf113f3cc578b16.html"

    @app.get("/", response_class=HTMLResponse)
    def api_root():
        client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        # Full HTML content from inference.py
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Email Inbox Agent</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
            <script src="https://accounts.google.com/gsi/client" async defer></script>
            <style>
                body, html {
                    margin: 0; padding: 0; width: 100%; min-height: 100vh;
                    font-family: 'Outfit', sans-serif; background: #0B0E14; color: #ffffff;
                    overflow-x: hidden; display: flex; align-items: flex-start; justify-content: center;
                }
                .orb {
                    position: fixed; border-radius: 50%; filter: blur(80px); opacity: 0.5;
                    animation: float 10s infinite ease-in-out alternate; z-index: 1; pointer-events: none;
                }
                .orb-1 { width: 400px; height: 400px; background: #4f46e5; top: -100px; left: -100px; }
                .orb-2 { width: 500px; height: 500px; background: #ec4899; bottom: -150px; right: -150px; animation-delay: -5s; }
                @keyframes float { 0% { transform: translate(0, 0); } 100% { transform: translate(50px, 50px); } }
                
                .glass-card {
                    position: relative; z-index: 10; background: rgba(255, 255, 255, 0.03);
                    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px;
                    padding: 40px; text-align: center; width: 95%; max-width: 1100px;
                    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); margin: 40px 0;
                }
                
                .status-badge {
                    display: inline-flex; align-items: center; gap: 10px;
                    background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3);
                    color: #10b981; padding: 8px 16px; border-radius: 50px;
                    font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 20px;
                }
                .pulse {
                    width: 8px; height: 8px; background: #10b981; border-radius: 50%;
                    box-shadow: 0 0 10px #10b981, 0 0 20px #10b981;
                    animation: pulse-ring 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                }
                @keyframes pulse-ring {
                    0% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                    70% { transform: scale(1); box-shadow: 0 0 0 15px rgba(16, 185, 129, 0); }
                    100% { transform: scale(0.8); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
                }
                
                h1 { font-size: 2.2rem; margin: 0; font-weight: 600; }
                .subtitle { color: #9ca3af; font-size: 1rem; margin: 10px 0 30px 0; font-weight: 300; }

                .analyzer-section {
                    background: rgba(0,0,0,0.2); border-radius: 18px; padding: 25px;
                    margin-bottom: 30px; border: 1px solid rgba(255,255,255,0.05);
                    text-align: left;
                }
                textarea {
                    width: 100%; box-sizing: border-box; height: 100px; background: rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
                    padding: 15px; color: #fff; font-family: 'Outfit', sans-serif;
                    font-size: 1rem; resize: none; margin-bottom: 15px; outline: none;
                }
                textarea:focus { border-color: #4f46e5; }
                
                button {
                    background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
                    color: white; border: none; padding: 12px 25px; border-radius: 12px;
                    font-size: 1rem; font-weight: 600; cursor: pointer; font-family: 'Outfit', sans-serif;
                    transition: 0.2s;
                }
                button:hover { transform: translateY(-2px); box-shadow: 0 8px 15px -5px #ec4899; }
                .btn-secondary { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #9ca3af; }
                
                .iframe-notice {
                    background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.2);
                    color: #f59e0b; padding: 15px; border-radius: 12px; font-size: 0.9rem;
                    margin-bottom: 25px; display: none; align-items: center; justify-content: space-between;
                }

                .kanban { display: flex; gap: 20px; text-align: left; flex-wrap: wrap; margin-top: 20px; }
                .column {
                    flex: 1; min-width: 240px; background: rgba(0, 0, 0, 0.2);
                    border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px;
                    padding: 20px; min-height: 250px;
                }
                .column h3 {
                    margin: 0 0 15px 0; font-size: 1rem; border-bottom: 1px solid rgba(255,255,255,0.05);
                    padding-bottom: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
                }
                #col-important h3 { color: #ef4444; } #col-schedule h3 { color: #f59e0b; }
                #col-reply h3 { color: #10b981; } #col-ignore h3 { color: #9ca3af; }
                
                .email-card {
                    background: rgba(255, 255, 255, 0.08); padding: 15px; border-radius: 10px;
                    margin-bottom: 12px; font-size: 0.85rem; border-left: 4px solid transparent;
                    animation: fadeUp 0.4s ease forwards;
                }
                #col-important .email-card { border-left-color: #ef4444; }
                #col-schedule .email-card { border-left-color: #f59e0b; }
                #col-reply .email-card { border-left-color: #10b981; }
                #col-ignore .email-card { border-left-color: #9ca3af; }
                .email-card strong { display: block; margin-bottom: 5px; color: #fff; }
                
                #globalLoading { position: fixed; bottom: 30px; right: 30px; background: #4f46e5; color: #fff; padding: 10px 20px; border-radius: 50px; display: none; font-weight: 600; z-index: 100; box-shadow: 0 10px 20px rgba(0,0,0,0.3); }
                @keyframes fadeUp { 0% { opacity: 0; transform: translateY(10px); } 100% { opacity: 1; transform: translateY(0); } }
            </style>
        </head>
        <body>
            <div class="orb orb-1"></div>
            <div class="orb orb-2"></div>
            
            <div id="globalLoading">System Processing...</div>

            <div class="glass-card">
                <div class="status-badge">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div class="pulse"></div> Environment Online
                    </div>
                </div>
                <h1>AI Email Inbox Agent</h1>
                <p class="subtitle">Analyze emails manually or connect your Gmail for real-time updates.</p>
                
                <div class="analyzer-section">
                    <textarea id="manualInput" placeholder="Paste sample email text here to test the AI..."></textarea>
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
                        <button onclick="analyzeManual()">Analyze Manual Email</button>
                        <div id="authContainer">
                            <button id="authBtn" onclick="handleAuthClick()">Connect Live Gmail</button>
                            <button id="logoutBtn" onclick="handleLogoutClick()" class="btn-secondary" style="display: none;">Logout</button>
                        </div>
                    </div>
                </div>

                <div class="kanban">
                    <div class="column" id="col-important"><h3>🚨 Important</h3></div>
                    <div class="column" id="col-schedule"><h3>🗓️ Schedule</h3></div>
                    <div class="column" id="col-reply"><h3>💬 Reply / To-Do</h3></div>
                    <div class="column" id="col-ignore"><h3>🗑️ Ignore</h3></div>
                </div>

                <footer style="margin-top: 60px; padding: 40px; border-top: 2px solid #4f46e5; text-align: center; background: rgba(255,255,255,0.05); border-radius: 0 0 24px 24px;">
                    <p style="font-size: 1rem; margin-bottom: 15px; color: #ffffff; font-weight: bold;">Legal & Privacy Information</p>
                    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
                        <a href="/privacy" style="color: #818cf8; text-decoration: underline; font-size: 0.9rem; font-weight: 600;">Privacy Policy</a>
                        <a href="/terms" style="color: #818cf8; text-decoration: underline; font-size: 0.9rem; font-weight: 600;">Terms of Service</a>
                    </div>
                    <p style="font-size: 0.8rem; color: #9ca3af;">&copy; 2026 AI Email Inbox Agent. Official Hackathon Submission.</p>
                </footer>
            </div>

            <script>
                let tokenClient;
                let accessToken = null;
                const clientId = '{{GOOGLE_CLIENT_ID}}';

                window.onload = function () {
                    if(clientId) {
                        tokenClient = google.accounts.oauth2.initTokenClient({
                            client_id: clientId,
                            scope: 'https://www.googleapis.com/auth/gmail.readonly',
                            callback: (resp) => {
                                accessToken = resp.access_token;
                                fetchGmail();
                            },
                        });
                    }
                };

                function handleAuthClick() { tokenClient.requestAccessToken(); }
                function handleLogoutClick() { accessToken = null; document.getElementById('authBtn').style.display='block'; }

                async function analyzeManual() {
                    const text = document.getElementById('manualInput').value;
                    if(!text.trim()) return;
                    showLoading("Analyzing...");
                    const res = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ email_text: text })
                    });
                    const data = await res.json();
                    addEmailToColumn("Manual Test", text, data.action);
                    hideLoading();
                }

                async function fetchGmail() {
                    // Similar logic to inference.py
                }

                function addEmailToColumn(subject, snippet, action) {
                    const colId = (action === 'mark_important') ? 'col-important' : (action === 'schedule') ? 'col-schedule' : (action === 'ignore') ? 'col-ignore' : 'col-reply';
                    const card = document.createElement('div');
                    card.className = 'email-card';
                    card.innerHTML = `<strong>${subject}</strong><p>${snippet}</p>`;
                    document.getElementById(colId).appendChild(card);
                }

                function showLoading(msg) { const l = document.getElementById('globalLoading'); l.innerText = msg; l.style.display = 'block'; }
                function hideLoading() { document.getElementById('globalLoading').style.display = 'none'; }
            </script>
        </body>
        </html>
        """
        return html_content.replace("{{GOOGLE_CLIENT_ID}}", client_id)

except Exception as e:
    print(f"Error initializing server: {e}")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
