# gemini_utils.py
import os, re, json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- keyword helpers ----------
WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

def extract_keywords(text: str) -> set[str]:
    """Return a set of lowercase keywords 3+ chars long."""
    return set(WORD_RE.findall(text.lower()))

def overlap_score(resume: str, jd: str) -> int:
    """Simple ATS: % of JD‑keywords that appear in resume."""
    res_kw = extract_keywords(resume)
    jd_kw  = extract_keywords(jd)
    if not jd_kw:
        return 0
    return min(int(len(res_kw & jd_kw) / len(jd_kw) * 100), 100)

# ---------- main analysis ----------
def analyze_resume_jobdesc(resume_text: str, jd_text: str) -> dict:
    local_score = overlap_score(resume_text, jd_text)

    # ask Gemini for skills / suggestions
    prompt = f"""
Analyse the resume vs. job‑description below.

RESUME
------
{resume_text}

JOB DESCRIPTION
---------------
{jd_text}

Output *only* JSON in exactly this format (no other text):
{{
  "resume_skills": list,      // 5–10 top skills from resume
  "jd_skills": list,          // 5–10 top skills required by JD
  "missing_skills": list,     // skills in JD but NOT in resume
  "improvements": list,       // 3 concrete improvement ideas
  "summary": str              // one‑sentence summary
}}
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        raw = model.generate_content(prompt).text
        json_blob = re.search(r"\{.*\}", raw, re.DOTALL)
        payload  = json.loads(json_blob.group(0)) if json_blob else {}
    except Exception as e:
        # fallback if Gemini fails
        payload = {
            "resume_skills": [],
            "jd_skills": [],
            "missing_skills": list(extract_keywords(jd_text) - extract_keywords(resume_text))[:5],
            "improvements": ["Add missing JD keywords to resume."],
            "summary": f"Gemini error: {e}"
        }

    # ALWAYS use locally‑calculated score
    payload["score"] = local_score
    return payload

# ---------- chat ----------
def gemini_chat(resume_text: str, jd_text: str, user_msg: str) -> str:
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")

        prompt = f"""
You are a resume‑analysis assistant.

Context:
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

User’s question:
{user_msg}

INSTRUCTIONS:
- Answer using bullet points ONLY.
- Each bullet **must start on its own new line** with a leading hyphen and a space (`- `).
- Do NOT put multiple bullets on the same line.
- Do NOT include any other text (no intro sentence).

Example output format:

- First point here.
- Second point here.
- Third point here.

Now, answer:
"""
        return model.generate_content(prompt).text.strip()

    except Exception as e:
        return f"Chat error: {e}"

