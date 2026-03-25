# Resume Tailor

A CLI tool that uses Claude to tailor your resume and write a cover letter for a specific job description — without fabricating experience you don't have.

The tool reads your master resume, analyzes the job description, asks you a small number of targeted clarifying questions, then rewrites your resume with language aligned to the role. Output is a formatted PDF.

---

## How it works

**Step 1 — Analyze**
Claude maps key phrases from the job description to the closest matching bullets on your resume. It identifies gaps between how you've described your work and how the company describes the role.

**Step 2 — Clarify**
Before rewriting anything, Claude asks 3–6 targeted questions grounded in those gaps. This keeps the output accurate to your actual experience.

**Step 3 — Generate**
After you answer the questions, Claude rewrites the resume and renders a PDF. If a cover letter file is provided, it generates that too.

The goal is **language alignment and framing** — not gap-filling. The tool is guardrailed to never add skills, tools, or experience that aren't already in your resume.

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your Anthropic API key**

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

**3. Add your resume**

Place your master resume (PDF, DOCX, or TXT) in the project folder. This is the single source of truth for every run — the tool only reframes what's already there.

**4. Create your candidate context file**

`candidate_context.txt` is a file of standing facts about your background that gets injected into every run. It prevents the tool from asking you the same questions repeatedly and keeps the guardrails accurate.

See `candidate_context_template.txt` for a prompt you can give to an LLM to generate this file from your own background.

---

## Usage

**Resume only:**
```bash
python tailor.py --resume your_resume.pdf --jd job_description.txt
```

**Resume + cover letter:**
```bash
python tailor.py --resume your_resume.pdf --jd job_description.txt --cover-letter cover_letter_base.pdf
```

Paste the full job description into `job_description.txt` before running.

**Skip the interactive questions (use a pre-written answers file):**
```bash
python tailor.py --resume your_resume.pdf --jd job_description.txt --answers-file my_answers.txt
```

---

## Output

The tool generates a PDF named after the company extracted from the job description. Output lands in the project folder.

---

## Architecture

Single-file tool (`tailor.py`) with three stages:

| Stage | What it does |
|---|---|
| Extract | Reads PDF, DOCX, or TXT into plain strings |
| Analyze | Calls Claude to map JD phrases to resume bullets; generates clarifying questions |
| Tailor + Render | Calls Claude for a structured JSON resume object; ReportLab renders the PDF |

The renderer handles section-type detection (Experience, Education, Skills, Coursework) and produces a consistently formatted single-page PDF. It tries six spacing compression levels before asking Claude to trim content.

---

## Adapting for yourself

- Update `_LINKEDIN_URL` and `_GITHUB_URL` in `tailor.py` with your own profile links
- Generate your `candidate_context.txt` using the template prompt in `candidate_context_template.txt`
- Tune spacing and font sizes via `SpacingConfig` and the constants at the top of `tailor.py`
- The output filename pattern (`YourName_{company}.pdf`) is set near the bottom of `tailor.py`

---

## Dependencies

`anthropic`, `pdfplumber`, `python-docx`, `python-dotenv`, `reportlab`

See `requirements.txt` for pinned versions.
