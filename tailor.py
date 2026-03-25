#!/usr/bin/env python3
"""Tailor a resume (and optional cover letter) to a job description using Claude."""

import argparse
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

import anthropic
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth

MODEL = "claude-sonnet-4-20250514"

CANDIDATE_CONTEXT_PATH = "candidate_context.txt"
QUESTIONS_CACHE_PATH   = ".questions_cache.json"


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    elif suffix == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif suffix == ".txt":
        return path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, DOCX, or TXT.")


# --------------------------------------------------------------------------- #
# Claude calls
# --------------------------------------------------------------------------- #

RESUME_SCHEMA = """
{
  "name": "Full Name",
  "company": "Hiring company name from the job description",
  "role": "Target role title from the job description",
  "contact": {
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "email": "email@example.com",
    "phone": "555-555-5555"
  },
  "summary": "Two to three sentence professional summary targeted to this role.",
  "sections": [
    {
      "title": "Skills & Tools",
      "items": [
        {
          "title": "",
          "subtitle": "Python · SQL · R · Tableau · advanced machine learning · statistical analysis · etc.",
          "date": "",
          "bullets": []
        }
      ]
    },
    {
      "title": "Experience",
      "items": [
        {
          "title": "Job Title",
          "subtitle": "Company Name",
          "date": "Date string exactly as in the original resume",
          "bullets": [
            "Bullet text (2-3 bullets per role, most relevant first)",
            "Another bullet"
          ]
        }
      ]
    },
    {
      "title": "Education",
      "items": [
        {
          "title": "Degree Name",
          "subtitle": "Institution Name",
          "date": "Date string exactly as in the original resume",
          "bullets": []
        }
      ]
    },
    {
      "title": "Relevant Coursework",
      "items": [
        {
          "title": "",
          "subtitle": "",
          "date": "",
          "bullets": [
            "Course Name: Brief description rephrased to mirror JD keywords"
          ]
        }
      ]
    }
  ]
}
"""

RESUME_SYSTEM = """\
You are an expert resume writer and ATS optimization specialist.
Tailor the candidate's resume to the job description by:
  • Writing a 2-3 sentence summary that directly targets this specific role.
    Tell the complete story of who the candidate is: a technically sharp, analytically
    grounded person whose work directly powers business outcomes. Lead with what they
    do and what they've built — not a job title intro. Name specific methods, domains,
    or systems from the resume. The reader should finish the summary thinking "this
    person is technically capable and understands how their work ties to the business."
    Never use generic filler phrases: "transform complex data challenges", "actionable
    insights", "data-driven professional", "passionate about", "proven track record",
    or similar. Do not quantify things like customer counts or revenue figures —
    describe the nature and impact of the work instead.
    Keep tight — must render in exactly 3 lines or fewer at 11pt Arial on a 7-inch
    text area. Err short rather than long.
  • Rephrasing bullets to surface the most relevant experience first within each role,
    but NEVER reordering roles — keep experience items in reverse chronological order
    (most recent job first) exactly as in the original resume.
  • Mirroring keywords and phrasing from the JD wherever they truthfully apply.
  • Keeping exactly 3 bullets per role EXCEPT for SquareTrade, which must have
    exactly 2 bullets. Never reduce bullets simply to save space — drop spacing
    instead.
  • Ending every bullet with a period.
  • NEVER inventing, fabricating, or implying any experience, skill, title, date,
    or accomplishment that does not already exist in the original resume.
  • Every action verb and claim in every bullet must trace directly to an existing
    bullet in the original resume. Reframe the language; never add new activities.
  • Do not name specific model implementations (e.g. XGBoost, LightGBM, CatBoost)
    in bullets unless the JD explicitly calls them out — use "advanced machine
    learning models", "gradient boosting models", or similar generic terms.
  • When a clarifying answer indicates limited or no direct work experience
    (e.g. "Coursework only", "No X experience", "Limited X"), that item MUST NOT
    appear in any work experience bullet — coursework knowledge stays in coursework.
  • Preserving every company name, school name, and institution EXACTLY as written.
  • Preserving all dates EXACTLY as written — do not paraphrase or reformat them.
  • Keeping total content tight enough to fit on exactly one page.
  • Extracting the hiring company name into the `company` field and the target
    role title into the `role` field from the job description.

JOB TITLES: The candidate's original titles (e.g. "Data Science Intern") are flexible.
  You may adapt a title to better match the role — e.g. "Data Analytics Intern",
  "Business Analytics Intern", "Operations Analytics Intern", or similar —
  if it more accurately reflects the work done and the target role. Never change
  company names, school names, or dates.

REQUIRED section order: Skills & Tools, Experience, Education, Relevant Coursework.
  • Skills & Tools: output a single flat keyword list — languages, tools, platforms,
    and methods all together. Include skills and tools that appear in the original
    resume, plus close technical synonyms clearly implied by the candidate's
    demonstrated work (e.g. standard libraries for a listed language or tool). Never
    add industry-specific domain terms (e.g. insurance, underwriting, actuarial, risk
    modeling, pricing models) unless they appear explicitly in the original resume.
    NEVER include "Linux" or "ETL processes" under any circumstances.
    Output as exactly 1 item: title = "" (empty string), subtitle = all skills
    separated by " · " (space, middle dot, space). No categories, no labels.
    The intent is a keyword-dense signal list — things the candidate genuinely has
    that the role is looking for.
    Do NOT name specific model implementations (e.g. XGBoost, LightGBM, CatBoost)
    unless the JD explicitly calls them out — use "advanced machine learning models",
    "gradient boosting", or similar generic terms instead.
  • Relevant Coursework: include ALL 5 courses from the original. For each course,
    select the topics/skills most relevant to this JD from the course content and
    list them as comma-separated keywords — not sentences. Format each bullet
    EXACTLY as "Course Name: topic1, topic2, topic3". NOTE: this renders as a
    subsection label under Education — not a section header.
  • Contact: always output fields in this order: linkedin, github, email, phone.
    Use the display values "LinkedIn" and "GitHub" (not URLs) for those two fields.

PUNCTUATION (hard stop): Never use em dashes (—) anywhere. Use a comma, semicolon, or period instead.

Return ONLY a single valid JSON object — no markdown fences, no commentary.\
"""

ANALYSIS_SYSTEM = """\
You are an expert resume optimizer. You will receive two inputs:
1. BASE RESUME — the candidate's current version
2. JOB DESCRIPTION — the target role

CRITICAL FILTERING RULE: If CANDIDATE BACKGROUND CONTEXT is provided, treat every
fact in it as a confirmed standing answer. Before writing any clarifying question,
check whether it is already answered there. If it is, skip that question entirely.
You are not required to ask any minimum number of questions — ask only about genuine
gaps that are BOTH (a) specific to this JD and (b) not covered by the context file.

Your job has two parts.

PART 1 — RESUME ANALYSIS (internal reasoning, do not print this section)
Review the base resume carefully. Note which experiences, skills, and accomplishments
most directly map to the target role's requirements, and where gaps or language
mismatches exist.

PART 2 — OUTPUT (print exactly this, in this order)

## JD Key Phrases
List the most important phrases the company uses to describe success, required skills,
and ideal candidate qualities. For each phrase, note the closest matching bullet from
the base resume (or "no direct match" if none exists).

## Clarifying Questions
Ask 0-4 targeted questions to help reframe the candidate's existing bullets using the
company's exact language. Only ask what you genuinely do not know after reading the
context file. Base each question on a specific gap between the resume's current phrasing
and the JD's phrasing. Do not invent experience — only ask what you need to better
describe what the candidate actually did. If the context file already covers a gap,
do not ask about it.\
"""

COVER_LETTER_SCHEMA = """
{
  "name": "Candidate full name from resume",
  "company": "Hiring company name from the job description",
  "contact": {
    "linkedin": "LinkedIn",
    "github": "GitHub",
    "email": "email@example.com",
    "phone": "555-555-5555"
  },
  "salutation": "Dear [Hiring Manager Name or 'Hiring Manager'],",
  "opening": "Opening paragraph text (75-100 words)",
  "bullet_intro": "Introductory sentence leading into the bullets, e.g. 'In my recent internship roles, I'",
  "bullets": [
    "First bullet: one specific accomplishment from one company (30-50 words)",
    "Second bullet: one specific accomplishment from a second company (30-50 words)",
    "Third bullet: one specific accomplishment (30-50 words)"
  ],
  "body_paragraphs": [
    "Optional prose paragraph (60-80 words) — baseball team, company motivation, or a differentiator not in bullets. Omit if not needed."
  ],
  "closing": "Closing paragraph text (50-75 words)",
  "sign_off": "Sincerely,"
}
"""

COVER_LETTER_SYSTEM = """\
You are an expert cover letter writer for data science and analytics candidates.
Write a tailored cover letter based on the candidate's master cover letter (use it for
authentic voice and confirmed project content) and the job description. Do NOT rephrase
the master letter line by line — write a fresh letter structured for this specific role.

STRUCTURE (follow exactly, in this order):

1. Salutation
   Use the hiring manager's name if visible in the JD. Otherwise: "Dear Hiring Manager,"
   Never use "To Whom It May Concern."

2. Opening (75-100 words)
   • PREFERRED PATTERN: Open with a concrete capability statement about what the candidate
     brings — not about the role, not about the company. Example structure: "[Specific skill
     or result], which is exactly what [role] demands." Lead with what you do, not what you want.
   • The opener earns its place only if it establishes the candidate's most differentiated
     angle within the first two sentences.
   • Cut any sentence that could be copy-pasted into a cover letter for a different company
     without changing a word. If it's generic, delete it.
   • Never open with "I am excited to apply for...", "My name is...", or "I am writing to apply..."

3. Bullet section
   • One intro sentence ending with "I" (e.g. "In my recent internship roles, I"), stored in "bullet_intro".
   • Exactly 3 bullets in the "bullets" array. Each bullet covers one specific accomplishment
     from a named company, 30-50 words. Lead with business impact, not tools.
   • Every bullet must contain: a tool or method, a scale or scope indicator, and a concrete
     outcome or projected impact. If a bullet is missing its outcome, fold it into another.
   • Order bullets by relevance to the JD, not chronology. Weakest bullet goes last.
   • Draw only from the master cover letter and resume — do not invent claims.
   • Mirror the JD's language in bullets wherever it truthfully applies. Do not mirror JD
     language in the closing paragraph.

4. Body paragraph (optional, 60-80 words)
   • Include only if there is genuine additional context: company-specific motivation, a
     career thread connecting the background to this role, or a differentiator not in the bullets.
   • The Cal Poly Baseball Analytics Team experience belongs here if relevant to the JD.
   • Omit entirely if the bullets cover everything essential — leave "body_paragraphs" as [].
   • Prose only — no bullet points.

5. Closing (two sentences maximum — hard limit)
   • Sentence 1: One specific value-add statement — what you bring to this role. Not a
     restatement of the company's mission. Not a generic expression of enthusiasm.
   • Sentence 2: One clean call to action.
     BANNED openers for the call to action: "I would welcome..." and any variant.
     APPROVED alternatives: "I look forward to discussing..." or "Happy to connect about
     how this background fits your team's needs." or similar direct phrasing.
   • No company flattery, no mission alignment, no apologies, no hedging.

6. Sign-off: "Sincerely,"

TARGET LENGTH: 250-350 words across all paragraphs. Never exceed one page.

TONE: Confident, direct, professional. Active voice throughout. No jargon dumps.
Write for a non-technical hiring manager — lead with impact and outcomes, not methods.

CONTACT: Extract the candidate's name, email, phone, LinkedIn display value, and GitHub
display value from the resume. Use display values "LinkedIn" and "GitHub" for those fields.
Extract the hiring company name into the `company` field.

PUNCTUATION (hard stop):
  • Never use em dashes (—) anywhere in the letter. Use a comma, semicolon, or period instead.

ANTI-FABRICATION (hard stops):
  • Every claim must trace directly to the master cover letter or original resume.
  • Never add skills, tools, titles, or experiences not in the source materials.
  • Never invent numbers, metrics, or team sizes.
  • Preserve all company names, school names, and dates exactly as written.

BLACKLISTED PHRASES (hard stops — do not use any of these anywhere in the letter):
  • "passion for"
  • "eager to bring"
  • "transforming business challenges"
  • "data-driven solutions" (in closing context)
  • "would welcome the opportunity"
  • "resonates" / "deeply resonates"
  • "well-positioned to"
  • "[Company]'s mission of" / "[Company]'s tradition of"
  • "aligns with your mission"
  • "contributing to your"
  • "protecting families" or any restatement of the company's stated mission
  • "this role represents exactly"

PRE-FLIGHT CHECK (required before returning the draft):
  1. Scan every sentence for the blacklisted phrases above. If any appear, rewrite the
     sentence. Do not return a draft containing any of them.
  2. Check the closing explicitly for "I would welcome" or "would welcome the opportunity"
     — this phrase is banned. If present, rewrite using an approved alternative.
  3. Confirm the closing is exactly two sentences. Cut anything beyond that.
  4. Read the opener. If any sentence could appear unchanged in a cover letter for a
     different company, delete or rewrite it.
  5. Confirm the closing paragraph contains no JD language and no company flattery.
  6. Only return the JSON after this check passes.

Return ONLY a single valid JSON object matching the schema. No markdown fences, no commentary.\
"""

EXPAND_COURSEWORK_SYSTEM = """\
You are a resume optimizer. The tailored resume below fits on one page but has extra
whitespace at the bottom. Add 2-4 more relevant keywords to one or two of the Relevant
Coursework bullets to better match the job description. Keywords must be drawn from the
course descriptions in the original content — do not fabricate.

Format rules (unchanged):
  • Each bullet: "Course Name: keyword1, keyword2, keyword3, ..."
  • Only expand existing bullets — do not add new course entries.
  • Do not change any other section.

Return ONLY the updated JSON resume object. No markdown fences, no commentary.\
"""

TRIM_RESUME_SYSTEM = """\
You are a resume editor. The resume below overflows a single page and must be trimmed
to fit. Reduce content in this order of preference (least to most disruptive):

1. Shorten verbose bullets — remove filler phrases and redundant words without losing
   key facts, metrics, or JD-aligned keywords. Target 1-2 rendered lines per bullet
   at 11pt on a 7-inch text area.
2. Tighten the summary — aim for 2 concise sentences, cut filler.
3. Trim the skills list — remove items that are redundant or not directly relevant.
   Skills are a flat list — remove the least relevant items, do not restructure.

Hard rules (never break):
  • Never remove a complete bullet — only shorten existing ones.
  • SquareTrade gets exactly 2 bullets; every other role gets exactly 3.
  • Keep all 5 coursework bullets exactly as-is.
  • Every bullet must end with a period.
  • Skills items must use " · " (space, middle dot, space) as separators.
  • Never change company names, school names, dates, or job titles.
  • Never add skills, tools, or experience not present in the original.

Return ONLY the updated JSON resume object. No markdown fences, no commentary.\
"""


def _safe_slug(s: str, max_len: int = 40) -> str:
    import re
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "_", s.strip())
    return s[:max_len]


def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        lines = raw.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[start:end])
    return raw


def analyze_and_question(resume_text: str, jd_text: str, candidate_context: str = "") -> str:
    client = anthropic.Anthropic()
    user_msg = (
        f"BASE RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )
    if candidate_context:
        user_msg += (
            f"\n\nCANDIDATE BACKGROUND CONTEXT (standing answers — do not ask questions "
            f"already addressed here; only ask about genuine gaps specific to this JD):\n{candidate_context}"
        )
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        system=ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        response = stream.get_final_message()
    print()
    return response.content[0].text.strip()


def _extract_questions_json(analysis_text: str) -> list:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You extract structured data from text. Return only valid JSON, no markdown fences.",
        messages=[{"role": "user", "content": (
            "Extract the clarifying questions from this resume analysis output. "
            "Return a JSON array where each element has:\n"
            "- \"question\": the full question text (complete and self-contained)\n"
            "- \"options\": array of 2-4 short suggested answer options appropriate for this specific question\n\n"
            f"ANALYSIS TEXT:\n{analysis_text}"
        )}],
    )
    return json.loads(_strip_fences(response.content[0].text.strip()))


def tailor_resume(resume_text: str, jd_text: str, user_answers: str = "", candidate_context: str = "") -> dict:
    client = anthropic.Anthropic()
    user_msg = (
        f"Return a JSON object matching this schema:\n{RESUME_SCHEMA}\n\n"
        f"ORIGINAL RESUME:\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )
    if candidate_context:
        user_msg += f"\n\nCANDIDATE BACKGROUND CONTEXT (use to inform bullet reframing; treat as confirmed facts about the candidate):\n{candidate_context}"
    if user_answers:
        user_msg += f"\n\nCANDIDATE ANSWERS TO CLARIFYING QUESTIONS (use these to reframe bullets with the company's exact language):\n{user_answers}"
    with client.messages.stream(
        model=MODEL,
        max_tokens=8192,
        system=RESUME_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        response = stream.get_final_message()
    return json.loads(_strip_fences(response.content[0].text.strip()))


def tailor_cover_letter(cl_text: str, resume_text: str, jd_text: str,
                        candidate_context: str = "") -> dict:
    client = anthropic.Anthropic()
    user_msg = (
        f"Return a JSON object matching this schema:\n{COVER_LETTER_SCHEMA}\n\n"
        f"MASTER COVER LETTER (voice and confirmed project content — use as reference, "
        f"not as a template to rephrase):\n{cl_text}\n\n"
        f"CANDIDATE RESUME (source of truth for accomplishments):\n{resume_text}\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )
    if candidate_context:
        user_msg += (
            f"\n\nCANDIDATE BACKGROUND CONTEXT (confirmed facts — same guardrails apply "
            f"as for the resume):\n{candidate_context}"
        )
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        system=COVER_LETTER_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        response = stream.get_final_message()
    return json.loads(_strip_fences(response.content[0].text.strip()))


def expand_coursework(data: dict, jd_text: str) -> dict:
    """Ask Claude to add more JD-relevant keywords to coursework bullets."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=EXPAND_COURSEWORK_SYSTEM,
        messages=[{"role": "user", "content": (
            f"CURRENT RESUME JSON:\n{json.dumps(data)}\n\n"
            f"JOB DESCRIPTION:\n{jd_text}"
        )}],
    )
    return json.loads(_strip_fences(response.content[0].text.strip()))


def trim_resume_content(data: dict) -> dict:
    """Ask Claude to tighten bullets/summary/skills so the resume fits one page."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=TRIM_RESUME_SYSTEM,
        messages=[{"role": "user", "content": (
            f"RESUME JSON TO TRIM:\n{json.dumps(data)}"
        )}],
    )
    return json.loads(_strip_fences(response.content[0].text.strip()))


# --------------------------------------------------------------------------- #
# PDF rendering constants
# --------------------------------------------------------------------------- #

PAGE_W, PAGE_H = letter          # 612 × 792 pt
MARGIN_RESUME  = 0.75 * inch     # 54 pt
MARGIN_CL      = 1.0  * inch     # 72 pt
TEXT_W_RESUME  = PAGE_W - 2 * MARGIN_RESUME   # 504 pt = 7"
TEXT_W_CL      = PAGE_W - 2 * MARGIN_CL       # 468 pt

# Helvetica is built into every PDF viewer and is visually identical to Arial
FONT_REG  = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITAL = "Helvetica-Oblique"

SECTION_BLUE = (0x1f / 255, 0x38 / 255, 0x64 / 255)  # #1f3864
LINK_COLOR   = (0.05, 0.28, 0.63)

BULLET_CHAR = "\u2022"   # •
BULLET_LEFT = 22         # pt: text start from left edge of text area
BULLET_HANG = 14         # pt: bullet char is BULLET_LEFT - BULLET_HANG from text-area left

_LINKEDIN_URL = "https://www.linkedin.com/in/tim-christian-lnkdn/"
_GITHUB_URL   = "https://github.com/tchristian327/data-science-portfolio"

_EXPERIENCE_SECTIONS = {"experience", "work experience", "professional experience", "work history"}
_EDUCATION_SECTIONS  = {"education", "academic background", "academics"}
_SKILLS_SECTIONS     = {"skill", "skills", "skills & tools", "skills and tools", "tools"}
_COURSEWORK_SECTIONS = {"coursework", "relevant coursework", "courses"}


# --------------------------------------------------------------------------- #
# Spacing configuration
# --------------------------------------------------------------------------- #

@dataclass
class SpacingConfig:
    name_after:             float = 3.0
    contact_after:          float = 4.0
    summary_section_before: float = 6.0
    summary_section_after:  float = 3.0
    summary_text_before:    float = 2.0
    summary_text_after:     float = 2.0
    skills_before:          float = 6.0
    skills_after:           float = 4.0
    section_before:         float = 6.0
    section_after:          float = 3.0
    item_first_before:      float = 4.0   # gap before 1st item after section header
    item_between_before:    float = 8.0   # gap between experience blocks
    edu_before:             float = 4.0
    bullet_leading:         float = 13.2  # line height for 11pt body text (1.2×)
    bullet_after:           float = 1.0
    coursework_before:      float = 4.0
    coursework_after:       float = 1.0


# 5 progressively tighter spacing levels
SPACING_LEVELS = [
    # Level 0 — default (matches original DOCX layout)
    SpacingConfig(),
    # Level 1 — slightly tighter
    SpacingConfig(
        summary_section_before=5, section_before=5,
        item_between_before=6, bullet_after=0,
    ),
    # Level 2 — moderate
    SpacingConfig(
        summary_section_before=4, section_before=4,
        item_between_before=5, skills_before=4, skills_after=3,
        bullet_after=0, coursework_before=3,
    ),
    # Level 3 — tight
    SpacingConfig(
        summary_section_before=3, section_before=3,
        item_between_before=4, skills_before=3, skills_after=2,
        bullet_after=0, coursework_before=2,
        contact_after=3, summary_text_after=1, summary_section_after=2,
    ),
    # Level 4 — very tight
    SpacingConfig(
        summary_section_before=2, section_before=2,
        item_between_before=3, skills_before=2, skills_after=1,
        bullet_after=0, coursework_before=2, contact_after=2,
        summary_text_after=0, summary_section_after=2,
        item_first_before=2, edu_before=2, name_after=2,
    ),
    # Level 5 — maximum compression (tightens line height for all body text)
    SpacingConfig(
        summary_section_before=2, section_before=2,
        item_between_before=3, skills_before=2, skills_after=1,
        bullet_after=0, coursework_before=2, contact_after=2,
        summary_text_after=0, summary_section_after=2,
        item_first_before=2, edu_before=2, name_after=2,
        bullet_leading=12.0,   # 1.09× instead of 1.2× — saves ~1.2pt per line
    ),
]


# --------------------------------------------------------------------------- #
# PDF renderer
# --------------------------------------------------------------------------- #

class _PdfRenderer:
    """
    Canvas-based resume PDF renderer.
    self.y tracks the current cursor (decreases as we draw downward).
    Call render() to draw everything and return the final y position.
    A final_y >= margin means content fits on one page.
    """

    def __init__(self, data: dict, sp: SpacingConfig, target,
                 margin: float = MARGIN_RESUME, text_w: float = TEXT_W_RESUME):
        self.data   = data
        self.sp     = sp
        self.margin = margin
        self.text_w = text_w
        self.x0     = margin
        self.x1     = PAGE_W - margin
        self.c      = canvas.Canvas(target, pagesize=letter)
        self.y      = PAGE_H - margin   # cursor starts at top content boundary

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _move(self, pts: float) -> None:
        self.y -= pts

    def _sw(self, text: str, font: str, size: float) -> float:
        return stringWidth(text, font, size)

    def _wrap(self, text: str, font: str, size: float, max_w: float) -> list:
        """Word-wrap text into lines fitting max_w."""
        if not text.strip():
            return [""]
        words = text.split()
        lines, line = [], ""
        for w in words:
            test = (line + " " + w).strip()
            if self._sw(test, font, size) <= max_w:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines or [""]

    # ── drawing primitives ───────────────────────────────────────────────────

    def _draw(self, x: float, y: float, text: str, font: str, size: float,
              color=(0, 0, 0)) -> float:
        """Draw text at (x, y), return width."""
        self.c.setFont(font, size)
        self.c.setFillColorRGB(*color)
        self.c.drawString(x, y, text)
        return self._sw(text, font, size)

    def _draw_right(self, x: float, y: float, text: str, font: str, size: float) -> None:
        self.c.setFont(font, size)
        self.c.setFillColorRGB(0, 0, 0)
        self.c.drawRightString(x, y, text)

    def _draw_center(self, y: float, text: str, font: str, size: float,
                     color=(0, 0, 0)) -> None:
        self.c.setFont(font, size)
        self.c.setFillColorRGB(*color)
        self.c.drawCentredString(PAGE_W / 2, y, text)

    def _link(self, x: float, y: float, text: str, font: str, size: float,
              url: str) -> float:
        """Draw hyperlink text in blue with underline, add clickable annotation. Returns width."""
        w = self._sw(text, font, size)
        self.c.setFont(font, size)
        self.c.setFillColorRGB(*LINK_COLOR)
        self.c.drawString(x, y, text)
        self.c.setStrokeColorRGB(*LINK_COLOR)
        self.c.setLineWidth(0.5)
        self.c.line(x, y - 1, x + w, y - 1)
        self.c.linkURL(url, (x, y - 2, x + w, y + size), relative=0)
        return w

    def _section_line(self, y: float) -> None:
        """Dark-blue rule 2pt below the text baseline."""
        self.c.setStrokeColorRGB(*SECTION_BLUE)
        self.c.setLineWidth(0.5)
        self.c.line(self.x0, y - 2, self.x1, y - 2)

    # ── element renderers ────────────────────────────────────────────────────

    def _draw_name(self) -> None:
        name = self.data.get("name", "")
        if not name:
            return
        self._move(18)   # move to baseline (font size = 18pt)
        self._draw_center(self.y, name, FONT_BOLD, 18)
        self._move(self.sp.name_after)

    def _draw_contact(self, font_size: float = 11) -> None:
        sp = self.sp
        contact = self.data.get("contact", {})
        linkedin = contact.get("linkedin", "").strip()
        github   = contact.get("github",   "").strip()
        email    = contact.get("email",    "").strip()
        phone    = contact.get("phone",    "").strip()

        sep = ("  |  ", FONT_REG, font_size, None)
        pieces = []
        if linkedin:
            pieces.append((linkedin, FONT_REG, font_size, _LINKEDIN_URL))
        if github:
            if pieces: pieces.append(sep)
            pieces.append((github, FONT_REG, font_size, _GITHUB_URL))
        if email:
            if pieces: pieces.append(sep)
            pieces.append((email, FONT_REG, font_size, None))
        if phone:
            if pieces: pieces.append(sep)
            pieces.append((phone, FONT_REG, font_size, None))

        if not pieces:
            return

        total_w = sum(self._sw(t, f, s) for t, f, s, _ in pieces)
        x = PAGE_W / 2 - total_w / 2
        self._move(font_size)   # baseline for this font size
        for text, font, size, url in pieces:
            if url:
                x += self._link(x, self.y, text, font, size, url)
            else:
                x += self._draw(x, self.y, text, font, size)
        self._move(sp.contact_after)

    def _draw_section_header(self, heading: str, before: float, after: float) -> None:
        self._move(before)
        self._move(14.4)   # baseline: line height for 12pt (1.2×)
        self._draw(self.x0, self.y, heading, FONT_BOLD, 12)
        self._section_line(self.y)
        self._move(after)

    def _draw_summary(self) -> None:
        sp = self.sp
        summary = self.data.get("summary", "").strip()
        if not summary:
            return
        self._draw_section_header("SUMMARY", sp.summary_section_before, sp.summary_section_after)
        self._move(sp.summary_text_before)
        for line in self._wrap(summary, FONT_REG, 11, self.text_w):
            self._move(sp.bullet_leading)
            self._draw(self.x0, self.y, line, FONT_REG, 11)
        self._move(sp.summary_text_after)

    def _draw_skills(self, section: dict) -> None:
        sp = self.sp
        # Collect all skills into one flat string
        skills_str = ""
        for item in section.get("items", []):
            s = item.get("subtitle", "").strip()
            if s:
                skills_str = s
                break
        if not skills_str:
            return
        if not skills_str.endswith("."):
            skills_str += "."

        # Render as inline bold label + regular content (no section header / rule)
        self._move(sp.skills_before)
        self._move(14.4)  # line height matching section header baseline
        label   = "Skills and Tools: "
        label_w = self._sw(label, FONT_BOLD, 11)
        avail_w = self.text_w - label_w
        first_lines = self._wrap(skills_str, FONT_REG, 11, avail_w)
        first_line  = first_lines[0]
        remaining   = " ".join(skills_str.split()[len(first_line.split()):])
        cont_lines  = self._wrap(remaining, FONT_REG, 11, self.text_w) if remaining else []
        self.c.setFont(FONT_BOLD, 11)
        self.c.setFillColorRGB(0, 0, 0)
        self.c.drawString(self.x0, self.y, label)
        self.c.setFont(FONT_REG, 11)
        self.c.drawString(self.x0 + label_w, self.y, first_line)
        for cont in cont_lines:
            self._move(sp.bullet_leading)
            self._draw(self.x0, self.y, cont, FONT_REG, 11)
        self._move(sp.skills_after)

    def _draw_experience(self, section: dict) -> None:
        sp = self.sp
        self._draw_section_header(
            section.get("title", "").upper(), sp.section_before, sp.section_after
        )
        for idx, item in enumerate(section.get("items", [])):
            title    = item.get("title",    "").strip()
            subtitle = item.get("subtitle", "").strip()
            date     = item.get("date",     "").strip()
            bullets  = item.get("bullets",  [])

            before = sp.item_first_before if idx == 0 else sp.item_between_before
            self._move(before)
            self._move(sp.bullet_leading)   # company line

            company = subtitle or title
            self.c.setFont(FONT_BOLD, 11)
            self.c.setFillColorRGB(0, 0, 0)
            self.c.drawString(self.x0, self.y, company)
            if date:
                self._draw_right(self.x1, self.y, date, FONT_REG, 11)

            if subtitle and title:
                self._move(sp.bullet_leading)
                self._draw(self.x0, self.y, title, FONT_ITAL, 11)

            for b in bullets:
                self._draw_bullet(b)

    def _draw_education(self, section: dict) -> None:
        sp = self.sp
        self._draw_section_header(
            section.get("title", "").upper(), sp.section_before, sp.section_after
        )
        last_sub = None
        for item in section.get("items", []):
            title    = item.get("title",    "").strip()
            subtitle = item.get("subtitle", "").strip()
            date     = item.get("date",     "").strip()
            bullets  = item.get("bullets",  [])

            if subtitle and subtitle != last_sub:
                self._move(sp.edu_before)
                self._move(sp.bullet_leading)
                self._draw(self.x0, self.y, subtitle, FONT_BOLD, 11)
                last_sub = subtitle

            if title:
                self._move(sp.bullet_leading)
                self.c.setFont(FONT_REG, 11)
                self.c.setFillColorRGB(0, 0, 0)
                self.c.drawString(self.x0, self.y, title)
                if date:
                    self._draw_right(self.x1, self.y, date, FONT_REG, 11)

            for b in bullets:
                self._draw_bullet(b)

    def _draw_coursework(self, section: dict) -> None:
        sp = self.sp
        self._move(sp.coursework_before)
        self._move(sp.bullet_leading)
        self._draw(self.x0, self.y, "Relevant Coursework:", FONT_BOLD, 11)
        self._move(sp.coursework_after)
        for item in section.get("items", []):
            for bullet in item.get("bullets", []):
                self._draw_bullet(bullet, is_coursework=True)

    def _draw_bullet(self, text: str, is_coursework: bool = False) -> None:
        sp       = self.sp
        bullet_x = self.x0 + (BULLET_LEFT - BULLET_HANG)
        text_x   = self.x0 + BULLET_LEFT
        avail_w  = self.text_w - BULLET_LEFT

        if is_coursework and ": " in text:
            course, desc = text.split(": ", 1)
            if not desc.endswith("."):
                desc += "."
            # Wrap using regular-weight metrics (italic width is ~equal for Helvetica)
            lines = self._wrap(course + ": " + desc, FONT_REG, 11, avail_w)

            self._move(sp.bullet_leading)
            self.c.setFillColorRGB(0, 0, 0)
            self._draw(bullet_x, self.y, BULLET_CHAR, FONT_REG, 11)

            # First line: italic course name + regular rest
            first = lines[0]
            if first.startswith(course):
                self.c.setFont(FONT_ITAL, 11)
                self.c.drawString(text_x, self.y, course)
                rest_text = first[len(course):]
                cx = text_x + self._sw(course, FONT_ITAL, 11)
                self.c.setFont(FONT_REG, 11)
                self.c.drawString(cx, self.y, rest_text)
            else:
                self._draw(text_x, self.y, first, FONT_REG, 11)

            for cont in lines[1:]:
                self._move(sp.bullet_leading)
                self._draw(text_x, self.y, cont, FONT_REG, 11)
            self._move(sp.bullet_after)
        else:
            lines = self._wrap(text, FONT_REG, 11, avail_w)
            for i, line in enumerate(lines):
                self._move(sp.bullet_leading)
                self.c.setFont(FONT_REG, 11)
                self.c.setFillColorRGB(0, 0, 0)
                if i == 0:
                    self.c.drawString(bullet_x, self.y, BULLET_CHAR)
                self.c.drawString(text_x, self.y, line)
            self._move(sp.bullet_after)

    def render(self) -> float:
        """Draw the full resume. Returns final y position."""
        self._draw_name()
        self._draw_contact()
        self._draw_summary()

        for section in self.data.get("sections", []):
            key    = section.get("title", "").lower()
            is_exp = any(x in key for x in _EXPERIENCE_SECTIONS)
            is_edu = any(x in key for x in _EDUCATION_SECTIONS)
            is_sk  = any(x in key for x in _SKILLS_SECTIONS)
            is_cw  = any(x in key for x in _COURSEWORK_SECTIONS)

            if   is_sk:  self._draw_skills(section)
            elif is_cw:  self._draw_coursework(section)
            elif is_exp: self._draw_experience(section)
            elif is_edu: self._draw_education(section)

        self.c.save()
        return self.y


def _render_resume(data: dict, sp: SpacingConfig, target) -> float:
    """Render resume to target (path or BytesIO). Returns final y position."""
    return _PdfRenderer(data, sp, target).render()


# --------------------------------------------------------------------------- #
# Page-fit loop
# --------------------------------------------------------------------------- #

# If final_y is more than 0.75" above the bottom margin, the page has too much
# whitespace and we attempt to expand coursework keywords.
_TOO_SHORT_THRESHOLD = MARGIN_RESUME + 54   # 54 pt = 0.75"


def _try_spacing_levels(data: dict, output_path: str | None = None) -> tuple[dict, int, float]:
    """
    Dry-run all spacing levels against data.
    Returns (data, best_level, final_y) for the first level that fits, or
    (data, -1, best_final_y) if none fit (best_final_y = least-overflowing level).
    If output_path is given, writes the file when a fit is found.
    """
    best_y = -999
    for level, sp in enumerate(SPACING_LEVELS):
        final_y = _render_resume(data, sp, BytesIO())
        if final_y >= MARGIN_RESUME:
            if output_path:
                _render_resume(data, sp, output_path)
            return data, level, final_y
        if final_y > best_y:
            best_y = final_y
    return data, -1, best_y


def fit_resume_to_one_page(data: dict, jd_text: str, output_path: str) -> None:
    """
    Fit the resume to exactly one page using a two-stage strategy:
      1. Try SPACING_LEVELS 0–4 (tighten whitespace, no content change).
      2. If still overflowing, ask Claude to trim bullet/summary wording, then retry.
    If level 0 fits with >0.75" slack, ask Claude to expand coursework keywords.
    """
    # Stage 1: spacing adjustments only
    _, level, final_y = _try_spacing_levels(data, output_path=None)

    if level >= 0:
        # Fits — check if too short on level 0
        sp = SPACING_LEVELS[level]
        if level == 0 and final_y > _TOO_SHORT_THRESHOLD:
            slack = final_y - MARGIN_RESUME
            print(f"  Spacing level 0: fits with {slack:.1f}pt slack — expanding coursework...")
            try:
                expanded = expand_coursework(data, jd_text)
                exp_y = _render_resume(expanded, sp, BytesIO())
                if exp_y >= MARGIN_RESUME:
                    _render_resume(expanded, sp, output_path)
                    print(f"  Page fit: expanded coursework, {exp_y - MARGIN_RESUME:.1f}pt from bottom.")
                    return
                else:
                    print("  Expanded version overflows — using original content.")
            except Exception as e:
                print(f"  Coursework expansion failed ({e}) — using original content.")
        _render_resume(data, sp, output_path)
        print(f"  Page fit: spacing level {level}, {final_y - MARGIN_RESUME:.1f}pt from bottom.")
        return

    # Stage 2: content trim (up to 2 rounds)
    current = data
    for trim_round in range(1, 3):
        overage = MARGIN_RESUME - final_y
        print(f"  All spacing levels exhausted ({overage:.1f}pt over) "
              f"— asking Claude to trim content (round {trim_round})...")
        try:
            current = trim_resume_content(current)
            _, trim_level, trim_y = _try_spacing_levels(current, output_path=None)
            if trim_level >= 0:
                _render_resume(current, SPACING_LEVELS[trim_level], output_path)
                print(f"  Page fit: trim round {trim_round}, spacing level {trim_level}, "
                      f"{trim_y - MARGIN_RESUME:.1f}pt from bottom.")
                return
            final_y = trim_y   # update for next round's overage message
        except Exception as e:
            print(f"  Content trim round {trim_round} failed ({e}).")
            break

    print(f"  Could not fit to 1 page after trimming — using tightest spacing.")
    _render_resume(current, SPACING_LEVELS[-1], output_path)


# --------------------------------------------------------------------------- #
# Cover letter PDF renderer
# --------------------------------------------------------------------------- #

def render_cover_letter_pdf(cl_data: dict, output_path: str) -> None:
    m    = MARGIN_CL
    tw   = TEXT_W_CL
    x0   = m
    c    = canvas.Canvas(output_path, pagesize=letter)
    y    = PAGE_H - m

    def move(pts):
        nonlocal y
        y -= pts

    def sw(text, font, size):
        return stringWidth(text, font, size)

    def wrap(text, font, size, max_w):
        if not text.strip():
            return [""]
        words = text.split()
        lines, line = [], ""
        for w in words:
            test = (line + " " + w).strip()
            if sw(test, font, size) <= max_w:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines or [""]

    name    = cl_data.get("name", "")
    contact = cl_data.get("contact", {})
    email   = contact.get("email",    "").strip()
    phone   = contact.get("phone",    "").strip()
    linkedin = contact.get("linkedin", "").strip()
    github   = contact.get("github",   "").strip()

    # No top header — start body at top margin
    move(13.2)

    def body_para(text, space_after=12):
        if not text or not text.strip():
            return
        lines = wrap(text.strip(), FONT_REG, 11, tw)
        for line in lines:
            move(13.2)
            c.setFont(FONT_REG, 11)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(x0, y, line)
        move(space_after)

    bullet_indent  = 18   # left indent for bullet text

    def bullet_para(text, space_after=4):
        if not text or not text.strip():
            return
        char_w   = sw(BULLET_CHAR + " ", FONT_REG, 11)
        avail_w  = tw - bullet_indent
        lines    = wrap(text.strip(), FONT_REG, 11, avail_w)
        for i, line in enumerate(lines):
            move(13.2)
            c.setFont(FONT_REG, 11)
            c.setFillColorRGB(0, 0, 0)
            if i == 0:
                c.drawString(x0 + bullet_indent - char_w, y, BULLET_CHAR)
                c.drawString(x0 + bullet_indent, y, line)
            else:
                c.drawString(x0 + bullet_indent, y, line)
        move(space_after)

    body_para(cl_data.get("salutation", ""))
    body_para(cl_data.get("opening", ""))

    bullet_intro = cl_data.get("bullet_intro", "").strip()
    bullets      = cl_data.get("bullets", [])
    if bullet_intro and bullets:
        body_para(bullet_intro, space_after=2)
        for i, b in enumerate(bullets):
            after = 2 if i < len(bullets) - 1 else 10
            bullet_para(b, space_after=after)

    for para in cl_data.get("body_paragraphs", []):
        body_para(para)
    body_para(cl_data.get("closing", ""))
    body_para(cl_data.get("sign_off", "Sincerely,"), space_after=0)

    # Signature gap + printed name + contact info
    move(24)
    move(13.2)
    c.setFont(FONT_REG, 11)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x0, y, name)

    # Contact line below name: email | phone | LinkedIn | GitHub
    sep = "  |  "
    pieces = []
    if email:
        pieces.append((email, FONT_REG, 10, None))
    if phone:
        if pieces: pieces.append((sep, FONT_REG, 10, None))
        pieces.append((phone, FONT_REG, 10, None))
    if linkedin:
        if pieces: pieces.append((sep, FONT_REG, 10, None))
        pieces.append((linkedin, FONT_REG, 10, _LINKEDIN_URL))
    if github:
        if pieces: pieces.append((sep, FONT_REG, 10, None))
        pieces.append((github, FONT_REG, 10, _GITHUB_URL))

    if pieces:
        move(13.2)
        cx = x0
        for text, font, size, url in pieces:
            piece_w = sw(text, font, size)
            c.setFont(font, size)
            if url:
                c.setFillColorRGB(*LINK_COLOR)
                c.drawString(cx, y, text)
                c.setStrokeColorRGB(*LINK_COLOR)
                c.setLineWidth(0.5)
                c.line(cx, y - 1, cx + piece_w, y - 1)
                c.linkURL(url, (cx, y - 2, cx + piece_w, y + size), relative=0)
            else:
                c.setFillColorRGB(0, 0, 0)
                c.drawString(cx, y, text)
            cx += piece_w

    c.save()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _collect_answers() -> str:
    print("\nYour answers (press Enter on a blank line when done):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Tailor your resume (and optionally cover letter) to a job description."
    )
    parser.add_argument("--resume", required=True, metavar="FILE",
                        help="Resume file: PDF, DOCX, or TXT")
    parser.add_argument("--jd", required=True, metavar="FILE",
                        help="Job description file: PDF, DOCX, or TXT")
    parser.add_argument("--cover-letter", metavar="FILE",
                        help="Cover letter master file: PDF, DOCX, or TXT")
    parser.add_argument("--cover-letter-only", action="store_true",
                        help="Generate only the cover letter (requires --cover-letter); skips resume generation")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Stream analysis, save questions to .questions_cache.json, then exit")
    parser.add_argument("--answers-file", metavar="FILE",
                        help="File with pre-written answers to clarifying questions (skips interactive prompt)")
    args = parser.parse_args()

    # Validate inputs
    for fpath, label in [(args.resume, "--resume"), (args.jd, "--jd")]:
        if not Path(fpath).exists():
            print(f"Error: {label} file not found: {fpath}", file=sys.stderr)
            sys.exit(1)
    if args.cover_letter and not Path(args.cover_letter).exists():
        print(f"Error: --cover-letter file not found: {args.cover_letter}", file=sys.stderr)
        sys.exit(1)
    if args.cover_letter_only and not args.cover_letter:
        print("Error: --cover-letter-only requires --cover-letter <FILE>", file=sys.stderr)
        sys.exit(1)

    # Load optional candidate context
    candidate_context = ""
    if Path(CANDIDATE_CONTEXT_PATH).exists():
        candidate_context = Path(CANDIDATE_CONTEXT_PATH).read_text(encoding="utf-8")

    # Extract text
    print("Extracting text...")
    resume_text = extract_text(args.resume)
    jd_text     = extract_text(args.jd)

    # --cover-letter-only: skip resume generation
    if args.cover_letter_only:
        print("\nExtracting cover letter master...")
        cl_text = extract_text(args.cover_letter)
        print("Tailoring cover letter...")
        tailored_cl = tailor_cover_letter(
            cl_text, resume_text, jd_text, candidate_context=candidate_context
        )
        company = _safe_slug(tailored_cl.get("company", "Company"))
        cl_out  = f"TimChristian_{company}_CoverLetter.pdf"
        render_cover_letter_pdf(tailored_cl, cl_out)
        print(f"Saved → {cl_out}")
        print("\nDone.")
        sys.exit(0)

    # If answers already provided, skip analysis
    if args.answers_file:
        if not Path(args.answers_file).exists():
            print(f"Error: --answers-file not found: {args.answers_file}", file=sys.stderr)
            sys.exit(1)
        user_answers = Path(args.answers_file).read_text(encoding="utf-8")
    else:
        print("\n" + "─" * 60)
        print("Analyzing JD and generating clarifying questions...\n")
        analysis_text = analyze_and_question(resume_text, jd_text, candidate_context=candidate_context)

        if args.analyze_only:
            print("\n" + "─" * 60)
            print("Extracting questions...", end="", flush=True)
            questions = _extract_questions_json(analysis_text)
            Path(QUESTIONS_CACHE_PATH).write_text(json.dumps(questions, indent=2), encoding="utf-8")
            print(f" done. {len(questions)} question(s) saved to {QUESTIONS_CACHE_PATH}")
            sys.exit(0)

        user_answers = _collect_answers()

    # Generate optimized resume
    print("\n" + "─" * 60)
    print("Generating optimized resume...\n")
    resume_data = tailor_resume(
        resume_text, jd_text,
        user_answers=user_answers,
        candidate_context=candidate_context,
    )

    company    = _safe_slug(resume_data.get("company", "Company"))
    resume_out = f"TimChristian_{company}.pdf"

    print("Checking page fit...")
    fit_resume_to_one_page(resume_data, jd_text, resume_out)
    print(f"Saved → {resume_out}")

    # Optional cover letter
    if args.cover_letter:
        print("\nExtracting cover letter...")
        cl_text = extract_text(args.cover_letter)
        print("Tailoring cover letter...")
        tailored_cl = tailor_cover_letter(
            cl_text, resume_text, jd_text, candidate_context=candidate_context
        )
        cl_out = f"TimChristian_{company}_CoverLetter.pdf"
        render_cover_letter_pdf(tailored_cl, cl_out)
        print(f"Saved → {cl_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
