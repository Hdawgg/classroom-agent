"""
Agentic study mode: type a request in plain English and a LangGraph ReAct agent decides which tools to call.

    python agent.py --study --agent             (or: python study.py --agent, python study_agent.py)
    python agent.py --study --agent --offline   sample data + scripted fake LLM, no Google or LLM calls

Examples: "Make revision notes for the Module 2 slides", "What's due this week in my AI courses?"

The tools reuse study.py and classroom_reader.py (search, details view, cached read-only Drive reader, the
notes/questions pipeline with chunking, retries and resumable .parts). Credit protection is enforced in code,
not left to the prompt:
  - generate_notes / generate_questions show the usual estimate and wait for y/N before any LLM call;
    after a "no", every generate call is blocked for the rest of that request;
  - a per-session cap on LLM calls (--max-calls, default 30) that counts the agent's own reasoning calls;
  - a per-request limit on tool calls (--max-steps, default 15);
  - a tool call that failed twice with the same arguments is not run again.
Every request is traced live (Thought / Action / Observation) and saved to study/agent_logs/<timestamp>.md.
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph

import classroom_reader
import study
from tool_names import normalize_tool_calls
from google_classroom import course_matches

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = study.BASE_DIR
LOG_DIR = study.STUDY_DIR / "agent_logs"
DEFAULT_MAX_CALLS = 30
DEFAULT_MAX_STEPS = 15
PREVIEW_CHARS = 800  # read_item_text default; never more than MAX_PREVIEW goes into the conversation
MAX_PREVIEW = 2000
SEARCH_SHOWN = 12
SEARCH_OPEN_FILES = 5  # with this many matches or fewer, search opens their Drive files to report readability
LOG_TEXT = 3000  # longest tool output / thought kept in the trace file


class SessionLimit(BaseException):
    """The session's LLM call cap was reached. A BaseException, so study.ask()'s retries and run_jobs()'s
    per-item error handling (both catch Exception) let it through and the whole request stops."""


class RequestStop(Exception):
    """Stops the current request (step limit, model failure); the session continues."""


class ToolError(Exception):
    """A tool failed in a way the agent should be told about (unknown id, bad course name...)."""


# ============================================================
# CREDIT ACCOUNTING
# ============================================================
class CallBudget:
    """Counts every LLM call in the session (agent reasoning + generation, retries included)."""

    def __init__(self, cap: int):
        self.cap, self.used, self.agent, self.generation = cap, 0, 0, 0

    def left(self) -> int:
        return self.cap - self.used

    def spend(self, kind: str) -> None:
        if self.used >= self.cap:
            raise SessionLimit(f"the session limit of {self.cap} LLM calls (--max-calls) was reached")
        self.used += 1
        setattr(self, kind, getattr(self, kind) + 1)


class BudgetedLLM:
    """Wraps the generation model so each call made by study.ask() (including retries) is counted first."""

    def __init__(self, llm, budget: CallBudget):
        self._llm, self._budget = llm, budget

    @property
    def model(self) -> str:  # study.model_name() reads this; keeps .parts file names the same as the menu's
        return study.model_name(self._llm)

    def invoke(self, prompt):
        self._budget.spend("generation")
        return self._llm.invoke(prompt)


# ============================================================
# TRACE (console + study/agent_logs/<timestamp>.md)
# ============================================================
SECRET_RES = [re.compile(p) for p in (
    r"nvapi-[\w-]{10,}", r"gsk_\w{10,}", r"\bsk-[\w-]{16,}", r"ya29\.[\w.-]+", r"\b1//[\w-]{20,}", r"GOCSPX-[\w-]+",
    r"(?i)(\"?(?:access_token|refresh_token|client_secret|api_key)\"?\s*[:=]\s*\"?)[^\s\",]+",
)]


def redact(text: str) -> str:
    """Removes API keys and OAuth tokens: values of *KEY/*TOKEN/*SECRET env vars, and well-known key formats."""
    text = str(text)
    for k, v in os.environ.items():
        if v and len(v) >= 8 and re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I):
            text = text.replace(v, "[REDACTED]")
    for r in SECRET_RES:
        text = r.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def _clip(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + f"\n... [{len(text) - n:,} more chars not shown]"


def _one_line(text: str, n: int = 160) -> str:
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    first = lines[0] if lines else "(empty)"
    more = f"  (+{len(lines) - 1} more line(s))" if len(lines) > 1 else ""
    return (first if len(first) <= n else first[:n - 3] + "...") + more


def _args_short(args: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" if len(repr(v)) <= 60 else f"{k}={repr(v)[:57]}...'" for k, v in args.items())


def _fence(text: str, lang: str = "text") -> str:
    return f"````{lang}\n{text}\n````"


class Trace:
    """Prints each step live and appends it to a Markdown log as it happens (so a crash still leaves a log)."""

    def __init__(self, request: str, session: "StudyAgentSession"):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = LOG_DIR / f"{stamp}.md"
        n = 2
        while self.path.exists():
            self.path = LOG_DIR / f"{stamp}_{n}.md"
            n += 1
        self.llm_calls = 0
        b = session.budget
        self._write(
            f"# Study agent trace: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
            f"- **Request:** {request}\n"
            f"- **Mode:** {'offline (sample data, scripted fake LLM)' if session.offline else 'live (Google Classroom, read-only)'}\n"
            f"- **Agent model:** {study.model_name(session.agent_llm)} · **Generation model:** {study.model_name(session.gen_llm)}\n"
            f"- **Limits:** {session.max_steps} tool calls per request; {b.cap} LLM calls per session "
            f"({b.used} used before this request)\n"
            f"- **Earlier requests in this session:** {session.requests_done}\n\n## Steps\n")

    def _write(self, text: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(redact(text) + "\n")

    def llm_call(self, msg: AIMessage, thought: str) -> None:
        self.llm_calls += 1
        self._write(f"\n### Agent LLM call {self.llm_calls}\n")
        if thought:
            print(f"\n  Thought: {redact(_one_line(thought, 200))}")
            self._write(f"**Thought:**\n\n{_fence(_clip(thought, LOG_TEXT))}\n")

    def action(self, step: int, name: str, args: Dict[str, Any]) -> None:
        print(f"  [{step}] Action: {name}({redact(_args_short(args))})")
        self._write(f"**Action {step}:** `{name}`\n\n{_fence(json.dumps(args, ensure_ascii=False, indent=1), 'json')}\n")

    def note(self, text: str) -> None:
        print(f"      note: {redact(text)}")
        self._write(f"`note: {text}`\n\n")

    def observation(self, step: int, result: str) -> None:
        print(f"      Observation: {redact(_one_line(result))}")
        self._write(f"**Observation {step}** ({len(result):,} chars):\n\n{_fence(_clip(result, LOG_TEXT))}\n")

    def event(self, title: str, text: str = "") -> None:
        self._write(f"\n**{title}**\n" + (f"\n{_fence(text)}\n" if text else ""))

    def final(self, text: str) -> None:
        print("\n" + "-" * 78 + f"\nAnswer:\n{redact(text)}\n" + "-" * 78)
        self._write(f"\n## Final answer\n\n{text}\n")

    def summary(self, lines: List[str]) -> None:
        self._write("\n## Summary\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")


def text_of(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) and b.get("type", "text") == "text" else
                          (b if isinstance(b, str) else "") for b in content)
    return str(content or "").strip()


def thought_of(msg: AIMessage) -> str:
    """The model's visible reasoning: reasoning_content (gpt-oss etc.), reasoning blocks, or text sent with tool calls."""
    parts = []
    kw = getattr(msg, "additional_kwargs", {}) or {}
    rc = kw.get("reasoning_content") or kw.get("reasoning")
    if isinstance(rc, dict):
        rc = rc.get("content") or rc.get("summary") or ""
    if rc:
        parts.append(str(rc).strip())
    if isinstance(msg.content, list):
        parts += [str(b.get("reasoning") or b.get("text") or "") for b in msg.content
                  if isinstance(b, dict) and b.get("type") == "reasoning"]
    if msg.tool_calls and text_of(msg):
        parts.append(text_of(msg))
    return "\n".join(p for p in parts if p)


# ============================================================
# SESSION: data, tools, graph
# ============================================================
SYSTEM_PROMPT = """You are a study assistant for a university student. You work with their Google Classroom \
courses through tools (read-only: you can never submit or change anything in Classroom).
Today is {today}. The student's courses: {courses}.

How to work:
1. Find the right course and item with search_items first (list_courses if you need an overview). Search with \
short keywords such as "Module 5", "expert systems" or "A*", not the whole request. If nothing matches, try a \
shorter keyword or a synonym (e.g. "Mod 5", the topic name).
2. If several items could be what the user means and you cannot tell which one, list the candidates (number, \
label, kind, posted date, id) and ask the user to choose. Never guess.
3. Before generating, check the item with get_item_details (read_item_text gives a short preview). If an item has \
no readable text or very little (under ~200 characters), say so and ask the user before generating, because \
the result would be generic.
4. generate_notes writes revision notes; generate_questions writes practice questions with answers (10 unless \
the user asks for a number). Prefer one generate call per item. They spend LLM credits and ask the user to \
confirm. If a tool returns "User declined", do not retry or work around it: tell the user nothing was generated.
5. You cannot quiz the user interactively yet: for "quiz me" requests, generate practice questions and point to \
the file.
6. For "what's due" questions use list_pending_work; say that submission status is not checked.
7. Never invent course content, dates or deadlines: only report what the tools returned. If a tool keeps \
failing, stop and report the problem.
8. Finish with a short answer. When files were written, list their paths exactly as the tools returned them."""


class _RequestState:
    def __init__(self):
        self.steps = 0
        self.failures: Dict[str, int] = {}
        self.declined = False
        self.files: List[str] = []
        self.tool_results: Dict[str, str] = {}
        self.renamed: Dict[str, str] = {}  # call id -> raw tool name the model sent (see tool_names.py)


class StudyAgentSession:
    def __init__(self, courses, drive, agent_llm, gen_llm, args, offline: bool):
        self.courses, self.drive, self.args, self.offline = courses, drive, args, offline
        self.budget = CallBudget(args.max_calls)
        self.max_steps = args.max_steps
        self.agent_llm = agent_llm
        self.gen_llm = BudgetedLLM(gen_llm, self.budget)
        self.items = {item["id"]: (item, course) for course in courses for item in course["items"]}
        self.opened = set()  # ids of items whose Drive files were read this session
        self.history: List[Any] = []
        self.requests_done = 0
        self.over = False
        self.req = _RequestState()
        self.trace: Optional[Trace] = None
        self.tools = self._make_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self.model = agent_llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    # ---------- helpers ----------
    def _find(self, item_id: str):
        key = str(item_id).strip().strip("'\"")
        if key.lower().startswith("id="):
            key = key[3:]
        if key in self.items:
            return self.items[key]
        raise ToolError(f"No item with id '{item_id}'. Use the exact id shown by search_items.")

    def _courses(self, course: Optional[str]):
        if not course or course.strip().lower() in ("all", "any", "none", ""):
            return self.courses
        found = [c for c in self.courses if course_matches(c["name"], course)]
        if not found:  # the model may abbreviate ("OS" for Operating Systems) or use part of a word
            f = course.strip().lower()
            found = [c for c in self.courses
                     if f == "".join(w[0] for w in re.findall(r"[A-Za-z]+", re.sub(r"\(.*?\)", "", c["name"]))).lower()
                     or (len(f) >= 4 and f in c["name"].lower())]
        if not found:
            raise ToolError(f"No course matches '{course}'. Courses: " + "; ".join(c["name"] for c in self.courses))
        return found

    def _open(self, item) -> None:
        """Reads the item's Drive attachments (read-only, through the .study_cache cache) once per session."""
        if self.drive is None or item["id"] in self.opened:
            return
        todo = [a for a in study.drive_files(item) if "status" not in a]
        if todo:
            print(f"      (opening {len(todo)} Drive file(s) of '{study.display(item)[:50]}', read-only, cached)")
            classroom_reader.read_attachments(self.drive, [item], refresh=self.args.refresh,
                                              progress=lambda m: print("    " + m))
        self.opened.add(item["id"])

    @staticmethod
    def _readability(item) -> str:
        files = study.drive_files(item)
        links = len(item["attachments"]) - len(files)
        unopened = sum(1 for a in files if "status" not in a)
        if unopened:
            return f"{len(files)} Drive file(s) not opened yet (get_item_details opens them); message/description {len(item['description']):,} chars"
        chars = study.readable_chars(item)
        parts = [f"{chars:,} chars"]
        if files:
            parts.append(f"{sum(1 for a in files if a.get('status') == 'read')} of {len(files)} Drive file(s) read")
        if links:
            parts.append(f"{links} link(s)/video(s) not fetched")
        if chars < study.LITTLE_TEXT:
            parts.append("TOO LITTLE TEXT: generated output would be generic")
        return ", ".join(parts)

    def _row(self, n: int, item, extra: str = "") -> str:
        bits = [f"id={item['id']}", item["kind"], f"posted {item.get('posted') or 'unknown'}", item["course"]]
        if item["topic"]:
            bits.append(f"topic: {item['topic']}")
        bits.append(f'"{study.display(item)}"')
        if item["due"]:
            bits.append(f"due {item['due']}")
        bits.append(f"readable: {self._readability(item)}")
        if extra:
            bits.append(extra)
        return f"{n}. " + " | ".join(bits)

    # ---------- tools ----------
    def _make_tools(self):
        session = self

        @tool
        def list_courses() -> str:
            """List the student's courses with how many assignments, materials and announcements each has. No LLM credits."""
            lines = [f"{len(session.courses)} course(s):"]
            for n, c in enumerate(session.courses, 1):
                kinds = {}
                for i in c["items"]:
                    kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
                counts = ", ".join(f"{v} {k.lower()}{'s' if v != 1 else ''}" for k, v in sorted(kinds.items())) or "no items"
                lines.append(f"{n}. {c['name']}" + (f" ({c['section']})" if c.get("section") else "") + f": {counts}")
                lines += [f"   warning: {w}" for w in c.get("warnings", [])]
            return "\n".join(lines)

        @tool
        def search_items(query: str, course: Optional[str] = None) -> str:
            """Keyword search over item titles/labels, messages/descriptions and attachment file names (items that
            match only inside a file's text are flagged). Use short keywords like 'Module 5' or 'expert systems'.
            `course` limits the search to courses whose name matches that word (e.g. 'AI').
            Returns numbered matches with their item ids and readability. No LLM credits."""
            q = (query or "").strip()
            if not q:
                raise ToolError("query is empty: pass a keyword such as 'Module 2'")
            courses = session._courses(course)
            # The all-keywords fallback ignores short words ("a", "of", "5"), which would match almost anything
            words = [w for w in q.split() if len(re.sub(r"[\W_]+", "", w)) >= 3]
            found = []
            for c in courses:
                for item in c["items"]:
                    whole = study.search_match(item, q)
                    each = [study.search_match(item, w) for w in words] if len(words) > 1 else []
                    if whole == "item":
                        tier, how = 0, "match in title/text/file names"
                    elif each and all(each) and all(m == "item" for m in each):
                        tier, how = 1, "all keywords match (not as one phrase)"
                    elif whole == "file" or (each and all(each)):
                        tier, how = 2, "match only inside attached file text"
                    else:
                        continue
                    found.append((tier, item, how))
            found.sort(key=lambda t: t[1].get("created", ""), reverse=True)
            found.sort(key=lambda t: t[0])
            where = "all courses" if courses is session.courses else "; ".join(c["name"] for c in courses)
            if not found:
                return (f"No items match '{q}' in {where}. Try a shorter keyword, a different spelling "
                        "(e.g. 'Module-2', 'Mod 2') or the topic name.")
            if len(found) <= SEARCH_OPEN_FILES:
                for _, item, _ in found:
                    session._open(item)
            lines = [f"{len(found)} item(s) match '{q}' in {where}:"]
            lines += [session._row(n, item, how) for n, (_, item, how) in enumerate(found[:SEARCH_SHOWN], 1)]
            if len(found) > SEARCH_SHOWN:
                lines.append(f"(+{len(found) - SEARCH_SHOWN} more not shown: use a more specific keyword or the course filter)")
            if len(found) > 1:
                lines.append("If more than one of these could be what the user means, list them and ask which one. Do not guess.")
            return "\n".join(lines)

        @tool
        def get_item_details(item_id: str) -> str:
            """Details of one item: message/description (first 600 chars), each attachment with its read status or
            why it was skipped, readable text size and output folder. Opens the item's Drive files (read-only,
            cached) if needed. No LLM credits."""
            item, course = session._find(item_id)
            session._open(item)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                study.show_details(item, course["items"])
            chars = study.readable_chars(item)
            verdict = ("enough for notes/questions" if chars >= study.LITTLE_TEXT else
                       "TOO LITTLE: tell the user and ask before generating, the result would be generic")
            return (f"id={item['id']} | course: {course['name']}\n{buf.getvalue().strip()}\n\n"
                    f"Readable course text: {chars:,} chars ({verdict}).")

        @tool
        def read_item_text(item_id: str, max_chars: int = PREVIEW_CHARS) -> str:
            """A short preview (max_chars, default 800, at most 2000) of an item's readable course text and its
            total length. The full text is never returned; generate_notes/generate_questions use all of it. No LLM credits."""
            item, _ = session._find(item_id)
            session._open(item)
            text = study.source_text(item)
            if not text.strip():
                return f"No readable text for id={item['id']} ({session._readability(item)})."
            k = max(100, min(int(max_chars or PREVIEW_CHARS), MAX_PREVIEW))
            return (f"Total readable text: {len(text):,} chars ({session._readability(item)}).\n"
                    f"Preview of the first {min(k, len(text)):,} chars:\n<<<\n{text[:k]}\n>>>"
                    + ("\n(preview cut; the rest is not shown)" if len(text) > k else ""))

        @tool
        def generate_notes(item_id: str) -> str:
            """Write revision notes for ONE item to study/<course>/<date>_<label>/notes.md. Spends LLM credits:
            the user sees an estimate and must confirm. Returns the file path, or 'User declined'."""
            return session._generate(item_id, want_notes=True, n_questions=0)

        @tool
        def generate_questions(item_id: str, n: int = study.DEFAULT_QUESTIONS) -> str:
            """Write n practice questions with answers (conceptual, viva, predict-the-output) for ONE item to
            study/<course>/<date>_<label>/questions.md. Spends LLM credits: the user sees an estimate and must
            confirm. Returns the file path, or 'User declined'."""
            if not 1 <= int(n) <= 50:
                raise ToolError("n must be between 1 and 50")
            return session._generate(item_id, want_notes=False, n_questions=int(n))

        @tool
        def list_pending_work(course: Optional[str] = None) -> str:
            """Due dates from Classroom item metadata (items with a due date from 14 days ago onwards), soonest
            first, with today's date and the end of this week. Submission status is NOT checked. No LLM credits."""
            courses = session._courses(course)
            today = date.today()
            week_end = today + timedelta(days=6 - today.weekday())
            rows, undated = [], 0
            for c in courses:
                for item in c["items"]:
                    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", item.get("due") or "")
                    if not m:
                        undated += item["kind"] == "Assignment"
                        continue
                    d = date(*map(int, m.groups()))
                    if (d - today).days >= -14:
                        rows.append((d, item))
            rows.sort(key=lambda r: r[0])
            lines = [f"Today is {today:%Y-%m-%d (%A)}; this week ends {week_end:%Y-%m-%d (%A)}."]
            if not rows:
                lines.append("No items with a due date from 14 days ago onwards.")
            for d, item in rows:
                days = (d - today).days
                when = ("today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days" if days > 0
                        else f"{-days} day(s) ago, past due")
                flags = (" [this week]" if today <= d <= week_end else "") + (" [within 7 days]" if 0 <= days <= 7 else "")
                lines.append(f"- {d:%Y-%m-%d (%a)}, {when}{flags} | {item['course']} | {item['kind']} | "
                             f"{study.display(item)} | id={item['id']}")
            if undated:
                lines.append(f"{undated} assignment(s) have no due date and are not listed.")
            lines.append("Due dates come from Classroom metadata; whether the work was already submitted is not checked.")
            return "\n".join(lines)

        return [list_courses, search_items, get_item_details, read_item_text, generate_notes, generate_questions,
                list_pending_work]

    def _generate(self, item_id: str, want_notes: bool, n_questions: int) -> str:
        what = "generate_notes" if want_notes else "generate_questions"
        if self.req.declined:
            return ("BLOCKED: the user already declined generation in this request. Do not call generate_notes or "
                    "generate_questions again in this request; tell the user nothing was generated.")
        item, course = self._find(item_id)
        self._open(item)
        llm = self.gen_llm
        jobs = study.plan([item], course["items"], want_notes, n_questions, self.args.chunk_chars, llm)
        calls = jobs[0]["calls"]
        if calls + 1 > self.budget.left():  # +1 so the agent can still report back
            raise ToolError(
                f"{what} needs {calls} LLM call(s) (plus 1 for your reply) but only {self.budget.left()} of this "
                f"session's {self.budget.cap} remain. Do not generate; tell the user to restart with a higher --max-calls.")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            overwrites = study.show_estimate(jobs, llm, n_questions)
        estimate = buf.getvalue()
        print(estimate, end="")
        if study.readable_chars(item) < study.LITTLE_TEXT:
            print(f"\n  [!] This item has only {study.readable_chars(item)} characters of readable course text: "
                  "the result would be mostly generic.")
        print(f"  Session LLM calls: {self.budget.used} used of {self.budget.cap}; this would use {calls} more.")
        question = (f"\nSend to the LLM and overwrite {overwrites} existing file(s)? [y/N]: " if overwrites
                    else "\nSend to the LLM? [y/N]: ")
        answer = study.read_input(question)
        self.trace.event(f"Confirmation asked by {what} (estimate shown to the user)", estimate.strip())
        self.trace.event(f"User answered: {answer!r}")
        if answer.lower() not in ("y", "yes"):
            self.req.declined = True
            print("      Cancelled - nothing was sent.")
            return ("User declined: nothing was sent and no file was written. Do not retry, do not call any generate "
                    "tool again in this request, and do not work around it; tell the user nothing was generated.")

        used_before = self.budget.used
        results = study.run_jobs(llm, jobs, n_questions)
        lines = []
        for r in results:
            rel = str(r["path"].relative_to(BASE_DIR))
            if r["ok"]:
                self.req.files.append(rel)
                lines.append(f"Wrote {rel}" + (f" ({r['note']})" if r["note"] else ""))
            else:
                parts = r["path"].parent / ".parts"
                saved = f" Finished parts are saved in {parts.relative_to(BASE_DIR)}; the same request resumes them." \
                    if parts.exists() else ""
                lines.append(f"ERROR: {r['kind']} for this item failed: {r['note']}.{saved}")
        lines.append(f"LLM calls used: {self.budget.used - used_before} (session: {self.budget.used} of {self.budget.cap}).")
        text = "\n".join(lines)
        return text if any(r["ok"] for r in results) else "ERROR: " + text.removeprefix("ERROR: ")

    # ---------- tool execution with the limits ----------
    def run_tool(self, call_id: str, name: str, args: Dict[str, Any]) -> str:
        self.req.steps += 1
        step = self.req.steps
        if step > self.max_steps:
            raise RequestStop(f"step limit reached: this request already used {self.max_steps} tool calls (--max-steps)")
        self.trace.action(step, name, args)
        if call_id in self.req.renamed:
            self.trace.note(f"tool name normalized from '{self.req.renamed.pop(call_id)}'")
        key =f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if self.req.failures.get(key, 0) >= 2:
            result = (f"ERROR: {name} with these exact arguments already failed twice, so it was not run again. "
                      "Do not call it again. Stop and report the problem to the user.")
            self.req.failures[key] += 1
        else:
            t = self.tools_by_name.get(name)
            try:
                if t is None:
                    raise ToolError(f"unknown tool '{name}'. Available: {', '.join(self.tools_by_name)}")
                result = str(t.invoke(args))
            except ToolError as e:
                result = f"ERROR: {e}"
            except Exception as e:  # bad arguments (validation) or an unexpected bug: report it to the agent
                result = f"ERROR: {name} failed: {type(e).__name__}: {str(e)[:500]}"
            if result.startswith(("ERROR", "BLOCKED")):
                self.req.failures[key] = self.req.failures.get(key, 0) + 1
                if self.req.failures[key] == 2:
                    result += ("\nThis call has now failed twice with the same arguments: do not retry it. "
                               "Stop and report the problem to the user.")
        self.req.tool_results[call_id] = result
        self.trace.observation(step, result)
        return result

    # ---------- the ReAct graph ----------
    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(today=f"{date.today():%Y-%m-%d (%A)}",
                                    courses="; ".join(c["name"] for c in self.courses))

    def _agent_node(self, state: MessagesState):
        messages = [SystemMessage(self._system_prompt())] + state["messages"]
        for attempt in (1, 2):
            self.budget.spend("agent")  # raises SessionLimit at the cap, before the call is made
            try:
                reply = self.model.invoke(messages)
                break
            except Exception as e:
                if attempt == 2 or study._permanent(e):
                    raise RequestStop(f"the model call failed: {type(e).__name__}: {str(e)[:300]}")
                print(f"      [retry] model call failed ({type(e).__name__}: {str(e)[:120]}); trying again in {study.RETRY_PAUSE}s...")
                time.sleep(study.RETRY_PAUSE)
        self.trace.llm_call(reply, thought_of(reply))
        # gpt-oss can leak "<|channel|>..." into tool names; fix known ones before dispatch and in the history
        for call_id, raw, _ in normalize_tool_calls(reply, self.tools_by_name):
            self.req.renamed[call_id] = raw
        return {"messages": [reply]}

    def _tools_node(self, state: MessagesState):
        last = state["messages"][-1]
        out = []
        for tc in last.tool_calls:
            out.append(ToolMessage(content=self.run_tool(tc["id"], tc["name"], tc.get("args") or {}),
                                   tool_call_id=tc["id"], name=tc["name"]))
        for bad in getattr(last, "invalid_tool_calls", None) or []:
            call_id = bad.get("id") or f"invalid_{uuid.uuid4().hex[:8]}"
            self.req.steps += 1
            if self.req.steps > self.max_steps:
                raise RequestStop(f"step limit reached: this request already used {self.max_steps} tool calls (--max-steps)")
            result = f"ERROR: could not parse the arguments of {bad.get('name')}: {str(bad.get('error'))[:200]}. Send valid JSON arguments."
            self.trace.action(self.req.steps, str(bad.get("name")), {"raw_args": str(bad.get("args"))[:200]})
            if bad.get("id") in self.req.renamed:
                self.trace.note(f"tool name normalized from '{self.req.renamed.pop(bad.get('id'))}'")
            self.trace.observation(self.req.steps, result)
            out.append(ToolMessage(content=result, tool_call_id=call_id, name=str(bad.get("name"))))
        return {"messages": out}

    @staticmethod
    def _route(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if last.tool_calls or getattr(last, "invalid_tool_calls", None) else END

    def _build_graph(self):
        g = StateGraph(MessagesState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._route, ["tools", END])
        g.add_edge("tools", "agent")
        return g.compile()

    # ---------- one request ----------
    def handle(self, request: str) -> None:
        self.req = _RequestState()
        self.trace = Trace(request, self)
        agent0, gen0 = self.budget.agent, self.budget.generation
        new: List[Any] = [HumanMessage(request)]
        stop = ""
        print(f"\n[trace: {self.trace.path.relative_to(BASE_DIR)}]")
        try:
            for chunk in self.graph.stream({"messages": self.history + new},
                                           {"recursion_limit": 2 * self.max_steps + 5}, stream_mode="updates"):
                for data in chunk.values():
                    new.extend((data or {}).get("messages", []))
        except SessionLimit as e:
            stop, self.over = f"Stopped: {e}.", True
        except RequestStop as e:
            stop = f"Stopped: {e}."
        except GraphRecursionError:
            stop = f"Stopped: step limit reached ({self.max_steps} tool calls, --max-steps)."
        except KeyboardInterrupt:
            stop = "Stopped: interrupted with Ctrl+C."
        except Exception as e:
            stop = f"Stopped: unexpected error: {type(e).__name__}: {str(e)[:300]}"

        # Keep the history valid for follow-ups: every tool call needs a tool result, and a stopped request ends
        # with a short assistant note so the model knows what happened.
        answered = {m.tool_call_id for m in new if isinstance(m, ToolMessage)}
        for m in list(new):
            for tc in getattr(m, "tool_calls", None) or []:
                if tc["id"] not in answered:
                    new.append(ToolMessage(content=self.req.tool_results.get(tc["id"], f"Not run: {stop or 'request stopped'}"),
                                           tool_call_id=tc["id"], name=tc["name"]))
        if stop:
            new.append(AIMessage(content=f"({stop} Files written before stopping: {', '.join(self.req.files) or 'none'}.)"))
            print(f"\n[!] {redact(stop)}")
            if self.req.files:
                print("    Files written before stopping: " + ", ".join(self.req.files))
            self.trace.event(stop)
        else:
            answer = text_of(new[-1]) if isinstance(new[-1], AIMessage) else ""
            if answer:
                self.trace.final(answer)
            else:
                meta = getattr(new[-1], "response_metadata", {}) or {}
                stop = f"The model returned an empty reply (finish_reason={meta.get('finish_reason')})."
                print(f"\n[!] {stop}")
                self.trace.event(stop)
        self.history += new
        self.requests_done += 1

        agent_calls, gen_calls = self.budget.agent - agent0, self.budget.generation - gen0
        summary = [
            f"Outcome: {stop or 'completed'}",
            f"Tool calls: {self.req.steps} (limit {self.max_steps})",
            f"LLM calls this request: {agent_calls} agent + {gen_calls} generation; "
            f"session total {self.budget.used} of {self.budget.cap}",
            "Files written: " + (", ".join(self.req.files) or "none"),
        ]
        self.trace.summary(summary)
        print(f"\n[{summary[1]} · {summary[2]} · trace saved to {self.trace.path.relative_to(BASE_DIR)}]")


# ============================================================
# OFFLINE: scripted fake agent model
# ============================================================
STOPWORDS = set("""a an the and or for on of about in from to with into my me i you please can could would get give
make write create generate prepare some all any every everything find show tell now do same then also key ideas idea
notes note revision revise practice questions question quiz mcq mcqs study slides slide ppt pdf course courses
this that these those based just only them it its""".split())


class ScriptedStudyLLM(BaseChatModel):
    """Offline stand-in for the agent model. Picks tool calls with simple rules, the way a real model would:
    search_items -> get_item_details -> generate_questions/notes -> final answer (or asks when several items
    match, or when an item has too little text). Runs through the real graph, limits, confirm prompt and trace."""

    model: str = "scripted-fake-agent (offline, no credits used)"

    @property
    def _llm_type(self) -> str:
        return "scripted-fake-agent"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    @staticmethod
    def _call(name, args, thought):
        return AIMessage(content="", additional_kwargs={"reasoning_content": thought},
                         tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:10]}", "type": "tool_call"}])

    @staticmethod
    def _final(text, thought):
        return AIMessage(content=text, additional_kwargs={"reasoning_content": thought})

    @staticmethod
    def _intent(text: str):
        t = text.lower()
        n = re.search(r"\b(\d{1,2})\s+(?:\w+\s+)?(?:questions|qs|mcqs?)\b", t)
        n = int(n.group(1)) if n else study.DEFAULT_QUESTIONS
        if re.search(r"\bdue\b|deadline|pending", t):
            return "due", n
        if re.search(r"question|quiz|mcq|viva", t):
            return "questions", n
        if re.search(r"notes|revision|revise|summar", t):
            return "notes", n
        return None, n

    @staticmethod
    def _keywords(text: str) -> str:
        m = re.search(r"\b(module|unit|lab|chapter|experiment|exp|ia)\s*[-#]?\s*(\d+)", text, re.I)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        words = [w for w in re.findall(r"[A-Za-z0-9*+#'-]+", text) if w.lower() not in STOPWORDS
                 and not w.isdigit() and w not in ("AI", "OS")]
        return " ".join(words[:4])

    def _next(self, messages):
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        h = humans[-1]
        request = text_of(messages[h])
        after = messages[h + 1:]
        tools_done = [m for m in after if isinstance(m, ToolMessage)]
        intent, n = self._intent(request)
        if intent is None and len(humans) > 1:  # follow-up ("now do the same for Module 1", "2", "yes"): reuse the last intent
            for i in reversed(humans[:-1]):
                intent, n2 = self._intent(text_of(messages[i]))
                if intent:
                    n = n2 if not re.search(r"\d+\s+questions", request.lower()) else n
                    break
        course = "AI" if re.search(r"\bAI\b", request) else "OS" if re.search(r"\bOS\b", request) else None

        if not tools_done:
            earlier = [m for m in messages[:h] if isinstance(m, ToolMessage)]
            choice = re.fullmatch(r"\s*(?:no\.?|number|#)?\s*(\d{1,2})\s*\.?\s*", request, re.I)
            if choice and earlier:
                ids = next((re.findall(r"^\d+\. id=(\S+)", m.content, re.M) for m in reversed(earlier)
                            if m.name == "search_items"), [])
                k = int(choice.group(1))
                if 1 <= k <= len(ids):
                    return self._call("get_item_details", {"item_id": ids[k - 1]},
                                      f"The user picked candidate {k} from my last search. I'll check that item's details.")
            asked = next((text_of(m) for m in reversed(messages[:h]) if isinstance(m, AIMessage)), "")
            if re.fullmatch(r"\s*(y|yes|ok|okay|sure|go ahead|do it)\b.*", request, re.I) and "Do you still want" in asked:
                last_id = next((re.search(r"^id=(\S+)", m.content).group(1) for m in reversed(earlier)
                                if m.name == "get_item_details" and re.search(r"^id=(\S+)", m.content)), None)
                if last_id and intent in ("notes", "questions"):
                    return self._gen(intent, last_id, n, "The user confirmed they still want it, so I'll generate.")
            if intent == "due":
                return self._call("list_pending_work", {"course": course} if course else {},
                                  "The user asks what is due, so I'll list upcoming due dates.")
            query = self._keywords(request)
            if not query:
                return self._final("What would you like to study? Name a module, topic or item.", "The request names no topic.")
            args = {"query": query, **({"course": course} if course else {})}
            return self._call("search_items", args, f"I need to find the item first. I'll search for '{query}'.")

        last = tools_done[-1]
        result = text_of(last)
        if result.startswith(("ERROR", "BLOCKED")):
            return self._final(f"I couldn't finish this: {result.splitlines()[0]}", "A tool failed, so I'll report it instead of retrying.")
        if last.name == "search_items":
            rows = re.findall(r"^(\d+)\. id=(\S+) \| (\w+) \| posted (\S+) \|.*?\| \"(.*?)\".* \| ([^|]*)$", result, re.M)
            exact = [r for r in rows if r[5].startswith("match in title")]
            if len(exact) == 1:  # one clear phrase match beats weaker keyword / file-text matches
                rows = exact
            if not rows:
                return self._final(f"I couldn't find anything matching that in your courses. ({result.splitlines()[0]})",
                                   "The search found nothing; I shouldn't guess an item.")
            if len(rows) > 1:
                cands = "\n".join(f"{k}. {label} ({kind}, posted {posted}, id {iid})" for k, iid, kind, posted, label, _ in rows)
                return self._final(f"I found {len(rows)} items that could match:\n{cands}\nWhich one do you mean? Reply with its number.",
                                   "Several items match and I can't tell which one the user means, so I'll ask.")
            return self._call("get_item_details", {"item_id": rows[0][1]},
                              "Exactly one item matches. I'll check its details and whether its files are readable.")
        if last.name == "get_item_details":
            iid = re.search(r"^id=(\S+)", result).group(1)
            chars = int(re.search(r"Readable course text: ([\d,]+)", result).group(1).replace(",", ""))
            if intent not in ("notes", "questions"):
                return self._final(f"Here is what I found:\n{result}", "The user only asked to find the item.")
            if chars < study.LITTLE_TEXT:
                return self._final(f"This item has only {chars} characters of readable text, so the {intent} would be "
                                   "mostly generic. Do you still want me to generate them? (yes/no)",
                                   "Too little readable text: I must ask before generating.")
            return self._gen(intent, iid, n, f"The item has {chars:,} characters of readable text, enough to generate {intent}.")
        if last.name in ("generate_notes", "generate_questions"):
            if result.startswith("User declined"):
                return self._final("OK, nothing was generated and no credits were used.", "The user declined; I must not retry.")
            paths = re.findall(r"^Wrote (\S+)", result, re.M)
            return self._final("Done. Saved:\n" + "\n".join(f"- {p}" for p in paths), "Generation finished; I'll report the file paths.")
        if last.name == "list_pending_work":
            due = [l for l in result.splitlines() if l.startswith("- ")]
            week = [l for l in due if "[this week]" in l or "[within 7 days]" in l]
            body = "\n".join(week) if week else "Nothing is due in the next 7 days." + (f" Next up:\n{due[0]}" if due else "")
            return self._final(f"{body}\n(From Classroom due dates; I can't check whether you already submitted.)",
                               "I'll summarise the due dates for this week.")
        return self._final(result, "Reporting the tool result.")

    def _gen(self, intent, iid, n, thought):
        if intent == "notes":
            return self._call("generate_notes", {"item_id": iid}, thought)
        return self._call("generate_questions", {"item_id": iid, "n": n}, thought)


# ============================================================
# CLI
# ============================================================
OFFLINE_EXAMPLES = [
    "Make 5 practice questions on A* search",
    "Make revision notes on search          (several items match: the agent asks which one)",
    "What's due this week in my AI course?",
    "now do the same for Module 1           (follow-up)",
]


def main(argv=None, get_llm=None) -> int:
    parser = argparse.ArgumentParser(prog="study_agent", description="Agentic study mode: type a request, an agent picks the tools.")
    parser.add_argument("--offline", "--dry-run", dest="offline", action="store_true",
                        help="Use study_sample.json and a scripted fake LLM: no Google or LLM calls, no credits used")
    parser.add_argument("--course", default="all", help="Only load courses matching this name (whole word; default: all)")
    parser.add_argument("--refresh", action="store_true", help="Ignore the local cache and re-download attachments")
    parser.add_argument("--chunk-chars", type=int, default=study.DEFAULT_CHUNK_CHARS,
                        help=f"Max characters of course text per generation call (default {study.DEFAULT_CHUNK_CHARS})")
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                        help=f"Max LLM calls in this session, agent reasoning included (default {DEFAULT_MAX_CALLS})")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max tool calls per request (default {DEFAULT_MAX_STEPS})")
    args = parser.parse_args(argv)
    if args.chunk_chars < 500:
        parser.error("--chunk-chars must be at least 500")
    if args.max_calls < 1 or args.max_steps < 1:
        parser.error("--max-calls and --max-steps must be at least 1")

    print("=" * 78)
    print("STUDY MODE (AGENT): ask in plain English; the agent picks the tools")
    print("=" * 78)

    # Check the LLM before signing in, so a missing key doesn't cost a browser login
    if args.offline:
        agent_llm, gen_llm = ScriptedStudyLLM(), study.FakeLLM()
    else:
        factory = get_llm
        if factory is None:
            from agent import get_llm as factory
        agent_llm = factory(timeout=study.LLM_TIMEOUT)
        if agent_llm is None:
            print("[!] No LLM API key in .env (NVIDIA_API_KEY, GROQ_API_KEY or OPENAI_API_KEY). Add one, or use --offline.")
            return 1
        gen_llm = factory(timeout=study.LLM_TIMEOUT)

    courses, drive, code = study.load_courses(args.offline, args.course)
    if code is not None:
        return code

    try:
        session = StudyAgentSession(courses, drive, agent_llm, gen_llm, args, args.offline)
    except Exception as e:
        print(f"[ERROR] Could not set up the agent with {study.model_name(agent_llm)} (it must support tool calling): "
              f"{type(e).__name__}: {redact(str(e))[:300]}")
        return 1

    print(f"\nLoaded {len(courses)} course(s), {len(session.items)} item(s). Agent model: {study.model_name(agent_llm)}")
    print(f"Limits: {args.max_steps} tool calls per request, {args.max_calls} LLM calls this session (agent reasoning included).")
    print("Notes and questions are only generated after you confirm the estimate with y.")
    print(f"Each request's trace is saved in {LOG_DIR.relative_to(BASE_DIR)}/.")
    if args.offline:
        print("\nTry:\n" + "\n".join(f"  {e}" for e in OFFLINE_EXAMPLES))
    try:
        while True:
            request = study.read_input("\nWhat do you want to study? (q to quit): ")
            if request.lower() in ("q", "quit", "exit"):
                return 0
            if not request:
                continue
            session.handle(request)
            if session.over:
                print(f"\nThe session limit of {args.max_calls} LLM calls is used up. "
                      "Start again (optionally with a higher --max-calls) to continue.")
                return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
