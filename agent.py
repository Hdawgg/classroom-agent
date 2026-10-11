"""
ClassroomAgent: Autonomous Google Classroom Assignment Discovery & Lab Solver
AI - IA 2 Group Activity (3 Students) | Course: CS401 (Dr. Ayesha Hakim)
Powered by NVIDIA Nemotron & LangChain ReAct
"""

import os, sys, json, subprocess, argparse, re, tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import google_classroom
from tool_names import normalize_tool_calls

# Emoji in the trace output would crash on Windows cp1252 consoles/pipes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
SUBMISSIONS_DIR = BASE_DIR / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

IS_OFFLINE = False
COURSE_FILTER = "AI"

# Shared by the agent system prompt and the report synthesizer so both ask for the same structure
REPORT_SECTIONS = (
    "   # POST-LAB REPORT: <Title>\n"
    f"   Header line: Course: <the assignment's course name>, Faculty: {os.getenv('FACULTY', '')}, "
    f"Student: {os.getenv('STUDENT_NAME', '')}, Roll No: {os.getenv('ROLL_NO', '')}, Batch: {os.getenv('BATCH', '')} "
    "(use these exact values; leave any empty value blank - never invent names or numbers)\n"
    "   ## 1. AIM & OBJECTIVES\n"
    "   ## 2. THEORETICAL BACKGROUND & MATHEMATICAL/LOGICAL CONCEPTS\n"
    "   ## 3. ALGORITHM STEP-BY-STEP\n"
    "   ## 4. VERIFIED PYTHON SOURCE CODE\n"
    "   ## 5. EXPERIMENTAL EXECUTION OUTPUT\n"
    "   ## 6. TIME & SPACE COMPLEXITY ANALYSIS\n"
    "   ## 7. POST-LAB VIVA VOCE QUESTIONS & DETAILED ANSWERS (at least 4 questions specific to this algorithm, numbered Q1, Q2, ...)\n"
    "   ## 8. CONCLUSION\n"
)


def submission_dir(assignment_title: str, assignment_id: str) -> Path:
    """Folder for an assignment's files, named after its title (falls back to its ID)."""
    clean_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', assignment_title or assignment_id or "").strip().replace(" ", "_")
    return SUBMISSIONS_DIR / (clean_name or f"assignment_{assignment_id}")

# ============================================================
# TOOL 1: Find Assignments Left to Submit from Google Classroom
# ============================================================
@tool
def find_pending_classroom_assignments() -> str:
    """Connects to Google Classroom, inspects all courses, and returns ONLY coursework left to submit."""
    pending = google_classroom.get_pending_assignments(force_offline=IS_OFFLINE, course_filter=COURSE_FILTER)
    if not pending:
        return "No pending assignments found. All coursework has been turned in!"
    
    summary = [f"Found {len(pending)} pending assignment(s) left to submit:"]
    for p in pending:
        summary.append(
            f"- [{p.get('id')}] {p.get('title')} ({p.get('course')})\n"
            f"  Due Date: {p.get('dueDate', 'N/A')} | State: {p.get('submissionState', p.get('state'))}\n"
            f"  Description: {p.get('description')}\n"
            f"  Attachment/Specs: {p.get('attachment', p.get('materials', 'None'))}\n"
        )
    return "\n".join(summary)

# ============================================================
# TOOL 2: Execute and Test Python Code in Sandbox
# ============================================================
@tool
def run_python_code(code: str) -> str:
    """Executes Python code in a separate subprocess to verify correctness and capture stdout/stderr.
    The code argument MUST be pure Python code only (no conversational text or questions)."""
    # NOTE: this is process isolation only, not a security sandbox - the code runs with the user's permissions.
    # Clean up markdown fences or stray text if passed by LLM
    clean = code
    if "```python" in clean:
        clean = clean.split("```python")[1].split("```")[0]
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0]
    clean = clean.strip()

    try:
        # Run from a throwaway directory via a script file (avoids Windows command-line length limits)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "solution.py"
            script.write_text(clean, encoding="utf-8")
            res = subprocess.run(
                [sys.executable, str(script)], cwd=tmp, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=15,
                # Don't hand API keys to model-written code
                env={**{k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}, "PYTHONIOENCODING": "utf-8"},
            )
        if res.returncode == 0:
            return f"Execution SUCCESS:\n{res.stdout}"
        return f"Execution FAILED:\n{res.stderr}"
    except Exception as e:
        return f"Sandbox Error: {str(e)}"

def generate_rich_postlab(assignment_title: str, code: str, output: str = "", brief_summary: str = "") -> str:
    """Uses LLM directly to synthesize a complete academic lab report with post-lab viva Q&A."""
    llm = get_llm()
    if not llm:
        return brief_summary

    prompt = (
        f"You are an expert AI teaching assistant for university Computer Science & AI.\n"
        f"Generate a rigorous, complete academic Post-Lab Report in clean Markdown for this lab submission:\n\n"
        f"Assignment: {assignment_title}\n"
        f"Verified Source Code:\n```python\n{code}\n```\n"
        f"Execution Output: {output or 'N/A'}\n"
        f"Execution Summary: {brief_summary}\n\n"
        f"If the course name is not given above, leave it blank.\n"
        f"The report MUST use exactly these sections, with theory and viva questions specific to this assignment's algorithm:\n"
        f"{REPORT_SECTIONS}\n"
        f"Format directly as full Markdown with clear section headers."
    )
    try:
        res = llm.invoke(prompt)
        text = res.content if hasattr(res, "content") else str(res)
        return text.strip()
    except Exception as e:
        print(f"[WARN] Rich report synthesis fallback: {e}")
        return brief_summary

# ============================================================
# TOOL 3: Save Verified Code and Post-Lab Report
# ============================================================
@tool
def save_lab_submission(assignment_id: str, code: str, postlab_content: str, assignment_title: str = "") -> str:
    """Saves the verified Python solution file and formatted post-lab report into a dedicated folder per assignment."""
    target_dir = submission_dir(assignment_title, assignment_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # If the LLM passed a brief summary or missed the official post-lab Q&A, synthesize the full academic report
    if len(postlab_content) < 400 or ("Q1" not in postlab_content and "Question" not in postlab_content):
        print(f"\n[INFO] Synthesizing comprehensive academic post-lab report with college Q&A...")
        rich_report = generate_rich_postlab(assignment_title or assignment_id, code, brief_summary=postlab_content)
        if rich_report and len(rich_report) > len(postlab_content):
            postlab_content = rich_report

    code_file = target_dir / "solution.py"
    report_file = target_dir / "postlab_report.md"

    with open(code_file, "w", encoding="utf-8") as f:
        f.write(code.strip())
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(postlab_content.strip())

    return f"Saved verified code to {code_file} and post-lab report to {report_file}"

# ============================================================
# LLM INITIALIZATION (NVIDIA NEMOTRON / GROQ / OPENAI)
# ============================================================
def get_llm(timeout: float = None):
    """`timeout` (seconds) overrides the request timeout; solve mode uses the defaults."""
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from requests import Session
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        # Default to a model the NVIDIA client knows supports tool calling
        model = os.getenv("NVIDIA_MODEL") or "openai/gpt-oss-20b"
        llm = ChatNVIDIA(
            model=model,
            api_key=nvidia_key,
            temperature=0,
            timeout=timeout or 60,
            max_completion_tokens=8192,
            model_kwargs={"reasoning_effort": "low"} if model == "openai/gpt-oss-20b" else {},
        )
        # ChatNVIDIA does not accept max_retries; configure its HTTP transport.
        # These are private client internals, so skip quietly if a library upgrade removes them.
        client = getattr(llm, "_client", None)
        if client is not None and hasattr(client, "get_session_fn"):
            def retry_session():
                session = Session()
                session.verify = getattr(client, "verify_ssl", True)
                session.mount("https://", HTTPAdapter(max_retries=Retry(
                    total=2, backoff_factor=1, allowed_methods={"GET", "POST"},
                    status_forcelist=[429, 500, 502, 503, 504],
                )))
                return session
            client.get_session_fn = retry_session
        return llm
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="llama-3.3-70b-versatile", api_key=groq_key, base_url="https://api.groq.com/openai/v1", temperature=0.1, timeout=timeout)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.1, timeout=timeout)

    return None

# ============================================================
# RE-ACT AGENT RUNNER (AUTONOMOUS WORKFLOW)
# ============================================================
DEMO_ASSIGNMENT_ID = "lab_04"
DEMO_ASSIGNMENT_TITLE = "Lab 04: A* Search Algorithm for Romania Route Finding"

def run_agent(goal: str, assignment: dict = None):
    print("=" * 68)
    print("🤖 CLASSROOM AGENT: AUTONOMOUS ASSIGNMENT FINDER & LAB SOLVER")
    print(f"Goal: {goal}")
    print("=" * 68)

    llm = get_llm()

    if not llm:
        # The scripted demo only has a solution for Lab 04; don't pass it off as a solution to anything else
        if assignment and assignment.get("id") != DEMO_ASSIGNMENT_ID:
            print(f"\n⏩ [SKIP]: No LLM API key in .env - the offline demo can only solve '{DEMO_ASSIGNMENT_TITLE}'.")
            return
        print("\n[INFO] No LLM API key in .env. Running scripted demo of the ReAct workflow:\n")
        print("🤔 [THOUGHT]: User wants me to find all pending assignments left to submit in Google Classroom and solve them.")
        obs1 = find_pending_classroom_assignments.invoke({})
        print("⚡ [ACTION]: find_pending_classroom_assignments()")
        print(f"👁️  [OBSERVATION]:\n{obs1}\n")

        print("🤔 [THOUGHT]: Google Classroom returned Lab 04 (A* Search) as pending. I will implement the A* algorithm, test it in the sandbox, and verify the path.")
        a_star_code = """import heapq
graph = {'Arad': [('Zerind', 75), ('Sibiu', 140), ('Timisoara', 118)], 'Zerind': [('Arad', 75), ('Oradea', 71)], 'Oradea': [('Zerind', 71), ('Sibiu', 151)], 'Sibiu': [('Arad', 140), ('Oradea', 151), ('Fagaras', 99), ('Rimnicu Vilcea', 80)], 'Timisoara': [('Arad', 118), ('Lugoj', 111)], 'Lugoj': [('Timisoara', 111), ('Mehadia', 70)], 'Mehadia': [('Lugoj', 70), ('Drobeta', 75)], 'Drobeta': [('Mehadia', 75), ('Craiova', 120)], 'Craiova': [('Drobeta', 120), ('Pitesti', 138), ('Rimnicu Vilcea', 146)], 'Rimnicu Vilcea': [('Sibiu', 80), ('Craiova', 146), ('Pitesti', 97)], 'Fagaras': [('Sibiu', 99), ('Bucharest', 211)], 'Pitesti': [('Rimnicu Vilcea', 97), ('Craiova', 138), ('Bucharest', 101)], 'Bucharest': [('Fagaras', 211), ('Pitesti', 101)]}
h = {'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242, 'Fagaras': 176, 'Lugoj': 244, 'Mehadia': 241, 'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193, 'Sibiu': 253, 'Timisoara': 329, 'Zerind': 374}

def a_star(start, goal):
    q = [(h[start], 0, start, [start])]
    visited = {start: 0}
    while q:
        f, g, cur, path = heapq.heappop(q)
        if cur == goal: return path, g
        for nxt, cost in graph.get(cur, []):
            ng = g + cost
            if nxt not in visited or ng < visited[nxt]:
                visited[nxt] = ng
                heapq.heappush(q, (ng + h.get(nxt, 0), ng, nxt, path + [nxt]))
path, cost = a_star('Arad', 'Bucharest')
print(f'Optimal Path: {" -> ".join(path)}')
print(f'Total Cost: {cost}')"""

        print("⚡ [ACTION]: run_python_code(code='...')")
        obs2 = run_python_code.invoke({"code": a_star_code})
        print(f"👁️  [OBSERVATION]:\n{obs2}\n")

        print("🤔 [THOUGHT]: Code verified with cost 418. Now saving verified code and synthesizing post-lab report.")
        postlab_md = f"""# POST-LAB REPORT: LAB 04 - A* SEARCH ALGORITHM
**Course:** CS401 Artificial Intelligence | **Faculty:** Dr. Ayesha Hakim
**Aim:** Find shortest path from Arad to Bucharest using A* Search with straight-line heuristic.
**Algorithm:** Evaluation function f(n) = g(n) + h(n).
**Verified Output:** Path: Arad -> Sibiu -> Rimnicu Vilcea -> Pitesti -> Bucharest | Cost: 418
**Complexity:** Time: O(b^d), Space: O(b^d)
**Viva Q&A:**
- Q1: When is A* guaranteed optimal? A1: When heuristic h(n) is admissible (and consistent for graph search).
- Q2: What if h(n) = 0? A2: Degrades to Uniform Cost Search (Dijkstra's algorithm)."""

        obs3 = save_lab_submission.invoke({
            "assignment_id": DEMO_ASSIGNMENT_ID, "assignment_title": DEMO_ASSIGNMENT_TITLE,
            "code": a_star_code, "postlab_content": postlab_md,
        })
        print(f"⚡ [ACTION]: save_lab_submission(...)")
        print(f"👁️  [OBSERVATION]: {obs3}\n")
        if not obs3.startswith("Saved"):
            print("❌ [ERROR]: The submission was not saved.")
            return

        print("🎯 [FINAL ANSWER]:\nAutonomous workflow complete! Discovered pending Lab 04 in Google Classroom, implemented & verified code in sandbox, and generated submission files in 'submissions/' directory.")
        return

    # Live ReAct Agent with NVIDIA Nemotron
    system_prompt = (
        "You are ClassroomAgent, an autonomous AI academic assistant for engineering students.\n"
        "Your goal is to discover pending lab assignments from Google Classroom, write verified Python code, test it in the sandbox, and compile a comprehensive college Post-Lab Report.\n\n"
        "Execution Steps:\n"
        "1. Call find_pending_classroom_assignments() to get coursework left to submit.\n"
        "2. Choose an unsubmitted lab assignment (or the one requested in the user prompt).\n"
        "3. Write a self-contained, working Python solution and test it with run_python_code(code=...). The code must be pure Python only.\n"
        "   Verify computed results with assertions, not just a successful exit code - e.g. cross-check the result against an independent brute-force or exhaustive computation.\n"
        "4. Once code execution succeeds with exit code 0, call save_lab_submission(\n"
        "     assignment_id=...,      # the exact ID from Google Classroom\n"
        "     assignment_title=...,   # the exact title from Google Classroom\n"
        "     code=...,\n"
        "     postlab_content=...\n"
        "   )\n"
        "   CRITICAL REQUIREMENT FOR `postlab_content`:\n"
        "   It MUST be a full, rigorous Markdown academic document with these sections:\n"
        f"{REPORT_SECTIONS}\n"
        "5. Finish with a clear summary of the completed submission."
    )
    tools = [find_pending_classroom_assignments, run_python_code, save_lab_submission]
    agent = create_react_agent(_solve_model(llm, tools), tools, prompt=system_prompt)

    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", "LLM")
    print(f"\n[INFO] Connected to {model_name}. Starting ReAct loop...\n")
    print("⏳ [THINKING]: The model is formulating the strategy...")
    history = [("user", goal)]
    # The model occasionally stops after testing without saving; nudge it once to finish the job
    for attempt in range(2):
        tested, saved, save_tried, reply = _stream_agent(agent, history)
        # A custom goal may not involve saving at all: then the reply is just the answer
        if saved or not (tested or save_tried or assignment or _mentions_saving(reply)):
            if reply:
                print(f"\n🎯 [AGENT RESPONSE]:\n{reply}")
            return
        # Never show a summary that may claim a save that did not happen
        if reply:
            print("\n⚠️  [WARN]: The agent's final reply is not shown: save_lab_submission did not return success, "
                  "so nothing was saved.")
        # Only nudge when it got as far as verified code
        if attempt == 0 and tested:
            print("\n⚠️  [WARN]: The agent stopped before saving the submission. Asking it to continue...")
            history.append(("user", "You stopped before saving. Continue the task: if the code is verified, "
                                    "call save_lab_submission now with the code and the full post-lab report."))
        else:
            break
    print("\n❌ [ERROR]: The agent finished without saving a submission.")


def _mentions_saving(text: str) -> bool:
    return bool(re.search(r"\bsav(e|ed|ing)\b|save_lab_submission", text or "", re.IGNORECASE))


MAX_INVALID_RETRIES = 2  # per tool and run: replies with unparseable tool arguments sent back for another try


def _parse_error(bad: dict) -> str:
    """Why an invalid tool call could not be used (langchain leaves `error` empty when the JSON is bad)."""
    if bad.get("error"):
        return str(bad["error"])[:300]
    try:
        json.loads(bad.get("args") or "")
    except (TypeError, ValueError) as e:
        return f"the arguments are not valid JSON ({e})"[:300]
    return "the arguments are not a JSON object"


def _solve_model(llm, tools):
    """The model for create_react_agent, given as a dynamic model so the graph itself is unchanged.
    It binds the tools exactly as create_react_agent would, then for each reply:
    - fixes tool names where gpt-oss leaked "<|channel|>..." into them;
    - if the reply has invalid_tool_calls (arguments that are not valid JSON, e.g. the whole code and report
      in one call), sends the parse error back to the model as a tool message and asks again, at most
      MAX_INVALID_RETRIES times per tool in this run. The graph only sees the final reply; if the retries
      run out, the invalid calls are left out so nothing pretends they ran."""
    bound = llm.bind_tools(tools)
    names = [t.name for t in tools]
    retries = {}

    def call(messages, config):
        messages = messages.to_messages() if hasattr(messages, "to_messages") else list(messages)
        while True:
            reply = bound.invoke(messages, config)
            for _, raw, _ in normalize_tool_calls(reply, names):
                print(f"\n📝 [NOTE]: tool name normalized from '{raw}'")
            bad = reply.invalid_tool_calls
            if not bad:
                return reply
            cut_off = (getattr(reply, "response_metadata", {}) or {}).get("finish_reason") == "length"
            errors = {}
            for b in bad:
                errors[b.get("id")] = _parse_error(b) + (" Your reply hit the output token limit, so the arguments were cut off."
                                                         if cut_off else "")
                print(f"\n⚠️  [INVALID TOOL CALL]: {b.get('name')}: {errors[b.get('id')]}")
            bad_names = {str(b.get("name")) for b in bad}
            if any(retries.get(n, 0) >= MAX_INVALID_RETRIES for n in bad_names):
                print(f"❌ [ERROR]: Still invalid after {MAX_INVALID_RETRIES} retries; giving up on {', '.join(sorted(bad_names))}.")
                valid_ids = {tc["id"] for tc in reply.tool_calls}
                kwargs = dict(reply.additional_kwargs)
                kwargs["tool_calls"] = [tc for tc in kwargs.get("tool_calls", []) if tc.get("id") in valid_ids]
                if not kwargs["tool_calls"]:
                    del kwargs["tool_calls"]  # this is what is sent back to the API: no dangling calls
                return reply.model_copy(update={"additional_kwargs": kwargs})
            for n in bad_names:
                retries[n] = retries.get(n, 0) + 1
            print(f"🔁 [RETRY {max(retries[n] for n in bad_names)}/{MAX_INVALID_RETRIES}]: "
                  "sending the parse error back to the model...")
            feedback = [ToolMessage(content=f"Error: {b.get('name')} was NOT run (nothing was saved or executed): "
                                            f"{errors[b.get('id')]} Call {b.get('name')} again with complete, valid JSON "
                                            "arguments (escape quotes and newlines inside strings).",
                                    tool_call_id=b.get("id") or "invalid", name=str(b.get("name"))) for b in bad]
            feedback += [ToolMessage(content=f"Not run: another tool call in the same reply was invalid. "
                                             f"Call {tc['name']} again if it is still needed.",
                                     tool_call_id=tc["id"], name=tc["name"]) for tc in reply.tool_calls]
            messages = messages + [reply] + feedback

    return lambda state, runtime: RunnableLambda(call)


def _stream_agent(agent, history: list):
    """Runs the ReAct loop over `history` (extended in place), printing the trace. Returns (tested, saved,
    save_tried, reply): whether code ran successfully, whether save_lab_submission returned success, whether it
    was called at all, and the final reply's text (not printed here: run_agent decides whether to show it)."""
    tested = saved = save_tried = False
    reply = ""
    # Allow room for a few write/test/fix cycles before LangGraph aborts the loop
    for chunk in agent.stream({"messages": list(history)}, {"recursion_limit": 50}, stream_mode="updates"):
        for node, data in chunk.items():
            for msg in data.get("messages", []):
                history.append(msg)
                if msg.type == "tool" and msg.name == "run_python_code" and msg.content.startswith("Execution SUCCESS"):
                    tested = True
                if msg.type == "tool" and msg.name == "save_lab_submission" and msg.content.startswith("Saved"):
                    saved = True
                if msg.type == "ai" and any(tc.get("name") == "save_lab_submission" for tc in
                                            list(msg.tool_calls) + list(getattr(msg, "invalid_tool_calls", None) or [])):
                    save_tried = True
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n⚡ [ACTION]: {tc['name']}({tc['args']})")
                elif msg.type == "tool":
                    print(f"👁️  [OBSERVATION]: {msg.content[:400]}...\n")
                elif msg.type == "ai" and msg.content:
                    reply = msg.content
                elif msg.type == "ai":
                    # An empty reply ends the loop; surface why instead of exiting silently
                    meta = getattr(msg, "response_metadata", {}) or {}
                    invalid = [tc.get("name") for tc in getattr(msg, "invalid_tool_calls", None) or []]
                    print(f"\n⚠️  [WARN]: Model returned an empty reply (finish_reason={meta.get('finish_reason')}, "
                          f"usage={getattr(msg, 'usage_metadata', None)}, invalid_tool_calls={invalid or None})")
    return tested, saved, save_tried, reply

def is_already_solved(assignment_title: str, assignment_id: str) -> bool:
    """Checks if verified code and comprehensive post-lab report already exist for this assignment
    (in a folder named after either its title or its ID)."""
    for target_dir in {submission_dir(assignment_title, assignment_id), submission_dir("", assignment_id)}:
        code_file = target_dir / "solution.py"
        report_file = target_dir / "postlab_report.md"
        if code_file.exists() and report_file.exists():
            try:
                content = report_file.read_text(encoding="utf-8")
                if len(content) > 400 and ("Q1" in content or "Question" in content or "AIM" in content):
                    return True
            except Exception:
                pass
    return False

def run_all_pending(force: bool = False, course_filter: str = "AI", only_id: str = None):
    """Iterates through all pending assignments in Google Classroom (or just `only_id`), solving each one once."""
    print("=" * 68)
    print("🤖 CLASSROOM AGENT: BATCH AUTONOMOUS LAB SOLVER (ALL PENDING)")
    print(f"Target Subject Filter: {course_filter}")
    print("=" * 68)

    pending = google_classroom.get_pending_assignments(force_offline=IS_OFFLINE, course_filter=course_filter)
    if only_id:
        pending = [p for p in pending if str(p.get("id")) == only_id]
        if not pending:
            print(f"\n[INFO] No pending assignment with ID {only_id} matching '{course_filter}'.")
            return
    if not pending:
        print("\n[INFO] No pending assignments found! All coursework is turned in.")
        return

    print(f"\n[INFO] Found {len(pending)} pending assignment(s) matching '{course_filter}'.")
    for idx, item in enumerate(pending, 1):
        title = item.get("title", f"Assignment_{item.get('id')}").strip()
        course = item.get("course", "College Course")
        a_id = item.get("id")
        desc = item.get("description", "")

        print(f"\n{'=' * 68}")
        print(f"[{idx}/{len(pending)}] Assignment: {title} ({course})")

        # Skip administrative announcements or non-coding submissions
        title_lower = title.lower()
        if not any(k in title_lower for k in ["exp", "lab", "search", "algorithm", "problem", "solve", "system"]):
            print(f"⏩ [SKIP]: Non-coding announcement or report. Skipping.")
            continue

        if not force and is_already_solved(title, a_id):
            print(f"⏩ [SKIP]: Already solved and verified in 'submissions/'. Skipping to avoid duplicate work.")
            continue

        print(f"🚀 [SOLVING]: Launching autonomous agent ReAct solver for this assignment...")
        specs = item.get("attachment") or item.get("materials") or "None"
        goal = (
            f"Solve the pending assignment '{title}' for course '{course}' (ID: {a_id}).\n"
            f"Assignment Description:\n{desc}\n"
            f"Attachment/Specs:\n{specs}\n"
            f"Write the complete Python solution, test it in the sandbox using run_python_code, "
            f"and save it with save_lab_submission(assignment_id='{a_id}', assignment_title='{title}', ...) "
            f"including the complete academic post-lab report."
        )
        try:
            run_agent(goal, assignment=item)
        except Exception as e:
            print(f"\n❌ [ERROR] Could not complete '{title}': {e}. Skipping to next assignment...\n")

if __name__ == "__main__":
    # Study mode has its own options (see `python study.py --help`); solve mode below is unchanged
    if "--study" in sys.argv[1:]:
        import study
        sys.exit(study.main([a for a in sys.argv[1:] if a != "--study"], get_llm=get_llm))

    parser = argparse.ArgumentParser(description="ClassroomAgent CLI")
    parser.add_argument("--study", action="store_true", help="Study mode: revision notes / practice questions (options: python study.py --help)")
    parser.add_argument("--offline", action="store_true", help="Force offline/local classroom mode without Google OAuth")
    parser.add_argument("--all", action="store_true", help="Iteratively solve all pending assignments (skipping already solved ones)")
    parser.add_argument("--force", action="store_true", help="Re-solve assignments even if already saved in submissions/")
    parser.add_argument("--course", default="AI", help="Filter coursework by course name (default: 'AI', use 'all' for all)")
    parser.add_argument("--interval", type=int, default=0, help="Run continuously in a background loop every N seconds (e.g. 3600)")
    parser.add_argument("--id", help="Solve only the pending assignment with this Classroom ID")
    parser.add_argument("goal", nargs="*", help="Custom goal for the agent")
    args = parser.parse_args()

    IS_OFFLINE = args.offline
    COURSE_FILTER = args.course

    import time
    while True:
        if args.all or args.id or not args.goal:
            run_all_pending(force=args.force, course_filter=args.course, only_id=args.id)
        else:
            user_goal = " ".join(args.goal)
            run_agent(user_goal)

        if args.interval > 0:
            print(f"\n[CRON/LOOP] Sleeping for {args.interval} seconds before next check. Press Ctrl+C to stop...")
            time.sleep(args.interval)
        else:
            break

