# ClassroomAgent: Autonomous Google Classroom Lab Solver & Post-Lab Synthesizer

> **AI - IA 2 Project (20 Marks) | Group Activity**  
> **Course:** Artificial Intelligence (CS401)  
> **Instructor:** Dr. Ayesha Hakim  

ClassroomAgent is an autonomous **Agentic AI system** built with **LangChain / LangGraph** and the **ReAct framework**. It connects to Google Classroom, automatically scans all active courses, finds coursework **left to submit** (`state != 'TURNED_IN'`), writes and tests Python solutions in an isolated sandbox, and saves the verified code and post-lab report directly into `submissions/`.

---

## 👥 Team Collaboration Matrix (3 Students)

| Student | Role | Technical Focus |
|---|---|---|
| **Student 1** | **Agent Architect** | Prompt design, LLM integration (NVIDIA / Groq / OpenAI), LangGraph ReAct reasoning loop. |
| **Student 2** | **Classroom & Sandbox Engineer** | Google Classroom OAuth 2.0 integration & safe Python code sandbox. |
| **Student 3** | **Evaluation & Documentation Lead** | Trace logging, post-lab report synthesis, Classical vs Modern comparison. |

---

## ⚡ Autonomous Workflow

```
[User Request: "Access Google Classroom, find pending labs, and solve them"]
    │
    ▼
[Orchestrator Agent (ReAct Loop)]
    │
    ├── 1. Thought: Query Google Classroom for unsubmitted assignments
    ├── 2. Action: find_pending_classroom_assignments()
    ├── 3. Observation: Discovered pending Lab 04 (A* Search)
    │
    ├── 4. Thought: Formulate code, execute in sandbox to verify output
    ├── 5. Action: run_python_code(code=...)
    ├── 6. Observation: Optimal Path Arad->Bucharest | Cost: 418
    │
    ├── 7. Action: save_lab_submission(...)
    └── 8. Observation: Saved code and post-lab markdown to submissions/
    │
    ▼
[Final Output]: Fully Solved Lab & Submission Ready in `submissions/`
```

**Safeguards in the loop:**
- **Unparseable tool calls are retried.** Sometimes the model sends a tool call whose arguments aren't valid JSON, for example the whole code and report in one `save_lab_submission` call that gets cut off. When that happens, nothing is run. The parse error goes back to the model as a tool message (`[INVALID TOOL CALL]` / `[RETRY n/2]`) and it tries again, at most 2 retries per tool in a run.
- **The final summary is shown only after a real save.** It's printed only if `save_lab_submission` actually returned "Saved…". Otherwise the reply is held back and the run ends with `❌ [ERROR]: The agent finished without saving a submission.`
- **Leaked tool-name tokens are fixed.** Names with a leaked `<|channel|>…` suffix are cleaned up (see *Agentic study mode*).

---

## 🔑 Google Classroom Setup

### Option A: Real Google Classroom OAuth 2.0
1. Go to [Google Cloud Console](https://console.cloud.google.com/) and enable the **Google Classroom API**.
2. Create an **OAuth 2.0 Client ID** (Desktop Application) and download the credentials file.
3. Save it as `credentials.json` in the project folder (next to `agent.py`).
4. On first run, it will open your browser to authorize your college account and create `token.json`.

If the live connection fails, the agent prints a warning and falls back to the sample data in Option B.

### Option B: Offline / Simulated Classroom Mode (Default Fallback)
If `credentials.json` is not provided (or `--offline` is passed), the agent reads [classroom_assignments.json](classroom_assignments.json) and detects pending labs (`submissionState: "CREATED"` vs `"TURNED_IN"`). This allows testing and grading without needing external cloud credentials!

---

## 🚀 Quickstart (Windows PowerShell)

```powershell
# 1. Create the environment and install dependencies (first time only)
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure: copy .env.example to .env and add one LLM API key + your name/roll no/batch
Copy-Item .env.example .env

# 3. Run the agent on all pending assignments (sample data)
python agent.py --offline

# 4. Check generated submissions
Get-ChildItem -Recurse submissions
```

On macOS/Linux use `python3 -m venv venv` and `source venv/bin/activate` instead.

### CLI options

| Option | Effect |
|---|---|
| `--offline` | Use `classroom_assignments.json` instead of Google Classroom |
| `--all` | Solve every pending assignment (default when no goal is given) |
| `--force` | Re-solve assignments already saved in `submissions/` |
| `--id ID` | Solve only the pending assignment with this Classroom ID |
| `--course NAME` | Course name filter, whole-word match (default `AI`; `all` for every course) |
| `--interval N` | Repeat every N seconds |
| `"goal text"` | Run the agent once with a custom goal |

Without any LLM API key, the agent runs a scripted demo that can only solve Lab 04.

---

## 📚 Study Mode (revision notes & practice questions)

A terminal study assistant that is separate from the solver above. It signs in (read-only) and loads all your active courses with their assignments, materials and announcements. It reads the attached Drive files, and for the items you pick it writes revision notes and/or practice questions with answers. The menu makes plain LLM calls (same model settings as the solver, no agent loop); an optional agentic mode (`--agent`, see below) lets you ask in plain English instead. Neither ever submits or changes anything in Classroom.

### One-time setup
1. Install the new packages: `pip install -r requirements.txt` (adds `pypdf`, `python-docx`, `python-pptx`).
2. In [Google Cloud Console](https://console.cloud.google.com/), in the same project as your `credentials.json`, enable the **Google Drive API** (APIs & Services → Library). The Classroom API is already enabled.
3. If your OAuth consent screen is in *Testing* mode, make sure your college account is listed as a test user. Google may warn that the app is unverified, because `drive.readonly` is a restricted scope. Click *Continue* (it's your own app).
4. **Sign in again.** Study mode needs these read-only permissions, which your existing `token.json` doesn't have: `classroom.courses.readonly`, `classroom.coursework.me.readonly`, `classroom.courseworkmaterials.readonly`, `classroom.announcements.readonly`, `classroom.topics.readonly` (for topic names) and `drive.readonly`. If `token.json` is missing any of them, study mode stops and lists them. Then:
   ```powershell
   Remove-Item token.json
   python agent.py --study      # opens the browser; tick every box on the consent screen
   ```
   This sign-in also includes the solver's permissions, so one `token.json` works for both modes. (If you sign in through the solver first, its token won't cover study mode.)

### Usage
```powershell
python agent.py --study                 # or: python study.py
python agent.py --study --offline       # sample courses + fake LLM: no Google or LLM calls, no credits used
python agent.py --study --course AI     # only courses whose name matches
python agent.py --study --agent         # agentic mode: ask in plain English (see "Agentic study mode")
```

1. **Pick a course.** Its Drive attachments are read the first time you open it.
2. **Pick items.** Items are grouped by topic, newest first. Each row shows whether the item's files could be read, its posted date, its type, a label and its attachments:
   ```
     6. [2/3] 2026-09-11  Announcement  Submission rules for IA1 report    (3 files)
   ```
   `[OK ]` all read · `[1/2]` some read · `[ X ]` none read · `[ - ]` no Drive files.

   | Command | Does |
   |---|---|
   | `3`, `1,4,6-8` | select items by number |
   | `t2` / `all` | select a whole topic / every item shown |
   | `d3` | details: posted date, the message (first 600 characters), each attachment's type, size, and read status or why it was skipped |
   | `/word` | show only items whose title, text or attachment names contain *word*. Punctuation is ignored, so `/IA2` also finds "IA-2". Items that match only inside a file's text are flagged. `/` alone clears the search. |
   | `n` | show only items with at least 200 characters of readable text (hides empty and unreadable items); `n` again shows all |
   | `b` / `q` | back to courses / quit |
3. **Choose what to generate:** `a` revision notes, `b` practice questions (conceptual, viva, predict the output / next step), or `c` both. Then choose how many questions per item.
4. **Confirm.** Before any LLM call you'll see the items, the characters of text to be sent, and the number of calls. It also lists every existing file that would be overwritten. Nothing is sent unless you answer `y`.

### Announcement labels
Announcements have no title in Classroom, and most start with "Dear All,". Their labels are built for free (no LLM):
1. the first meaningful line, skipping greetings ("Dear All", "Dear Students", "Hi", "Hello", "Good morning", "Greetings"), filler ("PFA", "Please find attached", "Regards", bare links) and lines under 12 characters. Long lines are cut at the first sentence end, colon or bracket, or else at about 60 characters;
2. otherwise the attachment names, e.g. `Files: AI_Module1.pdf, PEAS.png (+2)`;
3. otherwise `(empty announcement)`.

`--ai-labels` (optional) asks the LLM for a 5-8 word label per announcement instead. It's one cheap call per course, shown with its size and only sent after you confirm. Labels are cached in `.study_cache/labels/`, so an announcement is only sent again if its text changes. Assignments and materials keep their Classroom titles. Without the flag, the menu makes no LLM calls.

### Output
```
study/<course>/<posted date>_<label>/notes.md       revision notes
study/<course>/<posted date>_<label>/questions.md   questions first; answers + short explanations in a separate
                                                    "Answers" section at the bottom, each collapsed (<details>)
```
Open them in VS Code's Markdown preview (`Ctrl+Shift+V`) to keep answers hidden until you click them. Folder names use the free label, so they don't change with `--ai-labels`. Two items that would still get the same name have their Classroom ID appended. Folders created by the first version of study mode (such as `Dear_All_826704`) are left untouched. The estimate and `d<N>` point out when an item already has output in one.

**Long items and failures:** each LLM call has a 180-second timeout and is retried up to 3 times (5, 10, then 15 seconds apart). Errors that can't be fixed by retrying, such as a bad API key, aren't retried. Each chunk's reply is saved to `<item folder>/.parts/` as soon as it arrives. If a run fails, run the same selection again: the estimate shows `resuming: N saved part(s) reused`, and only the missing calls are sent. If only the final merge of the notes fails, `notes.md` is still written from the parts. The `.parts` folder is deleted once the file is complete.

### What gets read
| Attachment | Handling |
|---|---|
| Google Docs, Google Slides | exported as text |
| PDF, Word `.docx`, PowerPoint `.pptx` (incl. speaker notes), plain-text/code files | downloaded and text extracted |
| Scanned PDFs, Sheets, old `.doc`/`.ppt`, files over 25 MB, files whose owner disabled downloads | skipped; the reason is shown under `d<N>` and in the generated file |
| YouTube videos, links, Forms | not fetched; only their titles are passed to the model |

Extracted text is cached in `.study_cache/` and reused until the file changes on Drive. Use `--refresh` to re-download everything.

### Study mode options
| Option | Effect |
|---|---|
| `--offline` / `--dry-run` | Use [study_sample.json](study_sample.json) and a fake LLM. Tests the menus and file saving without Google or NVIDIA. |
| `--course NAME` | Only courses matching NAME (whole word; default: all) |
| `--refresh` | Ignore the local cache and re-download attachments |
| `--ai-labels` | Label announcements with the LLM (asks first; cached) |
| `--chunk-chars N` | Max characters of course text per LLM call (default 20000, or `STUDY_CHUNK_CHARS` in `.env`). Longer items are split; notes from several chunks are merged with one extra call, and questions are shared out across chunks. |

`study/`, `.study_cache/`, `token.json`, `credentials.json` and `.env` are all in `.gitignore`.

### Agentic study mode (`--agent`)
Instead of the menu, type what you want in plain English. A **LangGraph ReAct agent** (an explicit `StateGraph` with an `agent` node and a `tools` node, in [study_agent.py](study_agent.py)) decides which tools to call. The menu above is unchanged and is still the default.

```powershell
python agent.py --study --agent             # or: python study.py --agent / python study_agent.py
python agent.py --study --agent --offline   # sample data + scripted fake LLM: no Google or LLM calls
```

Sign-in, course loading and the re-auth message are the same as the menu's. Then:
```
What do you want to study? (q to quit): Get my AI Module 5 notes and make 10 practice questions
  Thought: I need to find the item first...
  [1] Action: search_items(query='Module 5', course='AI')
      Observation: 1 item(s) match 'Module 5' in Artificial Intelligence ...
  [2] Action: get_item_details(item_id='...')
  [3] Action: generate_questions(item_id='...', n=10)
      (estimate)  Send to the LLM? [y/N]: y
Answer: Done. Saved: study\...\questions.md
```
The conversation is kept for the session, so follow-ups work ("now do the same for Module 1", or "2" to pick from a list the agent showed you). Other examples: "Make revision notes for the Module 2 slides", "What's due this week in my AI courses?", "Find everything about expert systems and quiz me on the key ideas".

| Tool | Does | LLM credits |
|---|---|---|
| `list_courses` | course names and item counts | no |
| `search_items(query, course)` | the `/word` search over titles, messages and attachment names (matches only inside file text are flagged); numbered results with id, kind, posted date and readability. With 5 matches or fewer it opens their Drive files (read-only, cached) to report readability | no |
| `get_item_details(item_id)` | the `d<N>` details view as text, plus the readable text size | no |
| `read_item_text(item_id, max_chars)` | a short preview (800 chars by default, 2000 at most) and the total length; the full text never goes into the conversation | no |
| `generate_notes(item_id)`, `generate_questions(item_id, n)` | the menu's pipeline: same estimate, chunking, retries, resumable `.parts`, same output folders | **yes, after you type `y`** |
| `list_pending_work(course)` | due dates from the item metadata, soonest first, flagged *this week* / *within 7 days*; submission status is not checked | no |

**Credit protection (enforced in code, not just in the prompt):**
- `generate_*` prints the usual estimate and waits for `y`. If you answer anything else, nothing is sent and the tool returns "User declined". Every further `generate_*` call in that request is blocked without asking you.
- **Session cap on LLM calls:** `--max-calls N` (default 30). It counts the agent's own reasoning calls as well as generation calls and their retries. A generate call that wouldn't fit in what's left is refused before the estimate. When the cap is reached the session stops and says so.
- **Step limit:** `--max-steps N` tool calls per request (default 15). The request stops with a message, and the session carries on.
- A tool call that fails twice with the same arguments isn't run again. The agent is told to stop and report the problem.
- Every model call is counted. One failed agent call is retried once (both attempts count). Errors such as a bad API key are not retried.

**Trace logs:** each request is printed live (Thought / Action / Observation) and saved to `study/agent_logs/<timestamp>.md`. The log holds the request, every agent LLM call with its visible reasoning, each tool's arguments and output (long output truncated), the estimate you were shown and your y/N answer, the final answer, and a summary (outcome, tool calls, LLM calls, files written). API keys and OAuth tokens are redacted from the console trace and the logs.

**Model:** the same `get_llm()` as solve mode, with study mode's 180 s timeout. The model must support tool calling; the default `openai/gpt-oss-20b` does. `--offline` uses a scripted fake agent model that calls the tools the way a real model would (search, then details, then generate, then answer; it asks when several items match or an item has too little text).

**Leaked channel tokens in tool names:** gpt-oss sometimes sends a tool name with part of its internal "harmony" format attached, e.g. `generate_questions<|channel|>commentary`. Before a tool is run, everything from `<|` onward is cut off and whitespace is stripped ([tool_names.py](tool_names.py)). If the result is a known tool, it runs normally (counted once), and the trace gets `note: tool name normalized from '<raw>'`. Otherwise the usual "unknown tool" error is returned. The cleaned name is also what goes back into the conversation history. Solve mode uses the same fix and prints a `[NOTE]` line when it applies.

| Option | Effect |
|---|---|
| `--offline` / `--dry-run` | sample data, scripted fake agent + fake generation LLM |
| `--max-calls N` | LLM calls allowed this session, agent reasoning included (default 30) |
| `--max-steps N` | tool calls allowed per request (default 15) |
| `--course NAME`, `--refresh`, `--chunk-chars N` | as in the menu |

### Notebook
```powershell
python -m ipykernel install --user --name ia2-venv
# then open AI_IA2_Notebook.ipynb in VS Code / Jupyter and pick the "ia2-venv" kernel
```
