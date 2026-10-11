import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any

# Allow Google OAuth to accept granted scopes without throwing 'Scope has changed' error
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

# Resolve project files relative to this module so the agent works from any working directory
BASE_DIR = Path(__file__).resolve().parent
CREDS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"
SAMPLE_FILE = BASE_DIR / "classroom_assignments.json"

# Scopes exactly matching your Google Cloud Console configuration
SCOPES = [
    'https://www.googleapis.com/auth/classroom.courses.readonly',
    'https://www.googleapis.com/auth/classroom.student-submissions.me.readonly',
    'https://www.googleapis.com/auth/classroom.coursework.me'
]


def course_matches(course_name: str, course_filter: str) -> bool:
    """Whole-word, case-insensitive course filter ('all' or empty matches everything).
    'AI' also matches 'Artificial Intelligence'."""
    if not course_filter or course_filter.lower() == "all":
        return True
    name, f = course_name.lower(), course_filter.lower()
    if re.search(rf"\b{re.escape(f)}\b", name):
        return True
    return f == "ai" and "artificial intelligence" in name


def _format_due_date(due: Any) -> str:
    """Classroom returns dueDate as {'year': .., 'month': .., 'day': ..}."""
    if isinstance(due, dict) and due.get("year"):
        return f"{due['year']:04d}-{due.get('month', 1):02d}-{due.get('day', 1):02d}"
    return due or "N/A"


def _list_all(request_fn, key: str, **kwargs) -> List[Dict[str, Any]]:
    """Follows nextPageToken so courses/coursework beyond the first page are not missed."""
    items, token = [], None
    while True:
        res = request_fn(pageToken=token, **kwargs).execute()
        items.extend(res.get(key, []))
        token = res.get("nextPageToken")
        if not token:
            return items


def get_credentials(scopes=SCOPES, flow_scopes=None):
    """Loads token.json (refreshing it if expired) or runs the browser OAuth flow and saves token.json.
    `scopes=None` keeps whatever scopes token.json was granted; `flow_scopes` is what a fresh login asks for."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif CREDS_FILE.exists():
            print("\n[Google OAuth] Opening browser to authenticate with Google Classroom...")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), flow_scopes or scopes)
            creds = flow.run_local_server(port=0)
            TOKEN_FILE.write_text(creds.to_json())
            print("[Google OAuth] Authentication successful! Saved token.json.")
        else:
            raise RuntimeError("token.json is invalid and credentials.json is missing")
    return creds


def _get_live_assignments(course_filter: str) -> List[Dict[str, Any]]:
    from googleapiclient.discovery import build

    creds = get_credentials()
    service = build("classroom", "v1", credentials=creds)
    courses = _list_all(service.courses().list, "courses", studentId="me", courseStates=["ACTIVE"])

    pending = []
    for course in courses:
        c_id = course["id"]
        c_name = course.get("name", "Unknown Course")
        if not course_matches(c_name, course_filter):
            continue

        for cw in _list_all(service.courses().courseWork().list, "courseWork", courseId=c_id):
            cw_id = cw["id"]
            subs = service.courses().courseWork().studentSubmissions().list(
                courseId=c_id, courseWorkId=cw_id, userId="me"
            ).execute().get("studentSubmissions", [])
            state = subs[0].get("state") if subs else "NEW"

            # Only include assignments that are genuinely pending (not turned in and not already returned/graded)
            if state not in ("TURNED_IN", "RETURNED"):
                pending.append({
                    "id": cw_id,
                    "course": c_name,
                    "title": cw.get("title", "").strip(),
                    "description": cw.get("description", "").strip(),
                    "dueDate": _format_due_date(cw.get("dueDate")),
                    "state": state,
                    "materials": [
                        m.get("driveFile", {}).get("driveFile", {}).get("title")
                        or m.get("link", {}).get("url")
                        or m.get("youtubeVideo", {}).get("title")
                        or m.get("form", {}).get("title", "")
                        for m in cw.get("materials", [])
                    ],
                })
    return pending


def _get_sample_assignments(course_filter: str) -> List[Dict[str, Any]]:
    if not SAMPLE_FILE.exists():
        return []
    data = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    return [
        a for a in data.get("assignments", [])
        if a.get("submissionState", "").upper() != "TURNED_IN" and course_matches(a.get("course", ""), course_filter)
    ]


def get_pending_assignments(force_offline: bool = False, course_filter: str = "AI") -> List[Dict[str, Any]]:
    """
    Connects to Google Classroom via OAuth 2.0, scans all enrolled courses, and returns ONLY the coursework
    left to submit. Uses the sample classroom_assignments.json when offline, when credentials are absent,
    or when the live connection fails.
    """
    if not force_offline and (CREDS_FILE.exists() or TOKEN_FILE.exists()):
        try:
            pending = _get_live_assignments(course_filter)
            if not pending:
                print("[Google Classroom] No unsubmitted assignments found in active courses.")
            return pending
        except KeyboardInterrupt:
            print("\n[WARN] Google OAuth interrupted.")
        except Exception as e:
            print(f"\n[WARN] Could not reach Google Classroom: {e}")
        print("[WARN] Falling back to SAMPLE data from classroom_assignments.json - these are NOT your real assignments.")

    return _get_sample_assignments(course_filter)
