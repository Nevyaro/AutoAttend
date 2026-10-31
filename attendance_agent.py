"""
attendance_agent.py - The "agent" layer of AutoAttend
=====================================================
Turns raw recognition results into attendance decisions and acts on them:

  * 3-second rule   : a student is marked only after being seen continuously for 3 s
  * LATE rule       : recognised more than 15 min after class start -> LATE
  * no duplicates   : one mark per student per session
  * proxy detection : unknown / spoofing faces are snapshotted to reports/suspicious/
  * analytics       : defaulters (<75 %), AT RISK (3 absent days in a row), smart suggestions
  * automation      : weekly report (auto on Mondays, or via button)
"""

import io
import random
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
SUSPICIOUS_DIR = REPORTS_DIR / "suspicious"
ATTENDANCE_CSV = REPORTS_DIR / "attendance_2024.csv"
for _d in (REPORTS_DIR, SUSPICIOUS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

COLUMNS = ["RollNo", "Name", "Class", "Subject", "Date", "Time", "Status", "Confidence", "SessionID"]
ATTENDED = ["PRESENT", "LATE"]

HOLD_SECONDS = 3            # must be seen this long to be marked
GAP_TOLERANCE_SEC = 1.5     # a gap longer than this restarts the 3 s timer
LATE_AFTER_MIN = 15         # minutes after start -> LATE
DEFAULTER_PCT = 75.0
SUSPICIOUS_COOLDOWN = 10    # seconds between snapshots

_lock = threading.RLock()
_session = None


# --------------------------------------------------------------------------- helpers
def _students():
    """Enrolled students; empty dict if the face engine can't be imported (e.g. in tests)."""
    try:
        import face_engine
        return face_engine.list_students()
    except Exception:
        return {}


def _ensure_csv():
    if not ATTENDANCE_CSV.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(ATTENDANCE_CSV, index=False)


_ensure_csv()


def load_attendance():
    """Full attendance history as a DataFrame (empty frame with the right columns if none)."""
    _ensure_csv()
    df = pd.read_csv(ATTENDANCE_CSV, dtype={"RollNo": str})
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


def _append_rows(rows):
    if not rows:
        return
    with _lock:
        _ensure_csv()
        pd.DataFrame(rows, columns=COLUMNS).to_csv(ATTENDANCE_CSV, mode="a", header=False, index=False)


def reset_attendance_data():
    """Wipe the attendance CSV (used to clear demo data)."""
    with _lock:
        pd.DataFrame(columns=COLUMNS).to_csv(ATTENDANCE_CSV, index=False)


# --------------------------------------------------------------------------- session logic
def start_class_session(class_name, subject, start_time=None):
    """Create a new session with a unique ID. Everything below works on the active session."""
    global _session
    start_time = start_time or datetime.now()
    with _lock:
        _session = {
            "session_id": f"{class_name}-{subject}-{start_time:%Y%m%d%H%M%S}".replace(" ", "_"),
            "class_name": class_name,
            "subject": subject,
            "start_time": start_time,
            "tracking": {},          # roll_no -> {first_seen, last_seen, best_conf}
            "marked": {},            # roll_no -> attendance row (prevents duplicates)
            "suspicious": 0,
            "last_suspicious": 0.0,
        }
    return _session


def get_session():
    return _session


def mark_attendance(recognized_students, now=None, require_live=True):
    """Core rule engine. Call once per processed frame with the output of
    face_engine.detect_and_recognize(). Returns the list of rows newly marked.

    `now` can be passed to simulate time (demo video / photo mode)."""
    s = _session
    if s is None:
        return []
    now = now or datetime.now()
    newly = []
    with _lock:
        for r in recognized_students:
            if r.get("status") != "Known":
                continue
            roll = r["roll_no"]
            if roll in s["marked"]:                       # no duplicates in a session
                continue
            if require_live and not r.get("live", False):  # photo/proxy: no blink -> ignore
                continue
            trk = s["tracking"].get(roll)
            if trk is None or (now - trk["last_seen"]).total_seconds() > GAP_TOLERANCE_SEC:
                trk = {"first_seen": now, "last_seen": now, "best_conf": r["confidence"]}
            else:                                          # still being seen continuously
                trk["last_seen"] = now
                trk["best_conf"] = max(trk["best_conf"], r["confidence"])
            s["tracking"][roll] = trk

            if (now - trk["first_seen"]).total_seconds() >= HOLD_SECONDS:
                late = now > s["start_time"] + timedelta(minutes=LATE_AFTER_MIN)
                row = {"RollNo": roll, "Name": r["name"], "Class": s["class_name"],
                       "Subject": s["subject"], "Date": now.strftime("%Y-%m-%d"),
                       "Time": now.strftime("%H:%M:%S"), "Status": "LATE" if late else "PRESENT",
                       "Confidence": round(trk["best_conf"], 1), "SessionID": s["session_id"]}
                s["marked"][roll] = row
                _append_rows([row])
                newly.append(row)
    return newly


def handle_suspicious(frame, results):
    """Proxy defence: if an UNKNOWN face is in front of the camera, save a snapshot
    (rate limited) to reports/suspicious/. `frame` should already have boxes drawn."""
    s = _session
    if s is None or not any(r.get("status") == "Unknown" for r in results):
        return None
    if time.time() - s["last_suspicious"] < SUSPICIOUS_COOLDOWN:
        return None
    import cv2
    s["last_suspicious"] = time.time()
    s["suspicious"] += 1
    path = SUSPICIOUS_DIR / f"suspicious_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def get_live_attendance():
    s = _session
    return pd.DataFrame(list(s["marked"].values()), columns=COLUMNS) if s else pd.DataFrame(columns=COLUMNS)


def end_class_session():
    """Close the session: everyone enrolled in the class who was never marked becomes ABSENT.
    Returns a summary dict (or None if no session is active)."""
    global _session
    s = _session
    if s is None:
        return None
    rows = []
    for roll, info in _students().items():
        if info.get("class_name") not in ("", None, s["class_name"]) or roll in s["marked"]:
            continue
        rows.append({"RollNo": roll, "Name": info["name"], "Class": s["class_name"],
                     "Subject": s["subject"], "Date": s["start_time"].strftime("%Y-%m-%d"),
                     "Time": "", "Status": "ABSENT", "Confidence": 0.0, "SessionID": s["session_id"]})
    _append_rows(rows)
    marked = list(s["marked"].values())
    summary = {"session_id": s["session_id"], "present": sum(m["Status"] == "PRESENT" for m in marked),
               "late": sum(m["Status"] == "LATE" for m in marked), "absent": len(rows),
               "suspicious": s["suspicious"], "csv": str(ATTENDANCE_CSV)}
    _session = None
    return summary


# --------------------------------------------------------------------------- analytics
def student_percentages(df=None):
    """Per-student attendance % = (PRESENT + LATE) / all recorded sessions."""
    df = load_attendance() if df is None else df
    if df.empty:
        return pd.DataFrame(columns=["RollNo", "Name", "Class", "Sessions", "Attended", "AttendancePct"])
    g = df.assign(att=df["Status"].isin(ATTENDED)).groupby("RollNo").agg(
        Name=("Name", "first"), Class=("Class", "first"), Sessions=("att", "size"), Attended=("att", "sum"))
    g["AttendancePct"] = (g["Attended"] / g["Sessions"] * 100).round(1)
    return g.reset_index()


def class_percentages(df=None):
    df = load_attendance() if df is None else df
    if df.empty:
        return pd.DataFrame(columns=["Class", "AttendancePct"])
    g = df.assign(att=df["Status"].isin(ATTENDED)).groupby("Class")["att"].mean().mul(100).round(1)
    return g.rename("AttendancePct").reset_index()


def get_defaulter_report(threshold=DEFAULTER_PCT):
    """Students whose attendance % is below the threshold (default 75), worst first."""
    sp = student_percentages()
    low = sp[sp["AttendancePct"] < threshold].sort_values("AttendancePct")
    return low.to_dict("records")


def auto_alert(consecutive=3):
    """Flag AT RISK when a student has been absent on the last `consecutive` class days in a row."""
    df = load_attendance()
    if df.empty:
        return []
    df = df.assign(att=df["Status"].isin(ATTENDED))
    out = []
    for roll, g in df.groupby("RollNo"):
        daily = g.groupby("Date")["att"].max().sort_index().tolist()
        run = 0
        for attended in reversed(daily):
            if attended:
                break
            run += 1
        if run >= consecutive:
            out.append({"RollNo": roll, "Name": g["Name"].iloc[0], "Class": g["Class"].iloc[0],
                        "ConsecutiveAbsences": run, "Flag": "AT RISK"})
    return out


def send_alert(roll_no, name, reason="attendance below 75%"):
    """Mock email: in production plug in smtplib / SendGrid here."""
    msg = f"[ALERT] Email sent to {name} ({roll_no}) and their mentor: {reason}"
    print(msg)
    return msg


def get_suggestions():
    """Auto-suggestions for faculty, e.g. weekday with unusually low attendance."""
    df = load_attendance()
    tips = []
    if df.empty:
        return tips
    df = df.assign(att=df["Status"].isin(ATTENDED), day=pd.to_datetime(df["Date"], errors="coerce"))
    df = df.dropna(subset=["day"])
    overall = df["att"].mean() * 100
    for wd, grp in df.groupby(df["day"].dt.day_name()):
        pct = grp["att"].mean() * 100
        if grp["day"].nunique() >= 2 and pct < 60 and pct < overall - 10:
            tips.append(f"Class attendance low ({pct:.0f}%) on {wd}s - suggest rescheduling?")
    n_susp = len(list(SUSPICIOUS_DIR.glob("*.jpg")))
    if n_susp:
        tips.append(f"{n_susp} suspicious snapshot(s) captured - review them in the Insights page.")
    return tips


# --------------------------------------------------------------------------- reports
def _excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            (frame if len(frame) else pd.DataFrame({"info": ["no data"]})).to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


def build_full_excel():
    """Bytes of an .xlsx with history, per-student %, defaulters and at-risk sheets."""
    return _excel_bytes({
        "Attendance": load_attendance(),
        "Student %": student_percentages(),
        "Defaulters": pd.DataFrame(get_defaulter_report()),
        "At Risk": pd.DataFrame(auto_alert()),
    })


def generate_weekly_report(today=None):
    """Summarise the last 7 days and save reports/weekly_report_<date>.xlsx. Returns (path, summary)."""
    today = today or date.today()
    df = load_attendance()
    df["_d"] = pd.to_datetime(df["Date"], errors="coerce")
    wk = df[(df["_d"] > pd.Timestamp(today) - pd.Timedelta(days=7)) & (df["_d"] <= pd.Timestamp(today))].drop(columns="_d")
    pct = student_percentages(wk) if len(wk) else student_percentages(wk)
    path = REPORTS_DIR / f"weekly_report_{today:%Y-%m-%d}.xlsx"
    path.write_bytes(_excel_bytes({"Week records": wk, "Student %": pct,
                                   "Defaulters": pd.DataFrame(get_defaulter_report()),
                                   "At Risk": pd.DataFrame(auto_alert())}))
    summary = {"records": len(wk),
               "avg_attendance": float(round(wk["Status"].isin(ATTENDED).mean() * 100, 1)) if len(wk) else 0.0}
    return path, summary


def maybe_auto_weekly_report():
    """Every Monday, create the weekly report once. Returns the path if one was created."""
    today = date.today()
    path = REPORTS_DIR / f"weekly_report_{today:%Y-%m-%d}.xlsx"
    if today.weekday() == 0 and not path.exists():
        return generate_weekly_report(today)[0]
    return None


# --------------------------------------------------------------------------- demo data
def seed_demo_history(days=14):
    """Fill the CSV with realistic synthetic history so the Insights page looks alive on stage.
    Uses enrolled students if any, otherwise a made-up roster. Fridays are intentionally weak."""
    rng = random.Random(42)
    roster = [{"roll": r, "name": i["name"], "cls": i.get("class_name") or "BCA"} for r, i in _students().items()]
    if not roster:
        names = ["Aarav", "Diya", "Rohan", "Sneha", "Karthik", "Meera", "Vikram", "Ananya"]
        roster = [{"roll": f"BCA{100 + i}", "name": n, "cls": "BCA" if i < 5 else "BBA"} for i, n in enumerate(names)]
    habitual = {roster[-1]["roll"]}            # a chronic low-attender -> defaulter
    at_risk = roster[0]["roll"]                # absent the last 3 days -> AT RISK
    day_list = [date.today() - timedelta(days=i) for i in range(days)]
    day_list = sorted(d for d in day_list if d.weekday() < 5)
    rows = []
    for idx, d in enumerate(day_list):
        for cls in sorted({s["cls"] for s in roster}):
            sid = f"DEMO-{cls}-{d:%Y%m%d}"
            for s in (x for x in roster if x["cls"] == cls):
                p = 0.9
                if d.weekday() == 4:
                    p = 0.4
                if s["roll"] in habitual:
                    p = 0.35
                present = rng.random() < p
                if s["roll"] == at_risk and idx >= len(day_list) - 3:
                    present = False
                minute = rng.randint(0, 22)
                rows.append({"RollNo": s["roll"], "Name": s["name"], "Class": cls, "Subject": "Demo Subject",
                             "Date": f"{d:%Y-%m-%d}", "Time": f"09:{minute:02d}:00" if present else "",
                             "Status": ("LATE" if minute > LATE_AFTER_MIN else "PRESENT") if present else "ABSENT",
                             "Confidence": round(rng.uniform(78, 97), 1) if present else 0.0, "SessionID": sid})
    _append_rows(rows)
    return len(rows)
