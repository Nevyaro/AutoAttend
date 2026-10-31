"""
app.py - AutoAttend Faculty Dashboard (Streamlit)
Run:  streamlit run app.py

Pages: 1) Enroll Students  2) Live Class (main demo)  3) AI Agent Insights  4) Reports
"""

import tempfile
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import attendance_agent as agent
import face_engine as fe

st.set_page_config(page_title="AutoAttend", page_icon="🎓", layout="wide")

CLASSES = ["BCA", "BBA"]
st.session_state.setdefault("running", False)

# Agent automation: on Mondays the weekly report is generated automatically (once per browser session)
if "weekly_checked" not in st.session_state:
    st.session_state.weekly_checked = True
    created = agent.maybe_auto_weekly_report()
    if created:
        st.toast(f"Monday auto-report generated: {created.name}")


# =========================================================================== helpers
def render_panel(stats_ph, table_ph, class_name):
    """Top stats (Total | Present | Absent) + live attendance table. Called every frame in live mode."""
    s = agent.get_session()
    marked = list(s["marked"].values()) if s else []
    enrolled = [r for r, i in fe.list_students().items() if i.get("class_name") in ("", None, class_name)]
    total = max(len(enrolled), len(marked))
    present = len(marked)
    with stats_ph.container():
        a, b, c = st.columns(3)
        a.metric("Total Students", total)
        b.metric("Present", present)
        c.metric("Absent", max(total - present, 0))
    if marked:
        df = pd.DataFrame(marked)[["RollNo", "Name", "Time", "Status"]].rename(columns={"RollNo": "Roll No"})
        df["Status"] = df["Status"].map(lambda x: ("🟢 " if x == "PRESENT" else "🟠 ") + x)
        table_ph.dataframe(df, hide_index=True, use_container_width=True)
    else:
        table_ph.info("Nobody marked yet.")


def run_webcam(video_ph, stats_ph, table_ph, class_name, require_live):
    """Live loop. cv2.VideoCapture(0) feeds frames into a Streamlit placeholder (video_ph).
    The loop ends when the user clicks 'End Class' (Streamlit reruns the script)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.running = False
        video_ph.error("Webcam not available. Tick 'Demo mode' and upload a classroom photo/video instead.")
        return
    last_render = 0.0
    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                video_ph.warning("Could not read from the webcam.")
                break
            if frame.shape[1] > 800:                       # keep CPU load reasonable
                scale = 800 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            results = fe.detect_and_recognize(frame, draw=True, check_live=require_live)
            newly = agent.mark_attendance(results, require_live=require_live)
            agent.handle_suspicious(frame, results)        # unknown face -> snapshot
            video_ph.image(frame, channels="BGR", use_column_width=True)
            if newly or time.time() - last_render > 1.0:
                render_panel(stats_ph, table_ph, class_name)
                last_render = time.time()
            time.sleep(0.01)
    finally:
        cap.release()                                      # always free the camera


def process_demo_file(uploaded, video_ph, stats_ph, table_ph, class_name):
    """DEMO MODE (stage backup): process an uploaded group photo or classroom_sample.mp4."""
    session = agent.get_session()
    t0 = session["start_time"]
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    fe.reset_tracking()

    if name.endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not read that image.")
            return
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, None, fx=1280 / frame.shape[1], fy=1280 / frame.shape[1])
        results = fe.detect_and_recognize(frame, draw=True, check_live=False)
        # A photo has no time dimension, so we simulate the 3-second hold by replaying the same
        # result at 1-second steps (each step must stay under agent.GAP_TOLERANCE_SEC)
        for step_s in range(agent.HOLD_SECONDS + 2):
            agent.mark_attendance(results, now=t0 + timedelta(seconds=step_s), require_live=False)
        agent.handle_suspicious(frame, results)
        video_ph.image(frame, channels="BGR", use_column_width=True)
        render_panel(stats_ph, table_ph, class_name)
        return

    # video: use video-time (frame index / fps) so the 3-second rule still applies
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(data)
    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, int(fps / 5))                            # analyse ~5 frames per video-second
    bar = st.progress(0.0, text="Processing demo video...")
    idx = 0
    try:
        while True:
            if idx % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, None, fx=960 / frame.shape[1], fy=960 / frame.shape[1])
            results = fe.detect_and_recognize(frame, draw=True, check_live=False)
            agent.mark_attendance(results, now=t0 + timedelta(seconds=idx / fps), require_live=False)
            agent.handle_suspicious(frame, results)
            video_ph.image(frame, channels="BGR", use_column_width=True)
            render_panel(stats_ph, table_ph, class_name)
            bar.progress(min(idx / total_frames, 1.0))
            idx += 1
    finally:
        cap.release()
        bar.empty()


# =========================================================================== PAGE 1
def page_enroll():
    st.title("👩‍🎓 Enroll Students")
    st.caption("Upload 3 clear, front-facing photos per student. Only the averaged face embedding is kept.")
    with st.form("enroll_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        roll = c1.text_input("Roll No")
        name = c2.text_input("Name")
        cls = c3.selectbox("Class", CLASSES)
        photos = st.file_uploader("Upload 3 photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        consent = st.checkbox("The student has given informed consent to biometric (face) processing")
        submitted = st.form_submit_button("Enroll", type="primary")
    if submitted:
        if not (roll and name):
            st.error("Roll No and Name are required.")
        elif len(photos) < 3:
            st.error("Please upload 3 photos.")
        elif not consent:
            st.error("Consent is required before enrolling a student.")
        else:
            folder = fe.student_folder(roll, name)         # dataset/RollNo_Name/
            paths = []
            for i, up in enumerate(photos[:3], 1):
                p = folder / f"img{i}.jpg"
                p.write_bytes(up.getvalue())
                paths.append(p)
            with st.spinner("Detecting face and computing embeddings..."):
                ok, msg = fe.enroll_student(roll, name, paths, class_name=cls)
            (st.success if ok else st.error)(msg)

    students = fe.list_students()
    st.metric("Total enrolled", len(students))
    if students:
        st.dataframe(pd.DataFrame([{"Roll No": r, "Name": i["name"], "Class": i["class_name"]}
                                   for r, i in students.items()]), hide_index=True, use_container_width=True)


# =========================================================================== PAGE 2
def page_live():
    st.title("🎥 Live Class")
    c1, c2, c3 = st.columns([1, 2, 2])
    class_name = c1.selectbox("Class", CLASSES)
    subject = c2.text_input("Subject", "Python Programming")
    demo = c3.checkbox("Demo mode (upload classroom photo / video)")
    liveness = c3.checkbox("Anti-proxy blink check (webcam)", value=True, disabled=demo)
    require_live = liveness and not demo

    b1, b2 = st.columns(2)
    start_clicked = b1.button("▶ Start Class", type="primary", use_container_width=True)
    end_clicked = b2.button("⏹ End Class & Save Report", use_container_width=True)

    if end_clicked:
        st.session_state.running = False
        summary = agent.end_class_session()
        if summary:
            st.success(f"Report saved to {summary['csv']} | Present {summary['present']} · "
                       f"Late {summary['late']} · Absent {summary['absent']} · "
                       f"Suspicious snapshots {summary['suspicious']}")
        else:
            st.warning("No active class to end.")
    if start_clicked:
        fe.reset_tracking()
        agent.start_class_session(class_name, subject, datetime.now())
        st.session_state.running = not demo

    session = agent.get_session()
    cls_for_panel = session["class_name"] if session else class_name
    stats_ph = st.empty()
    left, right = st.columns([3, 2])
    left.subheader("Camera")
    right.subheader("Live Attendance")
    video_ph = left.empty()
    table_ph = right.empty()
    render_panel(stats_ph, table_ph, cls_for_panel)

    if session is None:
        video_ph.info("Press **Start Class** to begin.")
    elif demo:
        up = left.file_uploader("Group photo or classroom_sample.mp4",
                                type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
        if up is not None and left.button("Process demo file"):
            process_demo_file(up, video_ph, stats_ph, table_ph, cls_for_panel)
    elif st.session_state.running:
        run_webcam(video_ph, stats_ph, table_ph, cls_for_panel, require_live)


# =========================================================================== PAGE 3
def page_insights():
    st.title("🤖 AI Agent Insights")
    a, b, c = st.columns(3)
    if a.button("Load demo history"):
        st.toast(f"Added {agent.seed_demo_history()} demo records")
    if b.button("Simulate Monday weekly report"):
        path, summary = agent.generate_weekly_report()
        st.success(f"{path.name}: {summary['records']} records, avg attendance {summary['avg_attendance']}%")
    if c.button("Clear all attendance data"):
        agent.reset_attendance_data()
        st.toast("Attendance data cleared")

    df = agent.load_attendance()
    if df.empty:
        st.info("No attendance recorded yet. Run a class or click 'Load demo history'.")
        return

    for tip in agent.get_suggestions():
        st.warning("💡 " + tip)

    col1, col2 = st.columns(2)
    cp = agent.class_percentages(df)
    fig = px.bar(cp, x="Class", y="AttendancePct", text="AttendancePct", range_y=[0, 100],
                 title="Class attendance %", color_discrete_sequence=["#3b82f6"])
    fig.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="75% minimum")
    col1.plotly_chart(fig, use_container_width=True)

    latest = df["Date"].max()
    day = df[df["Date"] == (datetime.now().strftime("%Y-%m-%d")
                            if (df["Date"] == datetime.now().strftime("%Y-%m-%d")).any() else latest)]
    present = int(day["Status"].isin(agent.ATTENDED).sum())
    pie = px.pie(names=["Present", "Absent"], values=[present, len(day) - present],
                 title=f"Present vs Absent ({day['Date'].iloc[0]})", color_discrete_sequence=["#2ecc71", "#e74c3c"])
    col2.plotly_chart(pie, use_container_width=True)

    sp = agent.student_percentages(df)
    sp["Below 75%"] = sp["AttendancePct"] < 75
    st.plotly_chart(px.bar(sp, x="Name", y="AttendancePct", color="Below 75%", range_y=[0, 100],
                           title="Attendance % per student",
                           color_discrete_map={True: "#e74c3c", False: "#2ecc71"}), use_container_width=True)

    st.subheader("🔴 Defaulters (< 75%)")
    defaulters = agent.get_defaulter_report()
    if defaulters:
        ddf = pd.DataFrame(defaulters)[["RollNo", "Name", "Class", "Attended", "Sessions", "AttendancePct"]]
        st.dataframe(ddf.style.apply(lambda r: ["background-color:#ffd6d6;color:#8b0000"] * len(r), axis=1),
                     hide_index=True, use_container_width=True)
        for d in defaulters:
            x, y = st.columns([4, 1])
            x.write(f"**{d['Name']}** ({d['RollNo']}) - {d['AttendancePct']}%")
            if y.button("Send Alert", key=f"alert_{d['RollNo']}"):
                st.success(agent.send_alert(d["RollNo"], d["Name"], f"attendance {d['AttendancePct']}% (<75%)"))
    else:
        st.success("No defaulters 🎉")

    st.subheader("⚠️ At Risk (absent 3 days in a row)")
    risk = agent.auto_alert()
    if risk:
        st.dataframe(pd.DataFrame(risk), hide_index=True, use_container_width=True)
        for r in risk:
            if st.button(f"Alert {r['Name']}", key=f"risk_{r['RollNo']}"):
                st.success(agent.send_alert(r["RollNo"], r["Name"], "absent 3 consecutive days"))
    else:
        st.info("No at-risk students.")

    snaps = sorted(agent.SUSPICIOUS_DIR.glob("*.jpg"))[-4:]
    if snaps:
        st.subheader("🕵️ Suspicious snapshots (unknown faces)")
        cols = st.columns(len(snaps))
        for col, p in zip(cols, snaps):
            col.image(str(p), caption=p.name, use_column_width=True)

    st.download_button("⬇ Download Full Report (Excel)", agent.build_full_excel(),
                       file_name="autoattend_full_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =========================================================================== PAGE 4
def page_reports():
    st.title("📄 Reports")
    df = agent.load_attendance()
    if df.empty:
        st.info("No attendance history yet.")
        return
    dates = pd.to_datetime(df["Date"], errors="coerce")
    c1, c2 = st.columns(2)
    rng = c1.date_input("Date range", value=(dates.min().date(), dates.max().date()))
    classes = c2.multiselect("Class", sorted(df["Class"].dropna().unique()), default=sorted(df["Class"].dropna().unique()))
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (rng[0], rng[0])) if isinstance(rng, tuple) else (rng, rng)
    view = df[(dates.dt.date >= start) & (dates.dt.date <= end) & (df["Class"].isin(classes))]
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button("⬇ Download CSV", view.to_csv(index=False), file_name="attendance_report.csv", mime="text/csv")


# =========================================================================== navigation
st.sidebar.title("🎓 AutoAttend")
page = st.sidebar.radio("Navigate", ["Enroll Students", "Live Class", "AI Agent Insights", "Reports"])
st.sidebar.caption(f"Face backend: **{fe.get_backend()}**")
st.sidebar.caption(f"Enrolled students: {fe.count_enrolled()}")

{"Enroll Students": page_enroll, "Live Class": page_live,
 "AI Agent Insights": page_insights, "Reports": page_reports}[page]()
