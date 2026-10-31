# AutoAttend: Zero-Touch Attendance Agent

An AI agent that takes attendance automatically from a classroom camera, blocks proxy attendance, and warns faculty about students who are slipping, so teachers get their class time back.

Built for **Analytica 4.0 (CampusAI: Agents for Academia)**, School of Economics and Business, RV University.
Focus area: **Teaching & Faculty Productivity Agents**.

---

## The problem

Manual roll call takes about 5 minutes per class. Across a day of classes and a whole year, that adds up to many hours of teaching time lost per teacher. Manual attendance also allows proxy marking, and nobody notices a student who is drifting away until it is too late.

## The solution

AutoAttend turns a laptop webcam (or an existing CCTV feed) into an attendance agent that **senses, decides and acts**:

| Step | What the agent does |
|------|---------------------|
| **Senses** | Detects every face in the room and recognises enrolled students |
| **Decides** | Marks PRESENT after 3 seconds of continuous presence, LATE after 15 minutes, never twice in one session |
| **Acts** | Blocks proxy (photos never blink), saves snapshots of unknown faces, flags defaulters and at-risk students, suggests actions, builds weekly reports |

### Features
- **Face detection** with Google MediaPipe (CPU only, no GPU, no dlib).
- **Face recognition** with `face_recognition` if it is installed, with an **automatic fallback to DeepFace (Facenet)** if it is missing or fails.
- **Anti-proxy blink check**: a face must blink before it is counted, so a printed photo or a photo on a phone screen is not accepted.
- **Unknown-face snapshots** saved for faculty review.
- **Defaulter report**: students below 75% attendance, with a "Send Alert" button (demo: prints a message, no real email).
- **At-risk flag**: students absent 3 class days in a row.
- **Smart suggestions**, for example: "Class attendance low (40%) on Fridays - suggest rescheduling?"
- **Weekly report** (Excel), generated automatically on Mondays or with a button.
- **Demo mode**: no webcam? Upload a classroom photo or video instead.

---

## How it works

```
 Webcam / CCTV / demo video
            |
            v
 +----------------------------+
 |       face_engine.py       |
 |  MediaPipe  -> find faces  |
 |  Embedding  -> face_recognition, or DeepFace (Facenet) as fallback
 |  Matching   -> similarity >= 0.55 = Known, else Unknown
 |  Liveness   -> MediaPipe Face Mesh, blink (eye aspect ratio)
 +--------------+-------------+
                |  who is in front of the camera, and are they live?
                v
 +----------------------------+
 |    attendance_agent.py     |
 |  3-second rule, LATE rule, no duplicates
 |  unknown faces -> reports/suspicious/
 |  defaulters, at-risk, suggestions, weekly report
 +--------------+-------------+
                v
 +----------------------------+
 |  app.py  (Streamlit)       |
 |  Enroll | Live Class | AI Agent Insights | Reports
 +----------------------------+
```

---

## Run it

**Requirements:** Windows, macOS or Linux, **Python 3.12** (3.10 or 3.11 also work; **3.13 and newer do not**, because TensorFlow and MediaPipe do not support them yet), and a webcam (optional, see Demo mode).

```bash
# 1. Create a virtual environment
py -3.12 -m venv venv           # Windows
# python3.12 -m venv venv       # macOS / Linux

# 2. Activate it
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS / Linux

# 3. Install (about 1 GB download, mostly TensorFlow)
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyarrow==17.0.0     # keeps Streamlit tables working with NumPy 1.26

# 4. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first time faces are recognised, DeepFace downloads the Facenet model (about 90 MB), so keep the internet on for that first run.

### Using the dashboard

1. **Enroll Students**: enter roll number, name and class, upload 3 clear front-facing photos, tick the consent box, click Enroll.
2. **Live Class**: choose a class and subject, click **Start Class**. Blink once at the camera and stay in view for about 3 seconds to be marked. Click **End Class & Save Report** to finish; anyone not seen is saved as ABSENT.
3. **AI Agent Insights**: charts, defaulters, at-risk students, suggestions and an Excel download. Click **Load demo history** for sample data.
4. **Reports**: filter by date and class, download as CSV.

### Demo mode (backup if the webcam fails)

Tick **Demo mode** on the Live Class page, click Start Class, upload a classroom group photo or a short video, then click **Process demo file**. Enroll the people in the photo first, otherwise they will show as Unknown.

---

## Privacy and ethics

- **Consent first:** a student cannot be enrolled without an explicit consent tick.
- **Stays on your machine:** nothing is sent to any server. Face data is stored locally.
- **Encrypted at rest:** the face embeddings (numbers, not photos) are encrypted with a local key.
- **Delete when done:** a purge function removes all embeddings, keys and photos; run it at the end of the semester.
- **Humans decide:** the agent flags and suggests, and faculty take the action.
- **Not in this repository:** face data, photos, attendance records and snapshots are excluded with `.gitignore` and are never uploaded.
- This project follows privacy principles such as consent, data minimisation and the right to erasure. It is **not** a legal compliance certification; a real deployment needs a review under the institution's policy and India's DPDP Act 2023.

## Limitations (honest ones)

- The blink check stops photos and simple screen replays, **not** a video of someone blinking or a 3D mask.
- Accuracy depends on lighting, camera angle and distance. A laptop webcam sees a few people close up; a whole classroom needs a better or wider camera.
- Alerts are a demo: they print or log a message and do not send real email or SMS yet.
- Tested with a small number of people, not a full class.

## Future scope

- Student Success Agent: combine attendance with marks to score each student's risk and draft personal follow-ups for faculty approval.
- Real email/SMS alerts, and links to the college ERP/LMS.
- Multi-camera support for large classrooms using existing CCTV.
- Stronger anti-spoofing (texture or depth based).
- Per-subject analytics and a student-facing view.

## Project files

```
face_engine.py         detection, recognition, liveness, encrypted storage
attendance_agent.py    attendance rules, analytics, alerts, reports
app.py                 Streamlit dashboard (4 pages)
requirements.txt       pinned dependencies
dataset/               enrollment photos          (created at runtime, not in Git)
embeddings/            encrypted face data        (created at runtime, not in Git)
reports/               attendance CSV, snapshots  (created at runtime, not in Git)
```

## Team

**Darshan**, MCA student
GitHub: [dj88-alt](https://github.com/dj88-alt)
