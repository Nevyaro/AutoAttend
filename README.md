# AutoAttend - Vision Agent for Zero-Touch Attendance

Built for **Analytica 4.0, RV University**.

## Problem
Faculty lose 5-10 minutes of every class to manual roll calls. Proxy attendance is common and there is no early warning for students drifting away.

## Solution
AutoAttend turns any laptop webcam or existing CCTV feed into a **zero-touch attendance agent**:

- Detects and recognises every face in the room (MediaPipe + face embeddings, CPU only).
- Marks **PRESENT** after 3 s of continuous presence, **LATE** after 15 min, never twice per session.
- Blocks **proxy**: a photo held up to the camera never blinks, so it is not counted. Unknown faces are snapshotted to `reports/suspicious/`.
- The **agent layer** flags defaulters (<75%), students **AT RISK** (3 absent days), suggests actions (e.g. "Attendance low (40%) on Fridays - suggest rescheduling?") and builds a weekly report every Monday.

## Architecture
```
 Webcam / CCTV / demo video
          |
          v
 +---------------------+   MediaPipe Face Detection
 |   face_engine.py    |-> Embedding: face_recognition  --(fails?)--> DeepFace Facenet
 |                     |-> Similarity >= 0.55 => Known / else Unknown
 |                     |-> MediaPipe Face Mesh -> blink (EAR) liveness
 +----------+----------+
            | [{roll_no, name, bbox, confidence, status, live}]
            v
 +---------------------+   3 s rule, LATE rule, no duplicates
 | attendance_agent.py |-> proxy snapshots -> reports/suspicious/
 |   (the "agent")     |-> CSV: reports/attendance_2024.csv
 |                     |-> defaulters, AT RISK, suggestions, weekly report
 +----------+----------+
            v
 +---------------------+
 |   app.py Streamlit  |  Enroll | Live Class | AI Agent Insights | Reports
 +---------------------+
```

## Folder structure
```
AutoAttend/
  dataset/RollNo_Name/        3 enrollment photos per student
  embeddings/face_encodings.pkl   encrypted averaged embeddings
  reports/attendance_2024.csv     attendance history
  reports/suspicious/             proxy / unknown-face snapshots
  face_engine.py  attendance_agent.py  app.py  requirements.txt  README.md
```

## How to Run
```bash
python -m venv venv && venv\Scripts\activate      # Windows  (source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
streamlit run app.py
```

1. **Enroll Students**: roll no, name, class, 3 photos, consent tick.
2. **Live Class**: choose class + subject, **Start Class**. Blink once at the camera so liveness passes; after 3 s you appear in the live table.
3. **AI Agent Insights**: charts, defaulters, at-risk, suggestions, Excel export. Click **Load demo history** for instant sample data.

### Demo mode (stage backup)
No webcam or bad lighting? Tick **Demo mode** on the Live Class page, press Start Class, upload a classroom group photo or `classroom_sample.mp4`, then **Process demo file**. Enroll the people in the photo first so they can be recognised.

### Face backend
`face_recognition` is tried first (if installed). If it is missing or throws, the engine **falls back to DeepFace + Facenet** automatically. Force it with `FACE_BACKEND=deepface`. dlib is **not** needed: MediaPipe does all detection. Embeddings from different backends are not compatible, so re-enroll if the backend changes (the app tells you).

## Privacy & Ethics
- **Consent-based**: enrollment requires an explicit consent tick per student.
- **Data minimisation**: only an averaged numeric embedding is used for matching; raw photos stay in `dataset/` and can be removed.
- **Local only**: nothing leaves the machine. Embeddings are **encrypted at rest** (Fernet) with a key stored locally in `embeddings/secret.key`.
- **Retention**: `face_engine.purge_all_data()` deletes all embeddings, keys and photos. Run it at the end of the semester (can be scheduled).
- **Designed in line with GDPR principles** (consent, purpose limitation, minimisation, right to erasure) and India's DPDP Act 2023. This is a design intent, not a legal certification; a real deployment needs an institutional policy review.
- **Human in the loop**: the agent flags and suggests; faculty decide.

## Limitations (be upfront with judges)
- Blink liveness stops printed photos and simple screen replays, not 3D masks or video replays of a blinking face.
- Recognition accuracy depends on lighting, camera angle and distance from the camera.

## Future Scope
Multi-camera CCTV support, LMS/ERP integration, real email/SMS alerts, anti-spoofing models (depth/texture), mobile app for students, per-subject analytics.

## Team
Your Name
