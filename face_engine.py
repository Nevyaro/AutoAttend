"""
face_engine.py - Face detection, embedding, recognition and liveness for AutoAttend
===================================================================================

Pipeline (explain this to the judges):
  1. DETECT   : Google MediaPipe Face Detection (CPU, fast, pip-installs everywhere - no dlib needed).
  2. EMBED    : face_recognition (128-d) if it imports; otherwise automatic fallback to
                DeepFace + Facenet (128-d). Fallback also triggers if face_recognition
                throws at runtime.
  3. MATCH    : similarity against enrolled embeddings, threshold 0.55.
                - DeepFace/Facenet : true cosine similarity.
                - face_recognition : dlib embeddings are not zero-centred, so raw cosine is
                  always ~0.9+. We map euclidean distance to a 0-1 score: 1 - d/1.2
                  (0.55 similarity == distance 0.54, close to dlib's standard 0.6 tolerance).
  4. LIVENESS : MediaPipe Face Mesh -> Eye Aspect Ratio (EAR). A real face blinks, a
                printed photo or phone screenshot does not.
  5. TRACKING : light centroid tracker so we only run the (slow) embedding about once a
                second per face, while blink detection runs on every frame.

Embeddings are stored in embeddings/face_encodings.pkl (encrypted with Fernet when the
`cryptography` package is installed).
"""

import os
import pickle
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

try:  # optional: encryption at rest
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # pragma: no cover
    Fernet = None
    InvalidToken = Exception

# --------------------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
EMB_DIR = BASE_DIR / "embeddings"
EMB_FILE = EMB_DIR / "face_encodings.pkl"
KEY_FILE = EMB_DIR / "secret.key"
for _d in (DATASET_DIR, EMB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SIM_THRESHOLD = 0.55          # >= this similarity -> "Known"
RECOG_INTERVAL = 1.0          # seconds between re-embedding the same tracked face
EAR_CLOSED = 0.21             # eye aspect ratio below this == eye closed
BLINK_VALID_SECONDS = 20.0    # a blink keeps a face "live" for this long
MIN_FACE_PX = 40              # ignore tiny detections

GREEN = (0, 255, 0)           # BGR
RED = (0, 0, 255)

UNKNOWN_IDENTITY = {"roll_no": None, "name": "Unknown", "confidence": 0.0, "status": "Unknown"}

# --------------------------------------------------------------------------- backend selection
_backend = None
_fr = None
_backend_lock = threading.Lock()


def get_backend():
    """Return 'face_recognition' or 'deepface'. Chosen once, lazily.
    Set env FACE_BACKEND=deepface to force the fallback (useful on Windows)."""
    global _backend, _fr
    if _backend is not None:
        return _backend
    with _backend_lock:
        if _backend is not None:
            return _backend
        if os.environ.get("FACE_BACKEND", "").strip().lower() != "deepface":
            try:
                import face_recognition  # needs dlib under the hood
                _fr = face_recognition
                _backend = "face_recognition"
            except Exception as exc:
                print(f"[face_engine] face_recognition unavailable ({exc}); falling back to DeepFace.")
        if _backend is None:
            _backend = "deepface"
        return _backend


def _switch_to_deepface(reason):
    global _backend
    print(f"[face_engine] face_recognition failed at runtime ({reason}); switching to DeepFace.")
    _backend = "deepface"


# --------------------------------------------------------------------------- MediaPipe detection
_detector = None
_det_lock = threading.Lock()


def detect_faces(frame, min_conf=0.6):
    """Detect faces with MediaPipe. Returns list of (x, y, w, h, score) in pixel coords."""
    global _detector
    with _det_lock:
        if _detector is None:
            # model_selection=1 -> full-range model, works for faces up to ~5 m (classroom)
            _detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=min_conf)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = _detector.process(rgb)
    H, W = frame.shape[:2]
    faces = []
    for det in res.detections or []:
        rb = det.location_data.relative_bounding_box
        x, y = max(0, int(rb.xmin * W)), max(0, int(rb.ymin * H))
        w, h = int(rb.width * W), int(rb.height * H)
        w, h = min(w, W - x), min(h, H - y)
        if w >= MIN_FACE_PX and h >= MIN_FACE_PX:
            faces.append((x, y, w, h, float(det.score[0])))
    return faces


def _crop(frame, bbox, margin=0.2):
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    return frame[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)]


# --------------------------------------------------------------------------- embeddings
def _deepface_embedding(frame, bbox):
    try:
        from deepface import DeepFace
    except Exception as exc:
        raise RuntimeError(
            "Neither face_recognition nor DeepFace could be imported. "
            "Run: pip install -r requirements.txt") from exc
    crop = _crop(frame, bbox)
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (160, 160))
    # detector_backend='skip': MediaPipe already found the face, DeepFace only embeds
    rep = DeepFace.represent(img_path=crop, model_name="Facenet",
                             detector_backend="skip", enforce_detection=False)
    emb = np.asarray(rep[0]["embedding"], dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def get_embedding(frame, bbox):
    """Return a 128-d embedding for the face at bbox=(x, y, w, h), or None."""
    if get_backend() == "face_recognition":
        try:
            x, y, w, h = bbox
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # face_recognition location order is (top, right, bottom, left)
            enc = _fr.face_encodings(rgb, known_face_locations=[(y, x + w, y + h, x)])
            return enc[0].astype(np.float32) if enc else None
        except Exception as exc:
            _switch_to_deepface(exc)
    return _deepface_embedding(frame, bbox)


def _similarities(query, matrix, backend):
    if backend == "face_recognition":
        d = np.linalg.norm(matrix - query, axis=1)
        return np.clip(1.0 - d / 1.2, 0.0, 1.0)
    q = query / (np.linalg.norm(query) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ q  # cosine similarity


# --------------------------------------------------------------------------- encrypted pickle store
_db_lock = threading.RLock()
_db_cache = {"mtime": None, "data": None}


def _fernet():
    if Fernet is None:
        return None
    if KEY_FILE.exists():
        key = KEY_FILE.read_bytes()
    else:
        key = Fernet.generate_key()
        KEY_FILE.write_bytes(key)
    return Fernet(key)


def load_db():
    """Return {'backend': str|None, 'students': {roll_no: {name, class_name, embedding}}}."""
    with _db_lock:
        if not EMB_FILE.exists():
            return {"backend": None, "students": {}}
        mtime = EMB_FILE.stat().st_mtime
        if _db_cache["mtime"] == mtime and _db_cache["data"] is not None:
            return _db_cache["data"]
        raw = EMB_FILE.read_bytes()
        f = _fernet()
        if f is not None:
            try:
                raw = f.decrypt(raw)
            except InvalidToken:
                pass  # legacy plaintext pickle
        data = pickle.loads(raw)
        _db_cache.update(mtime=mtime, data=data)
        return data


def save_db(db):
    with _db_lock:
        raw = pickle.dumps(db)
        f = _fernet()
        if f is not None:
            raw = f.encrypt(raw)
        EMB_FILE.write_bytes(raw)
        _db_cache.update(mtime=EMB_FILE.stat().st_mtime, data=db)


def list_students():
    """{roll_no: {'name':..., 'class_name':...}} without the embeddings."""
    return {r: {"name": s["name"], "class_name": s.get("class_name", "")}
            for r, s in load_db()["students"].items()}


def count_enrolled():
    return len(load_db()["students"])


def purge_all_data():
    """Privacy: delete every embedding, key and stored photo (use at end of semester)."""
    with _db_lock:
        for p in (EMB_FILE, KEY_FILE):
            if p.exists():
                p.unlink()
        _db_cache.update(mtime=None, data=None)
        for child in DATASET_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text).strip())


# --------------------------------------------------------------------------- enrollment
def enroll_student(roll_no, name, image_paths, class_name=""):
    """Detect the face in each photo (MediaPipe), embed it, AVERAGE the embeddings
    and store {roll_no: name, embedding} in the pickle. Returns (ok: bool, message: str)."""
    roll_no, name = str(roll_no).strip(), str(name).strip()
    if not roll_no or not name:
        return False, "Roll number and name are required."

    embeddings = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = detect_faces(img)
        if not faces:
            continue
        x, y, w, h, _ = max(faces, key=lambda f: f[2] * f[3])  # largest face = the student
        emb = get_embedding(img, (x, y, w, h))
        if emb is not None:
            embeddings.append(emb)
    if not embeddings:
        return False, "No face detected in the uploaded photos. Use clear, front-facing, well-lit images."

    backend = get_backend()  # read AFTER embedding: a runtime fallback may have changed it
    db = load_db()
    if db["students"] and db.get("backend") not in (None, backend):
        return False, (f"Stored embeddings use '{db['backend']}' but the engine is now on '{backend}'. "
                       "Purge data and re-enroll everyone so all embeddings match.")

    mean = np.mean(embeddings, axis=0)
    if backend == "deepface":
        mean = mean / (np.linalg.norm(mean) + 1e-9)
    db["backend"] = backend
    db["students"][roll_no] = {"name": name, "class_name": class_name, "embedding": mean}
    save_db(db)
    return True, f"Enrolled {name} ({roll_no}) using {len(embeddings)}/{len(image_paths)} usable photos."


def student_folder(roll_no, name):
    """dataset/RollNo_Name/ - where the raw enrollment photos are kept."""
    p = DATASET_DIR / f"{_safe(roll_no)}_{_safe(name)}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- liveness (blink)
_mesh = None
_mesh_lock = threading.Lock()
_blink_state = {}
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(pts):
    """Eye Aspect Ratio = (|p2-p6| + |p3-p5|) / (2*|p1-p4|). Drops sharply when the eye closes."""
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    c = np.linalg.norm(pts[0] - pts[3])
    return (a + b) / (2.0 * c + 1e-9)


def check_liveness(frame, face_bbox, key="default"):
    """True once a blink was seen on this face within the last BLINK_VALID_SECONDS.
    A static photo never blinks -> never 'live' -> attendance is not marked."""
    global _mesh
    st = _blink_state.setdefault(key, {"closed": False, "last_blink": 0.0})
    crop = _crop(frame, face_bbox, margin=0.25)
    if crop.size:
        with _mesh_lock:
            if _mesh is None:
                _mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=False,
                    min_detection_confidence=0.5)
            res = _mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            ch, cw = crop.shape[:2]
            lm = res.multi_face_landmarks[0].landmark
            pts = lambda idx: np.array([[lm[i].x * cw, lm[i].y * ch] for i in idx])
            ear = (_ear(pts(LEFT_EYE)) + _ear(pts(RIGHT_EYE))) / 2.0
            if ear < EAR_CLOSED:
                st["closed"] = True
            elif st["closed"]:           # eye re-opened -> one full blink
                st["last_blink"] = time.time()
                st["closed"] = False
    return st["last_blink"] > 0 and (time.time() - st["last_blink"]) < BLINK_VALID_SECONDS


# --------------------------------------------------------------------------- tracking
_tracks = {}
_next_id = 0


def reset_tracking():
    """Call when a new class session starts."""
    global _next_id
    _tracks.clear()
    _blink_state.clear()
    _next_id = 0


def _match_track(bbox, used, now):
    """Nearest-centroid tracker: same face across frames keeps its identity + blink state."""
    global _next_id
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    best, best_d = None, 1e9
    for tid, t in _tracks.items():
        if tid in used:
            continue
        d = np.hypot(cx - t["center"][0], cy - t["center"][1])
        if d < best_d and d < max(w, h):
            best, best_d = tid, d
    if best is None:
        best = _next_id
        _next_id += 1
        _tracks[best] = {"identity": None, "last_recog": 0.0}
    _tracks[best].update(center=(cx, cy), last_seen=now)
    return best


def _prune_tracks(now, max_age=2.0):
    for tid in [t for t, v in _tracks.items() if now - v["last_seen"] > max_age]:
        _tracks.pop(tid, None)
        _blink_state.pop(f"t{tid}", None)


# --------------------------------------------------------------------------- recognition
def _identify(frame, bbox, db):
    emb = get_embedding(frame, bbox)
    if emb is None:
        return dict(UNKNOWN_IDENTITY)
    rolls = list(db["students"].keys())
    matrix = np.stack([db["students"][r]["embedding"] for r in rolls])
    sims = _similarities(emb, matrix, db["backend"])
    i = int(np.argmax(sims))
    best = float(sims[i])
    if best >= SIM_THRESHOLD:
        return {"roll_no": rolls[i], "name": db["students"][rolls[i]]["name"],
                "confidence": round(best * 100, 1), "status": "Known"}
    out = dict(UNKNOWN_IDENTITY)
    out["confidence"] = round(best * 100, 1)
    return out


def detect_and_recognize(frame, draw=True, check_live=True):
    """Detect every face in `frame`, identify it, optionally run blink liveness and draw boxes.
    Returns list of {roll_no, name, bbox, confidence(%), status: Known/Unknown, live: bool}."""
    now = time.time()
    db = load_db()
    usable = bool(db["students"]) and db.get("backend") == get_backend()
    results, used = [], set()
    for (x, y, w, h, _score) in detect_faces(frame):
        bbox = (x, y, w, h)
        tid = _match_track(bbox, used, now)
        used.add(tid)
        track = _tracks[tid]
        if track["identity"] is None or now - track["last_recog"] > RECOG_INTERVAL:
            track["identity"] = _identify(frame, bbox, db) if usable else dict(UNKNOWN_IDENTITY)
            track["last_recog"] = now
        ident = track["identity"]
        live = check_liveness(frame, bbox, key=f"t{tid}") if check_live else True
        results.append({**ident, "bbox": bbox, "live": live})
    _prune_tracks(now)
    if draw:
        draw_results(frame, results)
    return results


def draw_results(frame, results):
    """GREEN box = Known student, RED box = Unknown. Label shows name + confidence %."""
    for r in results:
        x, y, w, h = r["bbox"]
        known = r["status"] == "Known"
        color = GREEN if known else RED
        label = f"{r['name']} {r['confidence']:.0f}%"
        if known and not r.get("live", True):
            label += " | blink"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = y - 6 if y - th - 10 > 0 else y + h + th + 6
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 4), color, -1)
        cv2.putText(frame, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return frame
