import sqlite3
import os
from datetime import datetime

DB_DIR = ".data"
DB_PATH = os.path.join(DB_DIR, "ergovision.db")

def init_db():
    """Initializes the hidden data directory and the SQLite database table."""
    os.makedirs(DB_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                duration_minutes REAL NOT NULL,
                ergonomic_score INTEGER NOT NULL,
                total_blinks INTEGER NOT NULL,
                avg_blink_rate INTEGER NOT NULL,
                slouch_warnings INTEGER NOT NULL,
                proximity_warnings INTEGER NOT NULL,
                fatigue_warnings INTEGER NOT NULL
            )
        """)
        conn.commit()

def log_session(record):
    """Inserts a completed monitoring session record into SQLite."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (
                timestamp, duration_minutes, ergonomic_score,
                total_blinks, avg_blink_rate, slouch_warnings,
                proximity_warnings, fatigue_warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"],
            record["duration_minutes"],
            record["ergonomic_score"],
            record["total_blinks"],
            record["avg_blink_rate"],
            record["slouch_warnings"],
            record["proximity_warnings"],
            record["fatigue_warnings"]
        ))
        conn.commit()

def generate_monthly_report():
    """
    Queries past sessions, computes aggregated monthly averages, 
    and generates clinical/ergonomic remarks for the client.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*),
                SUM(duration_minutes),
                AVG(ergonomic_score),
                AVG(avg_blink_rate),
                SUM(slouch_warnings),
                SUM(proximity_warnings),
                SUM(fatigue_warnings)
            FROM sessions
        """)
        row = cursor.fetchone()

    total_sessions = row[0] or 0
    if total_sessions == 0:
        return None

    total_hours = round((row[1] or 0.0) / 60.0, 2)
    avg_score = round(row[2] or 0.0, 1)
    avg_blink = round(row[3] or 0.0, 1)
    total_slouch = row[4] or 0
    total_proximity = row[5] or 0
    total_fatigue = row[6] or 0

    # Clinical & Ergonomic Evaluation Remarks
    remarks = []
    if avg_score >= 85:
        remarks.append("Exemplary overall workstation ergonomics. Minimal strain detected.")
    elif avg_score >= 70:
        remarks.append("Moderate ergonomic compliance. Mild postural fatigue observed.")
    else:
        remarks.append("Critical ergonomics risk! Persistent posture slouching or eye fatigue flagged.")

    if avg_blink < 10.0:
        remarks.append("Ocular Alert: Average blink frequency is below clinical threshold (12-15 BPM). High risk of Dry Eye Syndrome (CVS).")
    else:
        remarks.append("Ocular Status: Blink rate is within healthy physiological limits.")

    if total_proximity > total_slouch and total_proximity > 5:
        remarks.append("Behavioral Habit: High tendency to lean forward into the display. Recommend adjusting font size or display distance.")
    elif total_slouch > 5:
        remarks.append("Structural Habit: Frequent cervical spine flexion (Tech Neck). Recommend raising monitor height to eye level.")

    return {
        "total_sessions": total_sessions,
        "total_hours": total_hours,
        "avg_score": avg_score,
        "avg_blink": avg_blink,
        "total_slouch": total_slouch,
        "total_proximity": total_proximity,
        "total_fatigue": total_fatigue,
        "remarks": remarks
    }