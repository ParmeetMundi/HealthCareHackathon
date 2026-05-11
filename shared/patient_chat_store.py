"""
Patient chat history — PostgreSQL-backed per-patient conversation store.

Stores every question and response keyed by patient_id. Provides:
  - add_turn(): persist a Q or A turn
  - get_recent_turns(): fetch last N turns chronologically
  - get_context_for_prompt(): formatted string ready to inject as context

All data is stored in the patient_chat_history table (created by shared.db.init_db).
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

# Maximum number of recent turns to inject as context (configurable via env)
_MAX_CONTEXT_TURNS = int(os.getenv("PATIENT_CHAT_MAX_CONTEXT_TURNS", "20"))


def add_turn(patient_id: str, role: str, message: str, metadata: dict | None = None) -> None:
    """
    Record a single conversation turn.

    Args:
        patient_id: FHIR Patient resource ID (unique identifier).
        role:       "user" or "assistant".
        message:    The message text.
        metadata:   Optional dict of extra info (e.g. task_id, agent used).
    """
    if not patient_id or not message:
        return
    from shared.db import get_conn, put_conn

    meta_json = json.dumps(metadata or {})
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patient_chat_history (patient_id, role, message, metadata)
                VALUES (%s, %s, %s, %s);
                """,
                (patient_id, role, message, meta_json),
            )
        conn.commit()
        logger.info(
            "chat_history_stored patient_id=%s role=%s message_len=%d",
            patient_id, role, len(message),
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        put_conn(conn)


def get_recent_turns(patient_id: str, limit: int = _MAX_CONTEXT_TURNS) -> list[dict]:
    """
    Retrieve the most recent conversation turns for a patient.

    Returns a list of dicts ordered oldest-first:
        [{"role": "user", "message": "...", "created_at": "..."}, ...]
    """
    if not patient_id:
        return []
    from shared.db import get_conn, put_conn

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, message, metadata, created_at
                FROM patient_chat_history
                WHERE patient_id = %s
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (patient_id, limit),
            )
            rows = cur.fetchall()

        # Reverse so they're in chronological order (oldest first)
        turns = []
        for role, message, meta, created_at in reversed(rows):
            if isinstance(meta, dict):
                parsed_meta = meta
            elif meta:
                parsed_meta = json.loads(meta)
            else:
                parsed_meta = {}
            turns.append({
                "role": role,
                "message": message,
                "metadata": parsed_meta,
                "created_at": str(created_at),
            })
        logger.info(
            "chat_history_retrieved patient_id=%s turns=%d",
            patient_id, len(turns),
        )
        return turns
    finally:
        put_conn(conn)


def get_context_for_prompt(patient_id: str, max_turns: int = _MAX_CONTEXT_TURNS) -> str:
    """
    Build a formatted conversation history string ready to inject into
    the crew's question context.

    Returns an empty string if no history exists.
    """
    turns = get_recent_turns(patient_id, limit=max_turns)
    if not turns:
        return ""

    lines = ["[PATIENT CONVERSATION HISTORY]"]
    for turn in turns:
        prefix = "Patient/User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{prefix} ({turn['created_at']}): {turn['message']}")
    lines.append("[END PATIENT CONVERSATION HISTORY]\n")

    context = "\n".join(lines)
    logger.info(
        "chat_context_built patient_id=%s turns=%d context_len=%d",
        patient_id, len(turns), len(context),
    )
    return context
