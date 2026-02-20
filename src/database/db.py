"""
SQLite Database — Alarms, Watchlists, Signal History, Performance Tracking, ML.
Full signal lifecycle: PENDING → T1_HIT/T2_HIT/T3_HIT/SL_HIT/EXPIRED
"""
import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from src.config import DB_PATH

logger = logging.getLogger("matrix_trader.database")

# Ensure data directory exists
os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else "data", exist_ok=True)


class Database:
    """SQLite database manager with performance tracking and ML support."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alarms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    direction TEXT NOT NULL DEFAULT 'above',
                    is_bist INTEGER NOT NULL DEFAULT 0,
                    triggered INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    triggered_at TEXT
                );

                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    is_bist INTEGER NOT NULL DEFAULT 0,
                    added_at TEXT NOT NULL,
                    UNIQUE(user_id, symbol)
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    target1 REAL,
                    target2 REAL,
                    target3 REAL,
                    rr REAL,
                    is_crypto INTEGER NOT NULL DEFAULT 1,
                    sent_at TEXT NOT NULL,
                    -- Performance tracking
                    outcome TEXT DEFAULT 'PENDING',
                    t1_hit INTEGER DEFAULT 0,
                    t1_hit_at TEXT,
                    t1_duration_min INTEGER,
                    t2_hit INTEGER DEFAULT 0,
                    t2_hit_at TEXT,
                    t2_duration_min INTEGER,
                    t3_hit INTEGER DEFAULT 0,
                    t3_hit_at TEXT,
                    t3_duration_min INTEGER,
                    sl_hit INTEGER DEFAULT 0,
                    sl_hit_at TEXT,
                    sl_duration_min INTEGER,
                    max_favorable REAL DEFAULT 0,
                    max_adverse REAL DEFAULT 0,
                    exit_price REAL,
                    pnl_pct REAL,
                    closed_at TEXT,
                    -- Feature snapshot for ML
                    features TEXT
                );

                CREATE TABLE IF NOT EXISTS signal_cooldown (
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, direction)
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_scanned INTEGER DEFAULT 0,
                    signals_sent INTEGER DEFAULT 0,
                    crypto_signals INTEGER DEFAULT 0,
                    bist_signals INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0,
                    accuracy_pct REAL DEFAULT 0,
                    t1_hit_rate REAL DEFAULT 0,
                    t2_hit_rate REAL DEFAULT 0,
                    t3_hit_rate REAL DEFAULT 0,
                    summary TEXT
                );

                CREATE TABLE IF NOT EXISTS ml_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_data BLOB NOT NULL,
                    feature_names TEXT NOT NULL,
                    accuracy REAL DEFAULT 0,
                    total_samples INTEGER DEFAULT 0,
                    trained_at TEXT NOT NULL,
                    metrics TEXT
                );

                CREATE TABLE IF NOT EXISTS accuracy_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period TEXT NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    t1_hits INTEGER DEFAULT 0,
                    t2_hits INTEGER DEFAULT 0,
                    t3_hits INTEGER DEFAULT 0,
                    sl_hits INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_pnl REAL DEFAULT 0,
                    avg_t1_duration_min REAL,
                    avg_t2_duration_min REAL,
                    avg_t3_duration_min REAL,
                    by_tier TEXT,
                    by_confidence TEXT,
                    calculated_at TEXT NOT NULL
                );
            """)
            conn.commit()
        finally:
            conn.close()

    # ─── Alarms ──────────────────────────────────────────

    def add_alarm(self, user_id: str, symbol: str, target_price: float,
                  direction: str = "above", is_bist: bool = False) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO alarms (user_id, symbol, target_price, direction, is_bist, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, symbol.upper(), target_price, direction, int(is_bist), datetime.utcnow().isoformat())
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_active_alarms(self, user_id: str = None) -> list[dict]:
        conn = self._get_conn()
        try:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM alarms WHERE triggered = 0 AND user_id = ?", (user_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM alarms WHERE triggered = 0").fetchall()
            return [self._alarm_to_dict(r) for r in rows]
        finally:
            conn.close()

    def trigger_alarm(self, alarm_id: int):
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE alarms SET triggered = 1, triggered_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), alarm_id)
            )
            conn.commit()
        finally:
            conn.close()

    def delete_alarm(self, alarm_id: int, user_id: str):
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM alarms WHERE id = ? AND user_id = ?", (alarm_id, user_id))
            conn.commit()
        finally:
            conn.close()

    # ─── Watchlist ───────────────────────────────────────

    def add_to_watchlist(self, user_id: str, symbol: str, is_bist: bool = False) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist (user_id, symbol, is_bist, added_at) VALUES (?, ?, ?, ?)",
                (user_id, symbol.upper(), int(is_bist), datetime.utcnow().isoformat())
            )
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def remove_from_watchlist(self, user_id: str, symbol: str) -> bool:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?",
                (user_id, symbol.upper())
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_watchlist(self, user_id: str) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE user_id = ? ORDER BY added_at DESC", (user_id,)
            ).fetchall()
            return [{"id": r[0], "user_id": r[1], "symbol": r[2],
                      "is_bist": bool(r[3]), "added_at": r[4]} for r in rows]
        finally:
            conn.close()

    # ─── Signals ─────────────────────────────────────────

    def record_signal(self, symbol: str, direction: str, tier: str,
                      confidence: int, entry_price: float, stop_loss: float = 0,
                      targets: dict = None, rr: float = 0, is_crypto: bool = True,
                      features: dict = None) -> int:
        """Record a signal with full feature snapshot for ML training."""
        conn = self._get_conn()
        try:
            targets = targets or {}
            cursor = conn.execute(
                """INSERT INTO signals
                (symbol, direction, tier, confidence, entry_price, stop_loss,
                 target1, target2, target3, rr, is_crypto, sent_at, features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol, direction, tier, confidence, entry_price, stop_loss,
                    targets.get("t1", 0), targets.get("t2", 0), targets.get("t3", 0),
                    rr, int(is_crypto), datetime.utcnow().isoformat(),
                    json.dumps(features) if features else None,
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_pending_signals(self) -> list[dict]:
        """Get all signals with PENDING outcome for tracking."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM signals WHERE outcome = 'PENDING' ORDER BY sent_at DESC"
            ).fetchall()
            return [self._signal_to_dict(r) for r in rows]
        finally:
            conn.close()

    def update_signal_target(self, signal_id: int, target_num: int, hit_price: float):
        """Mark a target as hit with timestamp and duration."""
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        try:
            # Get sent_at to calculate duration
            row = conn.execute("SELECT sent_at FROM signals WHERE id = ?", (signal_id,)).fetchone()
            duration_min = 0
            if row:
                sent_at = datetime.fromisoformat(row[0])
                duration_min = int((datetime.utcnow() - sent_at).total_seconds() / 60)

            col_hit = f"t{target_num}_hit"
            col_at = f"t{target_num}_hit_at"
            col_dur = f"t{target_num}_duration_min"

            conn.execute(
                f"UPDATE signals SET {col_hit} = 1, {col_at} = ?, {col_dur} = ? WHERE id = ?",
                (now, duration_min, signal_id)
            )

            # Determine best outcome
            outcome = f"T{target_num}_HIT"
            conn.execute(
                "UPDATE signals SET outcome = ?, exit_price = ? WHERE id = ? AND outcome = 'PENDING'",
                (outcome, hit_price, signal_id)
            )
            conn.commit()
        finally:
            conn.close()

    def update_signal_sl_hit(self, signal_id: int, hit_price: float):
        """Mark signal as stopped out."""
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT sent_at, entry_price, direction FROM signals WHERE id = ?",
                               (signal_id,)).fetchone()
            duration_min = 0
            pnl_pct = 0.0
            if row:
                sent_at = datetime.fromisoformat(row[0])
                duration_min = int((datetime.utcnow() - sent_at).total_seconds() / 60)
                entry = row[1]
                direction = row[2]
                if entry > 0:
                    if direction == "BUY":
                        pnl_pct = (hit_price - entry) / entry * 100
                    else:
                        pnl_pct = (entry - hit_price) / entry * 100

            conn.execute(
                """UPDATE signals SET outcome = 'SL_HIT', sl_hit = 1, sl_hit_at = ?,
                   sl_duration_min = ?, exit_price = ?, pnl_pct = ?, closed_at = ?
                   WHERE id = ?""",
                (now, duration_min, hit_price, round(pnl_pct, 2), now, signal_id)
            )
            conn.commit()
        finally:
            conn.close()

    def update_signal_pnl(self, signal_id: int, exit_price: float, pnl_pct: float, outcome: str):
        """Update final PnL and outcome for a signal."""
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE signals SET exit_price = ?, pnl_pct = ?, outcome = ?, closed_at = ?
                   WHERE id = ?""",
                (exit_price, round(pnl_pct, 2), outcome, datetime.utcnow().isoformat(), signal_id)
            )
            conn.commit()
        finally:
            conn.close()

    def update_signal_trailing_sl(self, signal_id: int, trailing_sl: float):
        """Update trailing stop-loss value for a signal."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE signals SET stop_loss = ? WHERE id = ?",
                (round(trailing_sl, 8), signal_id)
            )
            conn.commit()
        finally:
            conn.close()

    def get_open_signals_count(self) -> int:
        """Get count of currently open (PENDING) signals."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome = 'PENDING'"
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def get_open_signals_by_direction(self, direction: str) -> int:
        """Get count of open signals in a specific direction."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE outcome = 'PENDING' AND direction = ?",
                (direction,)
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def get_today_losses(self) -> tuple[int, float]:
        """Get consecutive recent losses and total daily loss percentage."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        conn = self._get_conn()
        try:
            # Today's closed signals
            rows = conn.execute(
                """SELECT pnl_pct, outcome FROM signals
                   WHERE closed_at LIKE ? AND outcome != 'PENDING' AND outcome != 'EXPIRED'
                   ORDER BY closed_at DESC""",
                (f"{today}%",)
            ).fetchall()

            if not rows:
                return 0, 0.0

            # Count consecutive losses from most recent
            consecutive_losses = 0
            for row in rows:
                if row[0] is not None and row[0] < 0:
                    consecutive_losses += 1
                else:
                    break

            # Total daily loss
            total_loss = sum(r[0] for r in rows if r[0] is not None and r[0] < 0)

            return consecutive_losses, abs(total_loss)
        finally:
            conn.close()

    def update_signal_extremes(self, signal_id: int, max_favorable: float, max_adverse: float):
        """Update max favorable/adverse excursion for a signal."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE signals SET max_favorable = ?, max_adverse = ? WHERE id = ?",
                (round(max_favorable, 4), round(max_adverse, 4), signal_id)
            )
            conn.commit()
        finally:
            conn.close()

    def expire_old_signals(self, max_age_hours: int = 72):
        """Mark old PENDING signals as EXPIRED."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE signals SET outcome = 'EXPIRED', closed_at = ? WHERE outcome = 'PENDING' AND sent_at < ?",
                (datetime.utcnow().isoformat(), cutoff)
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_signals(self, limit: int = 20) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY sent_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._signal_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_closed_signals(self, limit: int = 500) -> list[dict]:
        """Get signals with known outcomes for ML training."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM signals
                   WHERE outcome != 'PENDING' AND outcome != 'EXPIRED'
                   ORDER BY sent_at DESC LIMIT ?""", (limit,)
            ).fetchall()
            return [self._signal_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_signals_with_features(self, limit: int = 1000) -> list[dict]:
        """Get signals that have feature snapshots for ML training."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM signals
                   WHERE outcome != 'PENDING' AND features IS NOT NULL
                   ORDER BY sent_at DESC LIMIT ?""", (limit,)
            ).fetchall()
            return [self._signal_to_dict(r) for r in rows]
        finally:
            conn.close()

    # ─── Cooldown ────────────────────────────────────────

    def check_cooldown(self, symbol: str, cooldown_minutes: int = 240,
                       direction: str = None) -> bool:
        """
        Smart pending-aware cooldown check.
        Returns True if signal is IN cooldown (cannot send).

        Rules (in order):
        1. PENDING signal exists for this symbol → always block
        2. Last signal was SL_HIT recently → 2× cooldown
        3. Last signal hit only T1 (not T2) → 1.5× cooldown
        4. Last signal hit T2/T3 → normal cooldown (re-entry OK)
        5. Last signal EXPIRED → 0.75× cooldown (market moved on)
        6. Fallback to simple time-based check in signal_cooldown table
        """
        conn = self._get_conn()
        try:
            # ── 1. Block if there's already a PENDING signal for this symbol ──
            pending_row = conn.execute(
                "SELECT id FROM signals WHERE symbol = ? AND outcome = 'PENDING' LIMIT 1",
                (symbol.upper(),)
            ).fetchone()
            if pending_row:
                logger.debug(f"[{symbol}] Cooldown: PENDING signal exists (id={pending_row[0]})")
                return True  # blocked

            # ── 2. Get last resolved signal to determine cooldown multiplier ──
            last_row = conn.execute(
                """SELECT outcome, t1_hit, t2_hit, t3_hit, sl_hit, sent_at
                   FROM signals WHERE symbol = ? AND outcome != 'PENDING'
                   ORDER BY sent_at DESC LIMIT 1""",
                (symbol.upper(),)
            ).fetchone()

            multiplier = 1.0
            if last_row:
                outcome, t1, t2, t3, sl, last_sent_at = last_row
                if sl:                        # SL_HIT → extra cooling off
                    multiplier = 2.0
                elif t1 and not t2 and not t3:  # Only T1 hit → cautious
                    multiplier = 1.5
                elif t2 or t3:                # T2/T3 hit → system's working, normal CD
                    multiplier = 1.0
                elif outcome == "EXPIRED":    # Expired → market moved, shorten CD
                    multiplier = 0.75

            effective_cooldown = cooldown_minutes * multiplier

            # ── 3. Time-based check using signal_cooldown table ──
            row = conn.execute(
                "SELECT sent_at FROM signal_cooldown WHERE symbol = ?",
                (symbol.upper(),)
            ).fetchone()
            if not row:
                return False  # No cooldown entry = can send

            sent_at = datetime.fromisoformat(row[0])
            elapsed_min = (datetime.utcnow() - sent_at).total_seconds() / 60
            blocked = elapsed_min < effective_cooldown

            if blocked:
                remaining = int(effective_cooldown - elapsed_min)
                logger.debug(
                    f"[{symbol}] Cooldown active: {remaining}min remaining "
                    f"(mult={multiplier:.1f}x, outcome={last_row[0] if last_row else 'none'})"
                )
            return blocked

        finally:
            conn.close()

    def set_cooldown(self, symbol: str, direction: str = "ANY"):
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO signal_cooldown (symbol, direction, sent_at) VALUES (?, ?, ?)",
                (symbol.upper(), direction, datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

    # ─── Accuracy & Stats ────────────────────────────────

    def was_sl_hit_recently(self, symbol: str, hours: int = 24) -> bool:
        """
        Returns True if this symbol had a SL_HIT outcome within the last `hours` hours.
        Used to raise confidence threshold before re-entering after a loss.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT id FROM signals
                   WHERE symbol = ? AND sl_hit = 1 AND sent_at > ?
                   ORDER BY sent_at DESC LIMIT 1""",
                (symbol.upper(), cutoff)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def get_accuracy_stats(self, days: int = 30) -> dict:
        """Calculate accuracy statistics for last N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM signals WHERE sent_at > ? AND outcome != 'PENDING'", (cutoff,)
            ).fetchall()
            signals = [self._signal_to_dict(r) for r in rows]

            if not signals:
                return {"total": 0, "win_rate": 0, "message": "Henüz veri yok"}

            total = len(signals)
            t1_hits = sum(1 for s in signals if s.get("t1_hit"))
            t2_hits = sum(1 for s in signals if s.get("t2_hit"))
            t3_hits = sum(1 for s in signals if s.get("t3_hit"))
            sl_hits = sum(1 for s in signals if s.get("sl_hit"))
            wins = sum(1 for s in signals if s.get("pnl_pct", 0) > 0)

            # Duration averages
            t1_durations = [s["t1_duration_min"] for s in signals if s.get("t1_duration_min")]
            t2_durations = [s["t2_duration_min"] for s in signals if s.get("t2_duration_min")]
            t3_durations = [s["t3_duration_min"] for s in signals if s.get("t3_duration_min")]

            # By tier breakdown
            tier_stats = {}
            for s in signals:
                tier = s.get("tier", "UNKNOWN")
                if tier not in tier_stats:
                    tier_stats[tier] = {"total": 0, "wins": 0, "t1": 0, "t2": 0, "t3": 0, "sl": 0}
                tier_stats[tier]["total"] += 1
                if s.get("pnl_pct", 0) > 0:
                    tier_stats[tier]["wins"] += 1
                if s.get("t1_hit"):
                    tier_stats[tier]["t1"] += 1
                if s.get("t2_hit"):
                    tier_stats[tier]["t2"] += 1
                if s.get("t3_hit"):
                    tier_stats[tier]["t3"] += 1
                if s.get("sl_hit"):
                    tier_stats[tier]["sl"] += 1

            # By confidence range
            conf_stats = {}
            for s in signals:
                conf = s.get("confidence", 0)
                bucket = f"{(conf // 10) * 10}-{(conf // 10) * 10 + 9}"
                if bucket not in conf_stats:
                    conf_stats[bucket] = {"total": 0, "wins": 0}
                conf_stats[bucket]["total"] += 1
                if s.get("pnl_pct", 0) > 0:
                    conf_stats[bucket]["wins"] += 1

            pnl_values = [s.get("pnl_pct", 0) for s in signals if s.get("pnl_pct") is not None]

            return {
                "total": total,
                "wins": wins,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                "t1_hits": t1_hits,
                "t1_rate": round(t1_hits / total * 100, 1) if total > 0 else 0,
                "t2_hits": t2_hits,
                "t2_rate": round(t2_hits / total * 100, 1) if total > 0 else 0,
                "t3_hits": t3_hits,
                "t3_rate": round(t3_hits / total * 100, 1) if total > 0 else 0,
                "sl_hits": sl_hits,
                "sl_rate": round(sl_hits / total * 100, 1) if total > 0 else 0,
                "avg_pnl": round(sum(pnl_values) / len(pnl_values), 2) if pnl_values else 0,
                "avg_t1_duration_min": round(sum(t1_durations) / len(t1_durations)) if t1_durations else None,
                "avg_t2_duration_min": round(sum(t2_durations) / len(t2_durations)) if t2_durations else None,
                "avg_t3_duration_min": round(sum(t3_durations) / len(t3_durations)) if t3_durations else None,
                "by_tier": tier_stats,
                "by_confidence": conf_stats,
            }
        finally:
            conn.close()

    def save_daily_stats(self, date: str, signals_sent: int,
                         crypto_signals: int = 0, bist_signals: int = 0):
        conn = self._get_conn()
        try:
            # Calculate accuracy for today
            accuracy = self.get_accuracy_stats(days=1)
            conn.execute(
                """INSERT OR REPLACE INTO daily_stats
                (date, signals_sent, crypto_signals, bist_signals,
                 accuracy_pct, t1_hit_rate, t2_hit_rate, t3_hit_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, signals_sent, crypto_signals, bist_signals,
                 accuracy.get("win_rate", 0), accuracy.get("t1_rate", 0),
                 accuracy.get("t2_rate", 0), accuracy.get("t3_rate", 0))
            )
            conn.commit()
        finally:
            conn.close()

    # ─── ML Model Storage ────────────────────────────────

    def save_ml_model(self, model_name: str, model_data: bytes,
                      feature_names: list, accuracy: float,
                      total_samples: int, metrics: dict = None):
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO ml_models
                (model_name, model_data, feature_names, accuracy, total_samples, trained_at, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (model_name, model_data, json.dumps(feature_names),
                 accuracy, total_samples, datetime.utcnow().isoformat(),
                 json.dumps(metrics) if metrics else None)
            )
            conn.commit()
        finally:
            conn.close()

    def get_latest_ml_model(self, model_name: str = "signal_predictor") -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """SELECT * FROM ml_models WHERE model_name = ?
                   ORDER BY trained_at DESC LIMIT 1""", (model_name,)
            ).fetchone()
            if not row:
                return None
            return {
                "id": row[0], "model_name": row[1], "model_data": row[2],
                "feature_names": json.loads(row[3]), "accuracy": row[4],
                "total_samples": row[5], "trained_at": row[6],
                "metrics": json.loads(row[7]) if row[7] else None,
            }
        finally:
            conn.close()

    # ─── Helpers ─────────────────────────────────────────

    @staticmethod
    def _alarm_to_dict(row) -> dict:
        return {
            "id": row[0], "user_id": row[1], "symbol": row[2],
            "target_price": row[3], "direction": row[4],
            "is_bist": bool(row[5]), "triggered": bool(row[6]),
            "created_at": row[7], "triggered_at": row[8],
        }

    @staticmethod
    def _signal_to_dict(row) -> dict:
        return {
            "id": row[0], "symbol": row[1], "direction": row[2],
            "tier": row[3], "confidence": row[4], "entry_price": row[5],
            "stop_loss": row[6], "target1": row[7], "target2": row[8],
            "target3": row[9], "rr": row[10], "is_crypto": bool(row[11]),
            "sent_at": row[12], "outcome": row[13],
            "t1_hit": bool(row[14]), "t1_hit_at": row[15], "t1_duration_min": row[16],
            "t2_hit": bool(row[17]), "t2_hit_at": row[18], "t2_duration_min": row[19],
            "t3_hit": bool(row[20]), "t3_hit_at": row[21], "t3_duration_min": row[22],
            "sl_hit": bool(row[23]), "sl_hit_at": row[24], "sl_duration_min": row[25],
            "max_favorable": row[26], "max_adverse": row[27],
            "exit_price": row[28], "pnl_pct": row[29], "closed_at": row[30],
            "features": json.loads(row[31]) if row[31] else None,
        }
