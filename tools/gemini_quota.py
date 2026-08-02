#!/usr/bin/env python3
"""
gemini_quota.py — cross-process quota limiter for the Gemini agent path.

Why this exists: RPM_LIMIT was a hardcoded 10 with a per-process sleep, which
(a) throttled a Tier-1 key ~100x below its ceiling and (b) could not coordinate
between concurrent agents, since quotas are per-project, not per-process.

Three limits are enforced together against a shared on-disk window:

    RPM  requests per minute
    TPM  tokens per minute      <- usually the binding one
    RPD  requests per day

TPM binds first in practice: at ~20k tokens/call, 2M TPM allows ~97 calls/min,
well under the 1000 RPM ceiling. Token cost per call also grows as an agent's
history accumulates, so the limiter reserves an estimate before each call and
reconciles against actual usage after.

Env overrides: GEMINI_RPM, GEMINI_TPM, GEMINI_RPD, GEMINI_QUOTA_SAFETY.
"""

import fcntl
import json
import os
import time
from datetime import date
from pathlib import Path

# Tier 1, Gemini 3.6 Flash (text-out). Per project, shared across all agents.
TIER1 = {"rpm": 1000, "tpm": 2_000_000, "rpd": 10_000}

# Stay this far under the published ceiling. A 429 costs a retry plus backoff,
# so leaving headroom is cheaper than reclaiming the last few percent.
DEFAULT_SAFETY = 0.80

# Fallback estimate for the first call, before any usage has been observed.
BOOTSTRAP_TOKENS = 8_000


class QuotaLimiter:
    def __init__(self, state_path=None, rpm=None, tpm=None, rpd=None, safety=None):
        s = float(os.environ.get("GEMINI_QUOTA_SAFETY", safety or DEFAULT_SAFETY))
        self.rpm = int((rpm or int(os.environ.get("GEMINI_RPM", TIER1["rpm"]))) * s)
        self.tpm = int((tpm or int(os.environ.get("GEMINI_TPM", TIER1["tpm"]))) * s)
        self.rpd = int((rpd or int(os.environ.get("GEMINI_RPD", TIER1["rpd"]))) * s)
        self.path = Path(state_path or os.environ.get(
            "GEMINI_QUOTA_STATE", "/tmp/rrma_gemini_quota.json"))
        self._recent = []  # this process's observed tokens/call, for estimation

    # -- shared state -------------------------------------------------------

    def _load(self, fh):
        fh.seek(0)
        raw = fh.read()
        if not raw.strip():
            return {"events": [], "day": str(date.today()), "day_count": 0}
        try:
            st = json.loads(raw)
        except json.JSONDecodeError:
            return {"events": [], "day": str(date.today()), "day_count": 0}
        if st.get("day") != str(date.today()):
            st = {"events": [], "day": str(date.today()), "day_count": 0}
        return st

    def _save(self, fh, st):
        fh.seek(0)
        fh.truncate()
        json.dump(st, fh)
        fh.flush()
        os.fsync(fh.fileno())

    def _estimate(self):
        if not self._recent:
            return BOOTSTRAP_TOKENS
        # Bias high: context grows, so the next call costs more than the mean.
        return int(max(self._recent[-5:]) * 1.1)

    # -- public API ---------------------------------------------------------

    def acquire(self, log=None):
        """Block until a call fits inside RPM, TPM and RPD. Returns waited secs."""
        est = self._estimate()
        waited = 0.0
        while True:
            self.path.touch(exist_ok=True)
            with open(self.path, "r+") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    st = self._load(fh)
                    now = time.time()
                    st["events"] = [e for e in st["events"] if now - e[0] < 60.0]

                    if st["day_count"] >= self.rpd:
                        raise RuntimeError(
                            f"daily request quota exhausted "
                            f"({st['day_count']}/{self.rpd} incl. safety margin)")

                    calls = len(st["events"])
                    tokens = sum(e[1] for e in st["events"])
                    over_rpm = calls + 1 > self.rpm
                    over_tpm = tokens + est > self.tpm

                    if not (over_rpm or over_tpm):
                        st["events"].append([now, est])
                        st["day_count"] += 1
                        self._save(fh, st)
                        return waited

                    oldest = min(e[0] for e in st["events"])
                    sleep_for = max(0.05, 60.0 - (now - oldest))
                    reason = "TPM" if over_tpm else "RPM"
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)

            if log:
                log(f"  quota: {reason} ceiling reached "
                    f"({calls} calls / {tokens:,} tok in window), waiting {sleep_for:.1f}s")
            time.sleep(sleep_for)
            waited += sleep_for

    def record(self, actual_tokens):
        """Reconcile the reservation against what the call actually cost."""
        if not actual_tokens:
            return
        self._recent.append(actual_tokens)
        est = self._estimate()
        delta = actual_tokens - est
        if abs(delta) < 1:
            return
        self.path.touch(exist_ok=True)
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                st = self._load(fh)
                if st["events"]:
                    st["events"][-1][1] = actual_tokens
                    self._save(fh, st)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def status(self):
        self.path.touch(exist_ok=True)
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                st = self._load(fh)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        now = time.time()
        ev = [e for e in st["events"] if now - e[0] < 60.0]
        return (f"quota: {len(ev)}/{self.rpm} rpm  "
                f"{sum(e[1] for e in ev):,}/{self.tpm:,} tpm  "
                f"{st['day_count']}/{self.rpd} rpd")
