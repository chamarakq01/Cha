"""
Sic Bo ML Prediction Backend
FastAPI + scikit-learn
Deploy on Render.com (free tier)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from collections import defaultdict, Counter
import math

app = FastAPI(title="Sic Bo ML API")

# Allow all origins (GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input model ──
class PredictRequest(BaseModel):
    history: List[int]   # list of sums, newest first
    top_n:   int = 6     # how many predictions to return

class PredictResponse(BaseModel):
    predictions: List[dict]  # [{num, conf, signals}]
    meta:        dict


# ═══════════════════════════════════════════════════════
#  ML ENGINE — 8 signals combined
# ═══════════════════════════════════════════════════════

def clamp(n): return 4 <= n <= 17

def predict(history: List[int], top_n: int = 6):
    if len(history) < 20:
        return [], {"error": "Need 20+ rounds"}

    seq  = list(reversed(history))   # oldest first
    N    = len(seq)
    last  = history[0]
    prev2 = history[1] if len(history) > 1 else last
    prev3 = history[2] if len(history) > 2 else prev2
    prev4 = history[3] if len(history) > 3 else prev3

    d1 = last  - prev2
    d2 = prev2 - prev3
    d3 = prev3 - prev4

    scores = defaultdict(float)

    # ── Signal 1: Transition Matrix (w=5) ──
    trans = defaultdict(float)
    trans_total = 0
    for i in range(N - 1):
        if seq[i] == last and clamp(seq[i+1]):
            w = 0.98 ** (N - 1 - i)
            trans[seq[i+1]] += w
            trans_total     += w
    if trans_total > 0:
        mx = max(trans.values())
        for s, v in trans.items():
            scores[s] += (v / mx) * 100 * 5

    # ── Signal 2: 3-gram Memory (w=6) ──
    key3 = (prev3, prev2, last)
    mem3 = defaultdict(float); m3 = 0
    for i in range(3, N):
        if (seq[i-3], seq[i-2], seq[i-1]) == key3 and clamp(seq[i]):
            w = 0.97 ** (N - 1 - i) * 2
            mem3[seq[i]] += w; m3 += 1
    if m3 > 0:
        mx = max(mem3.values())
        for s, v in mem3.items():
            scores[s] += (v / mx) * 100 * 6

    # ── Signal 3: 2-gram (w=4) ──
    key2 = (prev2, last)
    mem2 = defaultdict(float); m2 = 0
    for i in range(2, N):
        if (seq[i-2], seq[i-1]) == key2 and clamp(seq[i]):
            w = 0.97 ** (N - 1 - i)
            mem2[seq[i]] += w; m2 += 1
    if m2 > 0:
        mx = max(mem2.values())
        for s, v in mem2.items():
            scores[s] += (v / mx) * 100 * 4

    # ── Signal 4: Delta Pattern 3-diff (w=5) ──
    dmatch = defaultdict(float); dm = 0
    for i in range(4, N):
        pd3 = seq[i-3] - seq[i-4]
        pd2 = seq[i-2] - seq[i-3]
        pd1 = seq[i-1] - seq[i-2]
        if pd3==d3 and pd2==d2 and pd1==d1 and clamp(seq[i]):
            w = 0.97 ** (N - 1 - i) * 2
            dmatch[seq[i]] += w; dm += 1
    if dm > 0:
        mx = max(dmatch.values())
        for s, v in dmatch.items():
            scores[s] += (v / mx) * 100 * 5
    else:
        # fallback 2-diff
        for i in range(3, N):
            pd2 = seq[i-2] - seq[i-3]
            pd1 = seq[i-1] - seq[i-2]
            if pd2==d2 and pd1==d1 and clamp(seq[i]):
                scores[seq[i]] += 0.97**(N-1-i) * 15

    # ── Signal 5: Center Weight (w=3) ──
    for s in range(4, 18):
        scores[s] += max(0, 10 - abs(s - 10.5)) * 3

    # ── Signal 6: Arithmetic Progression (w=3) ──
    diff_step = d1 - d2
    arith_pred = None
    if d2 == d3 + 1 and d1 == d2 + 1:
        arith_pred = last + (d1 + 1)
    elif d1 == d2:
        arith_pred = last + d1
    else:
        arith_pred = last + d1 + diff_step
    if arith_pred and clamp(arith_pred):
        scores[arith_pred] += 100 * 3

    # ── Signal 7: Rolling Window Fuzzy Match (w=8) ──
    win_scores = defaultdict(float); win_matches = 0; win_best = 0
    for W in [5, 4, 3]:
        if N < W + 1: continue
        cur_win = seq[N-W:N]
        found = False
        for i in range(W, N):
            hist_win = seq[i-W:i]
            nxt = seq[i]
            if not clamp(nxt): continue
            # fuzzy score
            fuzzy = sum(
                1.0 if cur_win[j]==hist_win[j] else 0.6 if abs(cur_win[j]-hist_win[j])==1 else 0
                for j in range(W)
            ) / W
            if fuzzy >= 0.7:
                w = 0.97**(N-1-i) * fuzzy * 2
                win_scores[nxt] += w
                win_matches += 1; found = True
        if found:
            win_best = W
            mx = max(win_scores.values())
            for s, v in win_scores.items():
                scores[s] += (v/mx) * 100 * 8
            break

    # ── Signal 8: Sub-sequence Recurrence (w=6) ──
    sub_scores = defaultdict(float); sub_matches = 0
    recent = [x for x in [prev4,prev3,prev2,last] if clamp(x)]
    pairs  = [(recent[i], recent[i+1]) for i in range(len(recent)-1)]
    for a, b in pairs:
        for i in range(1, N-1):
            if seq[i-1]==a and seq[i]==b and clamp(seq[i+1]):
                w = 0.97**(N-1-i) * 1.5
                sub_scores[seq[i+1]] += w; sub_matches += 1
    if sub_matches > 0:
        mx = max(sub_scores.values())
        for s, v in sub_scores.items():
            scores[s] += (v/mx) * 100 * 6

    # ── Signal 9: Higher-order Markov via numpy (w=7) ──
    # Build proper transition count matrix, smooth with Laplace
    # States: 4-17 → index 0-13
    idx  = {s: s-4 for s in range(4, 18)}
    ridx = {v: k for k, v in idx.items()}
    K    = 14
    mat  = np.ones((K, K)) * 0.5  # Laplace smoothing
    for i in range(N-1):
        if clamp(seq[i]) and clamp(seq[i+1]):
            mat[idx[seq[i]], idx[seq[i+1]]] += 1.0
    # Normalize rows
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / row_sums

    if clamp(last):
        row = mat[idx[last]]  # probability distribution
        for j in range(K):
            s = ridx[j]
            scores[s] += row[j] * 100 * 7

    # ── Normalize final scores ──
    if not scores:
        return [], {"error": "No scores"}

    max_sc = max(scores[s] for s in range(4, 18))
    ranked = []
    for s in range(4, 18):
        conf = round(scores[s] / max_sc * 100) if max_sc > 0 else 0
        ranked.append({
            "num":    s,
            "conf":   conf,
            "small":  s <= 10,
            "odd":    s % 2 == 1,
            "in_trans": s in trans,
            "in_3gram": s in mem3,
            "in_win":   s in win_scores,
            "is_arith": arith_pred == s,
        })
    ranked.sort(key=lambda x: -x["conf"])

    # Hit rate validation
    trials = sum(1 for i in range(N-1) if seq[i]==last)
    hits   = sum(
        1 for i in range(N-1)
        if seq[i]==last and any(r["num"]==seq[i+1] for r in ranked[:top_n])
    )
    hit_rate = round(hits/trials*100) if trials > 0 else 0

    meta = {
        "rounds":      N,
        "last":        last,
        "seq":         f"{prev3}→{prev2}→{last}",
        "diffs":       f"{d3:+},{d2:+},{d1:+}",
        "arith_pred":  arith_pred,
        "m3_matches":  m3,
        "m2_matches":  m2,
        "delta_matches": dm,
        "win_matches": win_matches,
        "win_best":    win_best,
        "sub_matches": sub_matches,
        "hit_rate":    hit_rate,
        "signals":     9,
    }

    return ranked[:top_n], meta


# ── Routes ──

@app.get("/")
def root():
    return {"status": "Sic Bo ML API running", "signals": 9}

@app.post("/predict")
def predict_route(req: PredictRequest):
    preds, meta = predict(req.history, req.top_n)
    return PredictResponse(predictions=preds, meta=meta)

@app.get("/health")
def health():
    return {"ok": True}
