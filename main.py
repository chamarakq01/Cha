"""
Sic Bo ML Prediction Backend v3
Signals 12 + XGBoost ensemble + Neural voting
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from collections import defaultdict

# XGBoost optional — graceful fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

app = FastAPI(title="Sic Bo ML API v3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class PredictRequest(BaseModel):
    history: List[int]
    top_n: int = 7

class PredictResponse(BaseModel):
    predictions: List[dict]
    meta: dict

def clamp(n): return 4 <= n <= 17

FAMILIES = [
    [5,10,15],[3,6,9,12],[4,14],[8,13],
    [6,16],[9,11,13],[7,14,17],[12,15],
]

# ════════════════════════════════════════════════════
# FEATURE ENGINEERING — build feature vector for each
# (context → next) pair for ML training
# ════════════════════════════════════════════════════
def build_features(seq, i, window=8):
    """Build feature vector at position i (predicting seq[i+1])"""
    feats = []
    # Last 8 values (padded with 0)
    for w in range(window, 0, -1):
        feats.append(seq[i-w] if i>=w else 0)
    cur = seq[i]
    feats.append(cur)
    # Diffs
    for w in range(1, 5):
        feats.append(seq[i]-seq[i-w] if i>=w else 0)
    # Small/Big/Odd/Even of current
    feats.append(1 if cur<=10 else 0)
    feats.append(cur % 2)
    # Family membership
    for fam in FAMILIES:
        feats.append(1 if cur in fam else 0)
    # Rolling mean/std of last 10
    window10 = [seq[max(0,i-j)] for j in range(10)]
    feats.append(np.mean(window10))
    feats.append(np.std(window10) if len(window10)>1 else 0)
    # Diff trend
    diffs = [seq[i-j]-seq[i-j-1] for j in range(min(4,i)) if i-j-1>=0]
    feats.append(np.mean(diffs) if diffs else 0)
    feats.append(diffs[0] if diffs else 0)  # most recent diff
    return feats

def train_xgb(seq):
    """Train XGBoost model on sequence data — multiclass 4-17"""
    N = len(seq)
    if N < 30: return None
    X, y = [], []
    for i in range(8, N-1):
        if clamp(seq[i]) and clamp(seq[i+1]):
            X.append(build_features(seq, i))
            y.append(seq[i+1] - 4)  # 0-13 classes
    if len(X) < 20: return None
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    model = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.15,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
        num_class=14,
    )
    model.fit(X, y)
    return model

def train_simple_nn(seq):
    """Simple 2-layer neural net using pure NumPy"""
    N = len(seq)
    if N < 30: return None

    X, y = [], []
    for i in range(8, N-1):
        if clamp(seq[i]) and clamp(seq[i+1]):
            X.append(build_features(seq, i))
            # One-hot target for output 4-17
            tgt = np.zeros(14)
            tgt[seq[i+1]-4] = 1.0
            y.append(tgt)

    if len(X) < 20: return None

    X  = np.array(X, dtype=np.float32)
    y  = np.array(y, dtype=np.float32)

    # Normalize X
    mu  = X.mean(axis=0); sig = X.std(axis=0)+1e-8
    X   = (X-mu)/sig

    n_in  = X.shape[1]
    n_hid = 32
    n_out = 14

    # Xavier init
    np.random.seed(42)
    W1 = np.random.randn(n_in, n_hid) * np.sqrt(2/n_in)
    b1 = np.zeros(n_hid)
    W2 = np.random.randn(n_hid, n_out) * np.sqrt(2/n_hid)
    b2 = np.zeros(n_out)

    def relu(z): return np.maximum(0, z)
    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    lr = 0.008
    for _ in range(120):  # 120 epochs
        # Forward
        h   = relu(X @ W1 + b1)
        out = softmax(h @ W2 + b2)
        # Cross-entropy loss grad
        dL  = (out - y) / len(X)
        # Backward W2
        dW2 = h.T @ dL
        db2 = dL.sum(axis=0)
        # Backward W1
        dh  = dL @ W2.T * (h > 0)
        dW1 = X.T @ dh
        db1 = dh.sum(axis=0)
        # Update
        W1 -= lr*dW1; b1 -= lr*db1
        W2 -= lr*dW2; b2 -= lr*db2

    return {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"mu":mu,"sig":sig}

def nn_predict(model, feats):
    """Run trained NN forward pass"""
    x = np.array(feats, dtype=np.float32)
    x = (x - model["mu"]) / model["sig"]
    h = np.maximum(0, x @ model["W1"] + model["b1"])
    e = np.exp(h @ model["W2"] + model["b2"])
    return e / e.sum()

# ════════════════════════════════════════════════════
# MAIN PREDICT
# ════════════════════════════════════════════════════
def predict(history, top_n=7):
    if len(history) < 20:
        return [], {"error": "Need 20+ rounds"}

    seq   = list(reversed(history))
    N     = len(seq)
    last  = history[0]
    prev2 = history[1] if len(history)>1 else last
    prev3 = history[2] if len(history)>2 else prev2
    prev4 = history[3] if len(history)>3 else prev3

    d1=last-prev2; d2=prev2-prev3; d3=prev3-prev4
    scores = defaultdict(float)

    # ── Statistical signals (same as v2) ──

    # S1: Markov (w=5)
    K=14; idx={s:s-4 for s in range(4,18)}; ridx={v:k for k,v in idx.items()}
    mat=np.ones((K,K))*0.3
    for i in range(N-1):
        if clamp(seq[i]) and clamp(seq[i+1]):
            mat[idx[seq[i]],idx[seq[i+1]]]+=1.0
    mat=mat/mat.sum(axis=1,keepdims=True)
    if clamp(last):
        for j in range(K): scores[ridx[j]]+=mat[idx[last],j]*100*5

    # S2: 3-gram (w=6)
    k3=(prev3,prev2,last); mem3=defaultdict(float); m3=0
    for i in range(3,N):
        if (seq[i-3],seq[i-2],seq[i-1])==k3 and clamp(seq[i]):
            mem3[seq[i]]+=0.97**(N-1-i)*2; m3+=1
    if m3:
        mx=max(mem3.values())
        for s,v in mem3.items(): scores[s]+=(v/mx)*100*6

    # S3: 2-gram (w=4)
    k2=(prev2,last); mem2=defaultdict(float); m2=0
    for i in range(2,N):
        if (seq[i-2],seq[i-1])==k2 and clamp(seq[i]):
            mem2[seq[i]]+=0.97**(N-1-i); m2+=1
    if m2:
        mx=max(mem2.values())
        for s,v in mem2.items(): scores[s]+=(v/mx)*100*4

    # S4: 4-gram (w=8)
    k4=(prev4,prev3,prev2,last); mem4=defaultdict(float); m4=0
    for i in range(4,N):
        if (seq[i-4],seq[i-3],seq[i-2],seq[i-1])==k4 and clamp(seq[i]):
            mem4[seq[i]]+=0.97**(N-1-i)*3; m4+=1
    if m4:
        mx=max(mem4.values())
        for s,v in mem4.items(): scores[s]+=(v/mx)*100*8

    # S5: Delta pattern (w=5)
    dm=defaultdict(float); dmc=0
    for i in range(4,N):
        if seq[i-3]-seq[i-4]==d3 and seq[i-2]-seq[i-3]==d2 and seq[i-1]-seq[i-2]==d1 and clamp(seq[i]):
            dm[seq[i]]+=0.97**(N-1-i)*2; dmc+=1
    if dmc:
        mx=max(dm.values())
        for s,v in dm.items(): scores[s]+=(v/mx)*100*5
    else:
        for i in range(3,N):
            if seq[i-2]-seq[i-3]==d2 and seq[i-1]-seq[i-2]==d1 and clamp(seq[i]):
                scores[seq[i]]+=0.97**(N-1-i)*15

    # S6: Center weight (w=3)
    for s in range(4,18): scores[s]+=max(0,10-abs(s-10.5))*3

    # S7: Arithmetic (w=3)
    ds=d1-d2
    arith=last+(d1+1) if d2==d3+1 and d1==d2+1 else last+d1 if d1==d2 else last+d1+ds
    if clamp(arith): scores[arith]+=100*3

    # S8: Fuzzy window (w=8)
    win_sc=defaultdict(float); wm=0; wb=0
    for W in [5,4,3]:
        if N<W+1: continue
        cw=seq[N-W:N]; found=False
        for i in range(W,N):
            hw=seq[i-W:i]; nxt=seq[i]
            if not clamp(nxt): continue
            fz=sum(1.0 if cw[j]==hw[j] else 0.6 if abs(cw[j]-hw[j])==1 else 0 for j in range(W))/W
            if fz>=0.7:
                win_sc[nxt]+=0.97**(N-1-i)*fz*2; wm+=1; found=True
        if found:
            wb=W; mx=max(win_sc.values())
            for s,v in win_sc.items(): scores[s]+=(v/mx)*100*8
            break

    # S9: Sub-sequence (w=6)
    sub_sc=defaultdict(float); sm=0
    rn=[x for x in [prev4,prev3,prev2,last] if clamp(x)]
    for a,b in [(rn[i],rn[i+1]) for i in range(len(rn)-1)]:
        for i in range(1,N-1):
            if seq[i-1]==a and seq[i]==b and clamp(seq[i+1]):
                sub_sc[seq[i+1]]+=0.97**(N-1-i)*1.5; sm+=1
    if sm:
        mx=max(sub_sc.values())
        for s,v in sub_sc.items(): scores[s]+=(v/mx)*100*6

    # S10: Family boost (w=7)
    fam_hits=set()
    for fam in FAMILIES:
        fs=set(fam)
        if last not in fs: continue
        tr=hi=0; fn=defaultdict(int)
        for i in range(N-1):
            if seq[i] in fs:
                tr+=1
                if seq[i+1] in fs and clamp(seq[i+1]): hi+=1; fn[seq[i+1]]+=1
        if tr<3 or hi/tr<0.15: continue
        hr=hi/tr; mx=max(fn.values()) if fn else 1
        for n,cnt in fn.items():
            if n!=last and clamp(n): scores[n]+=hr*100*7*(cnt/mx); fam_hits.add(n)

    # S11: Exponential recency (w=9)
    for i in range(N-1):
        if seq[i]==last and clamp(seq[i+1]):
            scores[seq[i+1]]+=np.exp(-0.015*(N-2-i))*9

    # S12: Recent hot (w=4)
    r30=seq[max(0,N-30):]; f30=defaultdict(int)
    for s in r30:
        if clamp(s): f30[s]+=1
    if f30:
        mx=max(f30.values())
        for s,v in f30.items(): scores[s]+=(v/mx)*20

    # ── ML Models ──
    ml_used=[]

    # XGBoost (w=12) — highest weight
    xgb_probs=None
    if XGB_AVAILABLE and N>=30:
        try:
            xgb_model=train_xgb(seq)
            if xgb_model:
                feat=build_features(seq, N-1)
                feat_arr=np.array([feat],dtype=np.float32)
                probs=xgb_model.predict_proba(feat_arr)[0]
                xgb_probs=probs
                for j in range(14):
                    s=j+4
                    if clamp(s): scores[s]+=probs[j]*100*12
                ml_used.append("XGBoost")
        except Exception as e:
            pass

    # Neural Net (w=10)
    nn_probs=None
    if N>=30:
        try:
            nn_model=train_simple_nn(seq)
            if nn_model:
                feat=build_features(seq, N-1)
                probs=nn_predict(nn_model,feat)
                nn_probs=probs
                for j in range(14):
                    s=j+4
                    if clamp(s): scores[s]+=probs[j]*100*10
                ml_used.append("NeuralNet")
        except Exception as e:
            pass

    # ── Normalize & rank ──
    # Family guarantee
    guaranteed=set()
    for fam in FAMILIES:
        if last in fam:
            for n in fam:
                if n!=last and clamp(n): guaranteed.add(n)

    all_sc={s:scores[s] for s in range(4,18)}
    mx_sc=max(all_sc.values()) if all_sc else 1

    ranked=[]
    for s in range(4,18):
        conf=round(all_sc[s]/mx_sc*100)
        ranked.append({
            "num":s, "conf":conf,
            "small":s<=10, "odd":s%2==1,
            "in_family":s in fam_hits,
            "guaranteed":s in guaranteed,
            "in_4gram":s in mem4,
            "in_3gram":s in mem3,
            "in_window":s in win_sc,
            "is_arith":clamp(arith) and arith==s,
            "ml_boost": (
                (xgb_probs[s-4] if xgb_probs is not None else 0) +
                (nn_probs[s-4]  if nn_probs  is not None else 0)
            ),
        })
    ranked.sort(key=lambda x:-x["conf"])

    # Guarantee family members included
    top_nums={r["num"] for r in ranked[:top_n]}
    for g in guaranteed:
        if g not in top_nums:
            ranked.append({
                "num":g,"conf":15,"small":g<=10,"odd":g%2==1,
                "in_family":True,"guaranteed":True,"in_4gram":False,
                "in_3gram":False,"in_window":False,"is_arith":False,"ml_boost":0
            })
    ranked.sort(key=lambda x:-x["conf"])

    tr=sum(1 for i in range(N-1) if seq[i]==last)
    hi=sum(1 for i in range(N-1) if seq[i]==last and any(r["num"]==seq[i+1] for r in ranked[:top_n]))
    hr=round(hi/tr*100) if tr>0 else 0

    meta={
        "rounds":N,"last":last,"seq":f"{prev3}→{prev2}→{last}",
        "diffs":f"{d3:+},{d2:+},{d1:+}","arith_pred":arith if clamp(arith) else None,
        "m4":m4,"m3":m3,"m2":m2,"delta":dmc,"win":wm,"win_best":wb,"sub":sm,
        "hit_rate":hr,"signals":12,"ml_models":ml_used,
        "xgb":XGB_AVAILABLE,
    }
    return ranked[:top_n], meta

@app.get("/")
def root():
    return {"status":"Sic Bo ML API v3","signals":12,"xgb":XGB_AVAILABLE}

@app.post("/predict")
def predict_route(req:PredictRequest):
    p,m=predict(req.history,req.top_n)
    return PredictResponse(predictions=p,meta=m)

@app.get("/health")
def health(): return {"ok":True}
