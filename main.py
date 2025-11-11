# main.py
from __future__ import annotations
import os, re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


from fastapi.middleware.cors import CORSMiddleware



# Config & data paths
OUTDIR = Path(os.getenv("STACKREC_OUTDIR", "out_cleaned"))
KW_CSV = OUTDIR / "stack_tech_keywords_from_map.csv"
META_CSV = OUTDIR / "stack_scores_per_stack_year.csv"

app = FastAPI(
    title="Stack Recommender API",
    version="1.0.0",
    description="Simple APIs to recommend stacks and frameworks for new entrants and professionals.",
)

# Allow all origins/methods/headers during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],         
    allow_headers=["*"],
)

# Utilities – normalization & variants (keep consistent with your notebook)
_ALIAS = {
    "js": "javascript", "ts": "typescript", "py": "python", "rb": "ruby",
    "postgres": "postgresql", "mongo": "mongodb", "tf": "tensorflow",
    "sklearn": "scikit-learn", "k8s": "kubernetes", "gcp": "google cloud",
    "rn": "react native", "next.js": "nextjs", "react.js": "react", "node.js": "node",
}

def _norm_list(xs) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, (list, tuple, set)):
        it = xs
    else:
        it = [xs]
    out = []
    for x in it:
        t = str(x).strip().lower()
        if not t:
            continue
        out.append(_ALIAS.get(t, t))
    return out

def token_variants(tok: str) -> set[str]:
    t = tok.strip().lower()
    variants = {t}
    n = re.sub(r"[^\w#\+]", "", t)
    variants.add(n)
    if t.endswith(".js"):
        base = t[:-3]
        variants.add(base)
        variants.add(base + "js")
    if t == "c#": variants.add("csharp")
    if t == "f#": variants.add("fsharp")
    variants.add(t.replace(".", ""))
    variants.add(re.sub(r"\(.*?\)", "", t).strip())
    return {v for v in variants if v}


# Load data at startup
if not KW_CSV.exists():
    raise RuntimeError(f"{KW_CSV} not found. Run your notebook to generate it.")

df_kw = pd.read_csv(KW_CSV)
df_kw["keywords"] = df_kw["keywords"].fillna("").astype(str)

# token frequency for learn-next ranking
_all_tokens = [t for kws in df_kw["keywords"] for t in str(kws).split() if t]
_token_freq = Counter(_all_tokens)

# Latest per-stack meta from survey
if META_CSV.exists():
    meta = pd.read_csv(META_CSV)
    latest = meta.loc[meta.groupby("stack")["year"].idxmax()].set_index("stack")
else:
    latest = pd.DataFrame(index=df_kw["stack"].unique())

# add safe normalized columns
for col in ["A_t", "BarrierScore", "StackScore"]:
    if col not in latest.columns:
        latest[col] = 0.0

def _norm_col(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0 * s

latest["A_t_norm"] = _norm_col(latest["A_t"])
latest["BarrierScore_norm"] = _norm_col(latest["BarrierScore"])
latest["StackScore_norm"] = _norm_col(latest["StackScore"])

META = latest  

# Build quick lookup
_stack_to_tokens: Dict[str, set[str]] = {
    r.stack: set(str(r.keywords).split()) for _, r in df_kw.iterrows()
}
_all_stacks: List[str] = df_kw["stack"].dropna().unique().tolist()
_stack_allow = defaultdict(set)
for _, r in df_kw.iterrows():
    stk = r["stack"]
    for t in str(r["keywords"]).split():
        _stack_allow[stk].add(t.lower())

def _guess_category_for_stack(stk: str) -> str:
    return stk

_CATEGORY_STOP = {
    "AI/ML": {"css", "html", "javascript", "jquery", "node", "node.js", "php"},
    "Systems": {"css", "html", "javascript", "react", "vue", "angular"},
    "GameDev": {"css", "html", "jquery"},
    "Mobile": {"jquery", "php"},
}

INTEREST_TO_STACK = {
    # Web
    "web": "Web", "frontend": "Web", "front-end": "Web", "backend": "Web", "back-end": "Web",
    "fullstack": "Web", "full-stack": "Web",
    "react": "Web", "angular": "Web", "vue": "Web", "nextjs": "Web", "node": "Web",
    "express": "Web", "django": "Web", "flask": "Web", "fastapi": "Web",
    # Mobile
    "mobile": "Mobile", "android": "Mobile", "ios": "Mobile", "swift": "Mobile",
    "kotlin": "Mobile", "flutter": "Mobile", "react native": "Mobile",
    # Cloud / DevOps
    "cloud": "Cloud", "devops": "Cloud", "sre": "Cloud", "kubernetes": "Cloud",
    "docker": "Cloud", "terraform": "Cloud", "aws": "Cloud", "azure": "Cloud",
    "google cloud": "Cloud", "gcp": "Cloud",
    # Data / AI
    "data": "AI/ML", "data engineering": "AI/ML", "data science": "AI/ML",
    "machine learning": "AI/ML", "ml": "AI/ML", "deep learning": "AI/ML", "nlp": "AI/ML",
    # Security
    "security": "Security", "cybersecurity": "Security", "pentest": "Security",
    # Game
    "gamedev": "GameDev", "game": "GameDev", "unreal": "GameDev", "unity": "GameDev", "godot": "GameDev",
    # Desktop / Systems
    "desktop": "Desktop", "systems": "Systems", "system": "Systems", "embedded": "Systems", "firmware": "Systems",
    "rust": "Systems", "c": "Systems", "c++": "Systems", "go": "Systems",
    # Database / Viz / BI
    "database": "Database", "sql": "Database", "bi": "Visualization", "visualization": "Visualization",
    "analytics": "Visualization",
    # Blockchain
    "blockchain": "Blockchain", "web3": "Blockchain", "solidity": "Blockchain",
}

def _interest_buckets(interests) -> List[str]:
    toks = _norm_list(interests if isinstance(interests, (list, tuple, set)) else [interests])
    buckets = []
    for t in toks:
        if t in INTEREST_TO_STACK:
            buckets.append(INTEREST_TO_STACK[t])
        else:
            for k, stk in INTEREST_TO_STACK.items():
                if k in t:
                    buckets.append(stk)
    # dedupe preserve order
    seen, out = set(), []
    for b in buckets:
        if b not in seen:
            out.append(b); seen.add(b)
    return out

def suggest_techs_to_learn(stack: str, known_tokens=None, k: int = 6) -> List[str]:
    known = {t.lower() for t in (known_tokens or []) if t}
    kws = str(df_kw.loc[df_kw["stack"] == stack, "keywords"].squeeze() or "").split()
    allow = _stack_allow.get(stack, set())
    cand = [t for t in kws if t and t.lower() in allow and t.lower() not in known]
    # category stoplist
    cat = _guess_category_for_stack(stack)
    stop = _CATEGORY_STOP.get(cat, set())
    cand = [t for t in cand if t.lower() not in stop]
    scored = [(t, _token_freq.get(t, 0)) for t in cand]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [t for t, _ in scored[:k]]


_AI_CORE = {"python", "pytorch", "tensorflow", "scikit-learn", "numpy", "pandas", "matplotlib"}

def _boost_learn_next_if_py_ml(stack: str, langs_norm: set[str], interests_norm: List[str], items: List[str]) -> List[str]:
    if stack == "AI/ML" and ("python" in langs_norm or "ml" in interests_norm or "machine learning" in interests_norm):
        uniq = []
        seen = set()
        for t in list(_AI_CORE) + items:
            if t not in seen:
                uniq.append(t); seen.add(t)
        return uniq[:6]
    return items


def recommend_new_entrant_rule(
    languages_known: List[str],
    areas_of_interest,
    topk: int = 5,
    entrant_boost: bool = False
) -> tuple[pd.DataFrame, List[str]]:
    """
    Returns:
        (df, frameworks_by_language)
        - df: top stacks with scores & learn_next
        - frameworks_by_language: ordered list of frameworks inferred from languages_known (+ interest)
    """
    langs = set(_norm_list(languages_known))
    buckets = set(_interest_buckets(areas_of_interest))
    rows = []

    for stack in _all_stacks:
        kws = _stack_to_tokens.get(stack, set())
        overlap = len(kws & langs)
        A = float(META.loc[stack, "A_t_norm"]) if stack in META.index else 0.0
        B = float(META.loc[stack, "BarrierScore_norm"]) if stack in META.index else 0.0

        score = 2.0 * (stack in buckets) + 1.0 * min(overlap, 3) + 0.5 * A - 0.5 * B

        reasons = []
        if stack in buckets: reasons.append("interest match")
        if overlap: reasons.append(f"{overlap} overlap")
        if A > 0: reasons.append("popular")
        if B <= 0.4: reasons.append("lower barrier")

        learn_next = suggest_techs_to_learn(stack, known_tokens=langs, k=6)
        if entrant_boost:
            learn_next = _boost_learn_next_if_py_ml(stack, langs, _norm_list(areas_of_interest), learn_next)

        rows.append({
            "stack": stack,
            "score": float(score),
            "overlap": int(overlap),
            "A_t_norm": round(A, 3),
            "BarrierScore_norm": round(B, 3),
            "reason": "; ".join(reasons) if reasons else "general fit",
            "learn_next": learn_next,
            "top_techs": learn_next[:3],
        })

    df = (pd.DataFrame(rows)
          .sort_values(["score", "overlap", "A_t_norm"], ascending=[False, False, False])
          .head(topk)
          .reset_index(drop=True))

    frameworks = recommend_frameworks_by_language(
        languages_known=languages_known,
        interests=areas_of_interest,
        topk=12,
    )

    return df, frameworks


def recommend_professional_rule(
    all_skills: List[str],
    topk: int = 5,
    prefer_low_barrier: bool = False
) -> tuple[pd.DataFrame, List[str]]:
    """
    Returns:
        (df, frameworks_by_language)
        - df: top stacks with scores & learn_next
        - frameworks_by_language: ordered list of frameworks inferred from all_skills (treated as languages/tools)
    """
    known = set(_norm_list(all_skills))
    rows = []

    for stack in _all_stacks:
        kws = _stack_to_tokens.get(stack, set())
        overlap = len(kws & known)
        A = float(META.loc[stack, "A_t_norm"]) if stack in META.index else 0.0
        S = float(META.loc[stack, "StackScore_norm"]) if stack in META.index else 0.0
        B = float(META.loc[stack, "BarrierScore_norm"]) if stack in META.index else 0.0

        score = 1.1 * min(overlap, 5) + 0.6 * A + 0.4 * S - (0.4 * B if prefer_low_barrier else 0.0)

        reasons = []
        if overlap: reasons.append(f"{overlap} overlap")
        if A > 0: reasons.append("popular")
        if S > 0.6: reasons.append("rising/desired")
        if prefer_low_barrier and B > 0.5: reasons.append("penalized barrier")

        learn_next = suggest_techs_to_learn(stack, known_tokens=known, k=6)

        rows.append({
            "stack": stack,
            "score": float(score),
            "overlap": int(overlap),
            "A_t_norm": round(A, 3),
            "StackScore_norm": round(S, 3),
            "BarrierScore_norm": round(B, 3),
            "reason": "; ".join(reasons) if reasons else "general fit",
            "learn_next": learn_next,
            "top_techs": learn_next[:3],
        })

    df = (pd.DataFrame(rows)
          .sort_values(["score", "overlap", "A_t_norm", "StackScore_norm"], ascending=[False, False, False, False])
          .head(topk)
          .reset_index(drop=True))

    frameworks = recommend_frameworks_by_language(
        languages_known=all_skills,
        interests=None,
        topk=12,
    )

    return df, frameworks


# Framework recommender (language/interest aligned)
_LANG_TO_FRAMEWORKS = {
    "python": ["django", "fastapi", "flask", "streamlit"],
    "javascript": ["react", "nextjs", "express", "angular", "vue", "svelte", "nestjs"],
    "typescript": ["react", "nextjs", "nestjs", "angular", "vue"],
    "java": ["spring", "spring boot", "quarkus", "micronaut"],
    "c#": ["asp.net", "asp.net core", "blazor"],
    "php": ["laravel", "symfony", "codeigniter"],
    "ruby": ["rails", "sinatra"],
    "go": ["gin", "fiber", "echo"],
}

def recommend_frameworks_by_language(languages_known: List[str], interests: Optional[str | List[str]] = None, topk: int = 12) -> List[str]:
    langs = _norm_list(languages_known)
    baskets = set(_interest_buckets(interests)) if interests else set()
    # if interest mentions web/backend/mobile/data… bias to relevant lang families (keep simple for now)
    candidates = []
    for l in langs:
        candidates.extend(_LANG_TO_FRAMEWORKS.get(l, []))
    # dedupe while preserving order
    seen, ordered = set(), []
    for fw in candidates:
        if fw not in seen:
            ordered.append(fw); seen.add(fw)
    return ordered[:topk]

class NewEntrantRequest(BaseModel):
    languages_known: List[str] = []
    interests: Optional[str | List[str]] = None
    topk: int = 5
    entrant_boost: bool = False

class ProfessionalRequest(BaseModel):
    all_skills: List[str] = []
    topk: int = 5
    prefer_low_barrier: bool = False

class FrameworksRequest(BaseModel):
    languages_known: List[str] = []
    interests: Optional[str | List[str]] = None
    topk: int = 12

# Routes
@app.get("/health")
def health():
    return {"status": "ok", "kw_rows": int(len(df_kw)), "meta_rows": int(len(META))}

@app.post("/recommend/new-entrant")
def api_recommend_new_entrant(req: NewEntrantRequest):
    try:
        df, frameworks = recommend_new_entrant_rule(
            languages_known=req.languages_known,
            areas_of_interest=req.interests,
            topk=req.topk,
            entrant_boost=req.entrant_boost,
        )
        return {
            "frameworks_by_language": frameworks,
            "results": df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/professional")
def api_recommend_professional(req: ProfessionalRequest):
    try:
        df, frameworks = recommend_professional_rule(
            all_skills=req.all_skills,
            topk=req.topk,
            prefer_low_barrier=req.prefer_low_barrier,
        )
        return {
            "frameworks_by_language": frameworks,
            "results": df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/frameworks")
def api_recommend_frameworks(req: FrameworksRequest):
    try:
        out = recommend_frameworks_by_language(
            languages_known=req.languages_known,
            interests=req.interests,
            topk=req.topk,
        )
        return {"frameworks": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meta")
def api_meta():
    small = META[["A_t_norm", "BarrierScore_norm", "StackScore_norm"]].reset_index().rename(columns={"index": "stack"})
    return {"meta": small.to_dict(orient="records")}
