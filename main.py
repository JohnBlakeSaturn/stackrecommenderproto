"""
FastAPI application that mirrors the recommendation logic from preprocessing_nb.ipynb
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Tech Stack Recommender API",
    description="API for recommending tech stacks based on user skills and interests",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded data
df_kw = None
stack_names = None
stack_docs = None
tf = None
stack_vecs = None
latest = None
META = None
_token_freq = None
_stack_allow = defaultdict(set)
PRIOR_SCORE = {}

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Friendly aliases for user inputs
_ALIAS = {
    'js':'javascript','ts':'typescript','py':'python','rb':'ruby',
    'postgres':'postgresql','mongo':'mongodb','tf':'tensorflow','sklearn':'scikit-learn',
    'k8s':'kubernetes','gcp':'google cloud','rn':'react native',
    'next.js':'nextjs','react.js':'react','node.js':'node'
}

# Interest to stack mapping
INTEREST_TO_STACK = {
    # Web
    'web':'Web','frontend':'Web','front-end':'Web','backend':'Web','back-end':'Web','fullstack':'Web','full-stack':'Web',
    'react':'Web','angular':'Web','vue':'Web','nextjs':'Web','node':'Web','express':'Web','django':'Web','flask':'Web','fastapi':'Web',
    # Mobile
    'mobile':'Mobile','android':'Mobile','ios':'Mobile','swift':'Mobile','kotlin':'Mobile','flutter':'Mobile','react native':'Mobile',
    # Cloud / DevOps
    'cloud':'Cloud','devops':'Cloud','sre':'Cloud','kubernetes':'Cloud','docker':'Cloud','terraform':'Cloud',
    'aws':'Cloud','azure':'Cloud','google cloud':'Cloud','gcp':'Cloud',
    # Data / AI
    'data':'AI/ML','data engineering':'AI/ML','data science':'AI/ML','machine learning':'AI/ML','ml':'AI/ML','deep learning':'AI/ML','nlp':'AI/ML',
    # Security
    'security':'Security','cybersecurity':'Security','pentest':'Security',
    # Game
    'gamedev':'GameDev','game':'GameDev','unreal':'GameDev','unity':'GameDev','godot':'GameDev',
    # Desktop / Systems
    'desktop':'Desktop','systems':'Systems','system':'Systems','embedded':'Systems','firmware':'Systems',
    'rust':'Systems','c':'Systems','c++':'Systems','go':'Systems',
    # Database / Viz / BI
    'database':'Database','sql':'Database','bi':'Visualization','visualization':'Visualization','analytics':'Visualization',
    # Blockchain
    'blockchain':'Blockchain','web3':'Blockchain','solidity':'Blockchain',
}

# Category stoplists
_CATEGORY_STOP = {
    'AI/ML': {'css','html','javascript','jquery','php','node','nodejs','react','vue','angular'},
    'Cloud': set(),
    'Systems': {'css','html','javascript','react','vue','angular','jquery'},
    'GameDev': {'css','html','jquery'},
    'Database': set(),
    'Visualization': set(),
    'Web': set(),
    'Mobile': {'jquery','php'},
    'Security': set(),
    'DevTools': set(),
}

# Core libraries per stack and language
_CORE_LIBS = {
    'AI/ML': {
        'python': [
            'pytorch', 'tensorflow', 'keras', 'scikit-learn',
            'xgboost', 'lightgbm', 'numpy', 'pandas'
        ],
        'r': ['ggplot2', 'caret', 'shiny', 'tidyverse'],
        'julia': ['flux']
    },
    'Web': {
        'python': ['django', 'flask', 'fastapi'],
        'javascript': ['react', 'next.js', 'vue.js', 'angular', 'express', 'node.js'],
        'typescript': ['next.js', 'nestjs', 'react', 'angular']
    }
}

# Language to framework mapping
_LANG_TO_FW = {
    'python': ['django', 'fastapi', 'flask', 'streamlit'],
    'javascript': ['react', 'nextjs', 'express', 'angular', 'vue', 'svelte'],
    'typescript': ['react', 'nextjs', 'nestjs', 'angular', 'vue'],
    'java': ['spring', 'spring boot', 'quarkus', 'micronaut'],
    'c#': ['asp.net', 'asp.net core', 'blazor'],
    'php': ['laravel', 'symfony', 'codeigniter'],
    'ruby': ['rails', 'sinatra'],
    'go': ['gin', 'fiber', 'echo'],
    'kotlin': ['spring', 'ktor'],
    'swift': ['vapor'],
    'dart': ['flutter'],
    'scala': ['play framework'],
    'elixir': ['phoenix'],
    'rust': ['axum', 'actix'],
}

# Bucket to framework mapping
_BUCKET_FW = {
    'web': ['react', 'nextjs', 'express', 'django', 'fastapi', 'spring', 'spring boot', 'laravel', 'rails', 'asp.net core', 'vue', 'angular'],
    'mobile': ['react native', 'flutter', 'swiftui', 'kotlin', 'xamarin'],
    'ai/ml': ['scikit-learn', 'pytorch', 'tensorflow', 'keras', 'fastai', 'xgboost'],
    'cloud': ['serverless', 'spring boot', 'asp.net core', 'nestjs'],
    'backend': ['spring', 'spring boot', 'express', 'fastapi', 'django', 'asp.net core', 'gin', 'fiber', 'rails'],
}

_LOW_LEVEL_TOKENS = {'c','c++','rust','embedded','firmware','assembly'}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_token(tok: str):
    """Normalize a token string"""
    if not isinstance(tok, str):
        return ""
    t = tok.strip()
    t = re.sub(r'\s+', ' ', t)
    t = t.replace("()", "").replace("™", "").replace("®", "")
    return t

def token_variants(tok: str):
    """Return a set of likely alias tokens"""
    t = tok.strip().lower()
    variants = {t}
    # remove punctuation
    n = re.sub(r'[^\w#\+]', '', t)
    variants.add(n)
    # handle .js
    if t.endswith('.js'):
        base = t[:-3]
        variants.add(base)
        variants.add(base+'js')
    if t == 'c#':
        variants.add('csharp')
    if t == 'f#':
        variants.add('fsharp')
    variants.add(t.replace('.', ''))
    variants.add(re.sub(r'\(.*?\)','',t).strip())
    return set([v for v in variants if v])

def _norm_list(xs):
    """Normalize a list of tokens"""
    out = []
    if isinstance(xs, (list, tuple, set)):
        it = xs
    elif xs is None:
        it = []
    else:
        it = [xs]
    for x in it:
        t = str(x).strip().lower()
        if not t:
            continue
        t = _ALIAS.get(t, t)
        out.append(t)
    return out

def _interest_buckets(interests):
    """Convert interest strings to stack buckets"""
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
            out.append(b)
            seen.add(b)
    return out

def _stack_keywords(stack):
    """Get keywords for a stack"""
    row = df_kw.loc[df_kw['stack']==stack, 'keywords']
    return set(str(row.squeeze()).split()) if not row.empty else set()

def _guess_category_for_stack(stk: str) -> str:
    """Guess category for a stack (stacks are already categories)"""
    return stk

def _canon(t: str) -> str:
    """Canonicalize a token"""
    return str(t or '').strip().lower()

def _norm_tokens(xs):
    """Normalize tokens for framework recommendations"""
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
        # light aliasing
        t = {'node': 'javascript', 'js':'javascript', 'ts':'typescript',
             'csharp':'c#', 'asp.net core':'asp.net core', 'rn':'react native'}.get(t, t)
        out.append(t)
    return out

def _infer_buckets_from_langs(langs_norm):
    """Infer interest buckets from language skills"""
    buckets = set()
    L = set(langs_norm)
    if {'html','css'} & L or 'javascript' in L or 'typescript' in L:
        buckets.add('web')
    if 'dart' in L or 'swift' in L or 'kotlin' in L or 'javascript' in L:
        buckets.add('mobile')
    if 'python' in L or 'r' in L or 'julia' in L:
        buckets.add('ai/ml')
    if {'java','c#','go','rust','python'} & L:
        buckets.add('backend')
    return list(buckets)

def suggest_techs_to_learn(stack, known_tokens=None, k=6, for_new_entrant: bool=False):
    """
    Category-aware suggestions with allowlist + frequency ranking.
    If for_new_entrant=True, apply language-aware core-library boost.
    """
    known = {_canon(t) for t in (known_tokens or []) if t}
    kws = str(df_kw.loc[df_kw['stack']==stack, 'keywords'].squeeze() or '').split()

    # keep only tokens that are actually observed for this stack
    allow = _stack_allow.get(stack, set())
    cand = [t for t in kws if t and _canon(t) in allow and _canon(t) not in known]

    # drop category-inappropriate basics
    cat = _guess_category_for_stack(stack)
    stop = _CATEGORY_STOP.get(cat, set())
    cand = [t for t in cand if _canon(t) not in stop]

    if not cand:
        return []

    # base score by frequency (+ optional entrant-only boost)
    scored = []
    core_for_stack = _CORE_LIBS.get(stack, {})
    for t in cand:
        tok = _canon(t)
        base = _token_freq.get(tok, 0)

        boost = 0
        if for_new_entrant:
            for lang in known:
                if lang in core_for_stack and tok in map(_canon, core_for_stack[lang]):
                    boost += 250_000

        scored.append((tok, base + boost))

    # sort & dedupe
    scored.sort(key=lambda x: (-x[1], x[0]))
    seen, out = set(), []
    for tok, _ in scored:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
        if len(out) >= k:
            break
    return out

def top_techs_for_stack(stack, topk=3):
    """Get top techs for a stack"""
    kws = df_kw.loc[df_kw['stack']==stack, 'keywords'].squeeze()
    if not isinstance(kws, str) or kws.strip()=="":
        return []
    toks = [t for t in kws.split() if t]
    return toks[:topk]

def combined_score(content_sim, stack):
    """Combine content similarity and meta scores"""
    alpha, beta, gamma = 0.6, 0.35, 0.35
    meta = latest.loc[stack] if stack in latest.index else None
    desir = float(meta.get('StackScore_norm', 0.0)) if meta is not None else 0.0
    barrier = float(meta.get('BarrierScore_norm', 0.5)) if meta is not None else 0.5
    pop = float(meta.get('A_t_norm', 0.0)) if meta is not None else 0.0
    score = alpha * float(np.clip(content_sim,0,1)) + beta * desir - gamma * barrier + 0.15 * pop
    return float(score)

# ============================================================================
# CORE RECOMMENDATION FUNCTIONS
# ============================================================================

def recommend_for_new_entrant(user_langs: list, interests: str="", topk=5):
    """TF-IDF based recommendation for new entrants"""
    user_langs = [str(x).strip().lower() for x in (user_langs or []) if str(x).strip()]
    qtext = " ".join(user_langs) + " " + (interests or "")
    qv = tf.transform([qtext])
    sims = cosine_similarity(qv, stack_vecs).flatten()
    out = []
    for i, stack in enumerate(stack_names):
        out.append({
            'stack': stack,
            'score': combined_score(sims[i], stack),
            'content_sim': float(sims[i]),
            'top_techs': top_techs_for_stack(stack, topk=3)
        })
    df = pd.DataFrame(out).sort_values('score', ascending=False).reset_index(drop=True)
    return df.head(topk)

def recommend_for_professional(user_profile: dict, topk=5):
    """TF-IDF based recommendation for professionals"""
    years = float(user_profile.get('years_exp', 0.0) or 0.0)
    known = {str(x).strip().lower() for x in (user_profile.get('known_languages',[]) or []) if x}
    platforms = {str(x).strip().lower() for x in (user_profile.get('known_platforms',[]) or []) if x}
    webfws = {str(x).strip().lower() for x in (user_profile.get('known_webfw',[]) or []) if x}
    desired = user_profile.get('desired_fields','') or ""
    qv = tf.transform([desired])
    out_fit, out_growth = [], []
    for i, stack in enumerate(stack_names):
        kwset = set([t for t in df_kw.loc[df_kw['stack']==stack, 'keywords'].squeeze().split() if t])
        overlap = len(kwset & known) + len(kwset & platforms) + len(kwset & webfws)
        overlap_norm = overlap / max(1, len(kwset))
        content_sim = float(cosine_similarity(qv, stack_vecs[i]).flatten()[0])
        meta = latest.loc[stack] if stack in latest.index else None
        desir = float(meta.get('StackScore_norm',0.0)) if meta is not None else 0.0
        barrier = float(meta.get('BarrierScore_norm',0.5)) if meta is not None else 0.5
        pop = float(meta.get('A_t_norm',0.0)) if meta is not None else 0.0
        fit = 0.5*overlap_norm + 0.25*desir + 0.15*content_sim + 0.1*(min(years,10)/10)
        growth = 0.5*desir + 0.25*(1-barrier) + 0.15*content_sim - 0.05*overlap_norm + 0.05*pop
        out_fit.append({'stack':stack,'fit':fit,'overlap':overlap,'top_techs':top_techs_for_stack(stack,3)})
        out_growth.append({'stack':stack,'growth':growth,'overlap':overlap,'top_techs':top_techs_for_stack(stack,3)})
    df_fit = pd.DataFrame(out_fit).sort_values(['fit','overlap'], ascending=[False,False]).head(topk).reset_index(drop=True)
    df_growth = pd.DataFrame(out_growth).sort_values(['growth','overlap'], ascending=[False,True])
    df_growth = df_growth[df_growth['overlap'] <= 2].sort_values(['growth','overlap'], ascending=[False, True]).head(topk)
    return df_fit, df_growth

def recommend_new_entrant_rule(user_langs, interests, topk=6):
    """Rule-based recommendation for new entrants"""
    user_langs = _norm_list(user_langs)
    buckets = set(_interest_buckets(interests))
    kw_by_stack = {r.stack: set(str(r.keywords).split()) for _, r in df_kw.iterrows()}

    rows = []
    for stack in df_kw['stack']:
        score = 0.0
        reason_bits = []
        overl = len(kw_by_stack.get(stack, set()) & set(user_langs))

        # (a) interest bucket
        if stack in buckets:
            score += 2.0
            reason_bits.append(f"matches interest: {stack}")

        # (b) overlap
        gain = min(overl, 3) * 1.0
        score += gain
        if overl:
            reason_bits.append(f"{overl} skill overlap")

        # (c) prior
        prior = float(PRIOR_SCORE.get(stack, 0.0))
        if interests:
            score += 0.5 * prior
            if prior > 0:
                reason_bits.append("popular")
        else:
            # soften prior for niche profiles
            if (_LOW_LEVEL_TOKENS & set(user_langs)) and stack != 'Systems':
                score += 0.2 * prior
            else:
                score += 0.4 * prior
            if prior > 0:
                reason_bits.append("popular")

        # (d) low-level bias
        if not interests and (_LOW_LEVEL_TOKENS & set(user_langs)) and stack == 'Systems':
            score += 0.7
            reason_bits.append("low-level bias")

        # (e) barrier penalty
        if META is not None and 'BarrierScore_norm' in META.columns and stack in META.index:
            bpen = 0.5 * float(META.loc[stack, 'BarrierScore_norm'])
            if bpen > 0:
                score -= bpen
                reason_bits.append("lower barrier preferred")

        learn_next = suggest_techs_to_learn(stack, known_tokens=user_langs, k=6, for_new_entrant=True)

        rows.append({
            'stack': stack,
            'rule_score': float(score),
            'overlap': int(overl),
            'prior': float(prior),
            'reason': "; ".join(reason_bits) if reason_bits else "general fit",
            'learn_next': learn_next,
            'top_techs': learn_next[:3]
        })

    rb = pd.DataFrame(rows).sort_values(['rule_score','overlap','prior'], ascending=[False, False, False]).reset_index(drop=True)
    return rb.head(topk)

def recommend_professional_rule(all_skills, topk=5, prefer_low_barrier=False):
    """Rule-based recommendation for professionals"""
    known = set(_norm_list(all_skills))
    rows = []

    for stack in df_kw['stack'].unique():
        kws = set(str(df_kw.loc[df_kw['stack']==stack, 'keywords'].squeeze()).split())
        overlap = len(kws & known)

        A = float(META.loc[stack, 'A_t_norm']) if META is not None and stack in META.index else 0.0
        S = float(META.loc[stack, 'StackScore_norm']) if META is not None and stack in META.index else 0.0
        B = float(META.loc[stack, 'BarrierScore_norm']) if META is not None and stack in META.index else 0.0

        score = 1.1*min(overlap, 5) + 0.6*A + 0.4*S - (0.4*B if prefer_low_barrier else 0.0)

        learn_next = suggest_techs_to_learn(stack, known_tokens=known, k=6)

        rows.append({
            'stack': stack,
            'score': float(score),
            'overlap': int(overlap),
            'A_t_norm': round(A, 3),
            'StackScore_norm': round(S, 3),
            'BarrierScore_norm': round(B, 3),
            'learn_next': learn_next,
            'top_techs': learn_next[:3],
        })

    return (pd.DataFrame(rows)
              .sort_values(['score','overlap','A_t_norm','StackScore_norm'], ascending=[False, False, False, False])
              .head(topk)
              .reset_index(drop=True))

def recommend_frameworks_by_language(languages_known, interests=None, topk=8):
    """Recommend frameworks based on languages and interests"""
    langs = _norm_tokens(languages_known)
    # candidate frameworks from languages
    cands = []
    for lang in langs:
        if lang in _LANG_TO_FW:
            cands.extend(_LANG_TO_FW[lang])

    # buckets from interests OR inferred from langs
    if interests:
        buckets = _interest_buckets(interests)
    else:
        buckets = _infer_buckets_from_langs(langs)

    # add bucket frameworks
    for b in buckets:
        cands.extend(_BUCKET_FW.get(b.lower(), []))

    # fallback to popular web frameworks
    if not cands:
        cands = _BUCKET_FW['web'][:]

    # build canonical token set
    canon = set()
    for kws in df_kw['keywords']:
        canon.update(str(kws).split())
    canon = set(map(str.lower, canon))

    # rank by frequency + meta nudge
    scored = []
    seen = set()
    for fw in cands:
        key = fw.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        score = 1.0
        if key in canon:
            score += 0.5
        # small preference if it matches a known language family
        for lang, fwlist in _LANG_TO_FW.items():
            if key in [f.lower() for f in fwlist]:
                score += 0.2
        scored.append((key, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    out = [name for name, _ in scored][:topk]
    return out

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(keywords_path: str = "stack_tech_keywords_from_map.csv",
              meta_path: str = "stack_scores_per_stack_year.csv"):
    """Load all necessary data files and build models"""
    global df_kw, stack_names, stack_docs, tf, stack_vecs, latest, META, _token_freq, _stack_allow, PRIOR_SCORE
    
    # Load keywords
    kw_path = Path(keywords_path)
    if not kw_path.exists():
        raise FileNotFoundError(f"Keywords file not found: {keywords_path}")
    
    df_kw = pd.read_csv(kw_path)
    df_kw['keywords'] = df_kw['keywords'].fillna('').astype(str)
    stack_names = df_kw['stack'].tolist()
    stack_docs = df_kw['keywords'].tolist()
    
    # Build TF-IDF vectorizer
    tf = TfidfVectorizer(ngram_range=(1,2), max_features=4000)
    stack_vecs = tf.fit_transform(stack_docs)
    
    # Load meta scores if exists
    meta_path = Path(meta_path)
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        latest = meta.loc[meta.groupby('stack')['year'].idxmax()].set_index('stack')
    else:
        latest = pd.DataFrame(index=stack_names)
    
    # Ensure meta columns exist and normalize
    for col in ['StackScore','BarrierScore','A_t','D_t','Delta','SatisfactionProxy']:
        if col not in latest.columns:
            latest[col] = 0.0
        vals = latest[col].astype(float)
        latest[col + "_norm"] = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() > vals.min() else 0.0
    
    # Build META dataframe
    if meta_path.exists():
        m = pd.read_csv(meta_path)
        META = m.loc[m.groupby('stack')['year'].idxmax()].copy()
        for col in ['A_t','BarrierScore','StackScore']:
            if col not in META.columns:
                META[col] = 0.0
        def _norm(s):
            s = s.astype(float)
            return (s - s.min())/(s.max() - s.min()) if s.max() > s.min() else 0.0*s
        META['A_t_norm'] = _norm(META['A_t'])
        META['BarrierScore_norm'] = _norm(META['BarrierScore'])
        META['StackScore_norm'] = _norm(META['StackScore'])
        META = META.set_index('stack')
    
    # Build global token frequencies
    _all_tokens = [t for kws in df_kw['keywords'] for t in str(kws).split() if t]
    _token_freq = Counter(_all_tokens)
    
    # Build stack allowlists (use keywords as fallback)
    _stack_allow = defaultdict(set)
    for _, r in df_kw.iterrows():
        stk = r['stack']
        for t in str(r['keywords']).split():
            _stack_allow[stk].add(str(t).lower())
    
    # Build PRIOR_SCORE from A_t_norm
    PRIOR_SCORE = {}
    if META is not None and 'A_t_norm' in META.columns:
        for stack in META.index:
            PRIOR_SCORE[stack] = float(META.loc[stack, 'A_t_norm'])

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class NewEntrantRequest(BaseModel):
    languages_known: List[str] = Field(default=[], description="List of programming languages the user knows")
    areas_of_interest: Optional[str] = Field(default="", description="Areas of interest (e.g., 'web frontend', 'machine learning')")
    topk: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")
    use_tfidf: bool = Field(default=True, description="Use TF-IDF based recommendation instead of rule-based")

class ProfessionalRequest(BaseModel):
    # Simple mode (use all_skills)
    all_skills: Optional[List[str]] = Field(default=None, description="List of all known skills (languages, frameworks, platforms) - simple mode")
    
    # Detailed mode (more specific categorization)
    years_exp: Optional[float] = Field(default=None, description="Years of professional experience")
    known_languages: Optional[List[str]] = Field(default=None, description="Programming languages")
    known_platforms: Optional[List[str]] = Field(default=None, description="Platforms like AWS, Docker, Kubernetes")
    known_webfw: Optional[List[str]] = Field(default=None, description="Web frameworks")
    desired_fields: Optional[str] = Field(default=None, description="Career interests or desired fields")
    
    topk: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")
    prefer_low_barrier: bool = Field(default=False, description="Prefer stacks with lower learning barriers")

class FrameworkRequest(BaseModel):
    languages_known: List[str] = Field(description="List of programming languages the user knows")
    interests: Optional[str] = Field(default=None, description="Areas of interest")
    topk: int = Field(default=8, ge=1, le=20, description="Number of framework recommendations to return")

class HealthResponse(BaseModel):
    status: str
    message: str

class StackRecommendation(BaseModel):
    stack: str
    score: float
    overlap: Optional[int] = None
    reason: Optional[str] = None
    learn_next: List[str]
    top_techs: List[str]
    prior: Optional[float] = None
    content_sim: Optional[float] = None
    fit: Optional[float] = None
    growth: Optional[float] = None
    A_t_norm: Optional[float] = None
    StackScore_norm: Optional[float] = None
    BarrierScore_norm: Optional[float] = None

class RecommendationResponse(BaseModel):
    frameworks_by_language: List[str]
    results: List[Dict[str, Any]]

class ProfessionalResponse(BaseModel):
    frameworks_by_language: List[str]
    results: List[Dict[str, Any]]
    fit_recommendations: List[Dict[str, Any]]
    growth_recommendations: List[Dict[str, Any]]

class FrameworkResponse(BaseModel):
    frameworks: List[str]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    try:
        # Try to load from default paths
        load_data()
    except FileNotFoundError as e:
        print(f"Warning: Could not load data files on startup: {e}")
        print("Please call POST /load-data with correct file paths")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "message": "Tech Stack Recommender API is running"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if df_kw is None:
        return {
            "status": "unhealthy",
            "message": "Data not loaded. Please call POST /load-data"
        }
    return {
        "status": "healthy",
        "message": "All systems operational"
    }

@app.post("/load-data")
async def load_data_endpoint(keywords_path: str = "stack_tech_keywords_from_map.csv",
                             meta_path: str = "stack_scores_per_stack_year.csv"):
    """Load or reload data files"""
    try:
        load_data(keywords_path, meta_path)
        return {
            "status": "success",
            "message": f"Data loaded successfully. {len(stack_names)} stacks available."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

@app.post("/recommend/new-entrant", response_model=RecommendationResponse)
async def recommend_new_entrant(request: NewEntrantRequest):
    """
    Get tech stack recommendations for new entrants.
    
    By default uses TF-IDF based recommendations. Set use_tfidf=False for rule-based recommendations.
    """
    if df_kw is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Please call POST /load-data first.")
    
    try:
        # Get framework recommendations
        frameworks = recommend_frameworks_by_language(
            request.languages_known,
            request.areas_of_interest,
            topk=8
        )
        
        if request.use_tfidf:
            # TF-IDF based recommendation
            df = recommend_for_new_entrant(
                request.languages_known,
                request.areas_of_interest,
                request.topk
            )
        else:
            # Rule-based recommendation
            df = recommend_new_entrant_rule(
                request.languages_known,
                request.areas_of_interest,
                request.topk
            )
        
        # Convert to dict and add missing fields
        results = df.to_dict(orient='records')
        
        # Ensure all results have the required fields
        for result in results:
            # Add A_t_norm, BarrierScore_norm if not present
            stack = result.get('stack')
            if 'A_t_norm' not in result and latest is not None and stack in latest.index:
                result['A_t_norm'] = float(latest.loc[stack, 'A_t_norm'])
            if 'BarrierScore_norm' not in result and latest is not None and stack in latest.index:
                result['BarrierScore_norm'] = float(latest.loc[stack, 'BarrierScore_norm'])
            
            # Rename score field if needed
            if 'rule_score' in result:
                result['score'] = result.pop('rule_score')
            
            # Ensure reason field exists
            if 'reason' not in result:
                reasons = []
                if result.get('overlap', 0) > 0:
                    reasons.append(f"{result['overlap']} overlap")
                if result.get('content_sim', 0) > 0.3:
                    reasons.append("interest match")
                if result.get('A_t_norm', 0) > 0.3:
                    reasons.append("popular")
                if result.get('BarrierScore_norm', 0) < 0.5:
                    reasons.append("lower barrier")
                result['reason'] = "; ".join(reasons) if reasons else "general fit"
        
        return {
            "frameworks_by_language": frameworks,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/recommend/professional", response_model=ProfessionalResponse)
async def recommend_professional(request: ProfessionalRequest):
    """
    Get tech stack recommendations for professionals using TF-IDF.
    
    Supports two modes:
    1. Simple mode: Provide 'all_skills' list
    2. Detailed mode: Provide 'years_exp', 'known_languages', 'known_platforms', 'known_webfw', 'desired_fields'
    
    Returns frameworks_by_language, main results, fit recommendations, and growth recommendations.
    """
    if df_kw is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Please call POST /load-data first.")
    
    try:
        # Determine which mode to use
        if request.all_skills is not None:
            # Simple mode - use all_skills
            all_skills = request.all_skills
            user_profile = {
                'years_exp': 0.0,
                'known_languages': all_skills,
                'known_platforms': [],
                'known_webfw': [],
                'desired_fields': ''
            }
        else:
            # Detailed mode - use specific fields
            all_skills = []
            if request.known_languages:
                all_skills.extend(request.known_languages)
            if request.known_platforms:
                all_skills.extend(request.known_platforms)
            if request.known_webfw:
                all_skills.extend(request.known_webfw)
            
            user_profile = {
                'years_exp': request.years_exp or 0.0,
                'known_languages': request.known_languages or [],
                'known_platforms': request.known_platforms or [],
                'known_webfw': request.known_webfw or [],
                'desired_fields': request.desired_fields or ''
            }
        
        # Get framework recommendations
        frameworks = recommend_frameworks_by_language(
            all_skills,
            request.desired_fields if request.desired_fields else None,
            topk=8
        )
        
        # Get TF-IDF recommendations (fit and growth)
        df_fit, df_growth = recommend_for_professional(user_profile, request.topk)
        
        # Process fit recommendations for main results
        results = df_fit.to_dict(orient='records')
        
        for result in results:
            stack = result.get('stack')
            
            # Add A_t_norm, BarrierScore_norm if not present
            if 'A_t_norm' not in result and latest is not None and stack in latest.index:
                result['A_t_norm'] = float(latest.loc[stack, 'A_t_norm'])
            if 'BarrierScore_norm' not in result and latest is not None and stack in latest.index:
                result['BarrierScore_norm'] = float(latest.loc[stack, 'BarrierScore_norm'])
            
            # Rename fit to score for main results
            if 'fit' in result:
                result['score'] = result.pop('fit')
            
            # Build reason field
            reasons = []
            if result.get('overlap', 0) > 0:
                reasons.append(f"{result['overlap']} overlap")
            if result.get('score', 0) > 0.5:
                reasons.append("good fit")
            if result.get('A_t_norm', 0) > 0.3:
                reasons.append("popular")
            if result.get('BarrierScore_norm', 0) < 0.5 or request.prefer_low_barrier:
                reasons.append("lower barrier")
            result['reason'] = "; ".join(reasons) if reasons else "general fit"
            
            # Ensure learn_next exists (use top_techs if not present)
            if 'learn_next' not in result:
                result['learn_next'] = result.get('top_techs', [])
        
        # Process fit recommendations
        fit_recommendations = df_fit.to_dict(orient='records')
        for rec in fit_recommendations:
            stack = rec.get('stack')
            if 'A_t_norm' not in rec and latest is not None and stack in latest.index:
                rec['A_t_norm'] = float(latest.loc[stack, 'A_t_norm'])
            if 'BarrierScore_norm' not in rec and latest is not None and stack in latest.index:
                rec['BarrierScore_norm'] = float(latest.loc[stack, 'BarrierScore_norm'])
            if 'StackScore_norm' not in rec and latest is not None and stack in latest.index:
                rec['StackScore_norm'] = float(latest.loc[stack, 'StackScore_norm'])
            if 'learn_next' not in rec:
                rec['learn_next'] = rec.get('top_techs', [])
        
        # Process growth recommendations
        growth_recommendations = df_growth.to_dict(orient='records')
        for rec in growth_recommendations:
            stack = rec.get('stack')
            if 'A_t_norm' not in rec and latest is not None and stack in latest.index:
                rec['A_t_norm'] = float(latest.loc[stack, 'A_t_norm'])
            if 'BarrierScore_norm' not in rec and latest is not None and stack in latest.index:
                rec['BarrierScore_norm'] = float(latest.loc[stack, 'BarrierScore_norm'])
            if 'StackScore_norm' not in rec and latest is not None and stack in latest.index:
                rec['StackScore_norm'] = float(latest.loc[stack, 'StackScore_norm'])
            if 'learn_next' not in rec:
                rec['learn_next'] = rec.get('top_techs', [])
        
        return {
            "frameworks_by_language": frameworks,
            "results": results,
            "fit_recommendations": fit_recommendations,
            "growth_recommendations": growth_recommendations
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/recommend/frameworks", response_model=FrameworkResponse)
async def recommend_frameworks(request: FrameworkRequest):
    """
    Get framework recommendations based on languages and interests.
    """
    if df_kw is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Please call POST /load-data first.")
    
    try:
        frameworks = recommend_frameworks_by_language(
            request.languages_known,
            request.interests,
            request.topk
        )
        return {"frameworks": frameworks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating framework recommendations: {str(e)}")

@app.get("/stacks")
async def get_stacks():
    """Get list of all available stacks"""
    if df_kw is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Please call POST /load-data first.")
    
    return {"stacks": stack_names}

@app.get("/stack/{stack_name}")
async def get_stack_info(stack_name: str):
    """Get detailed information about a specific stack"""
    if df_kw is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Please call POST /load-data first.")
    
    if stack_name not in stack_names:
        raise HTTPException(status_code=404, detail=f"Stack '{stack_name}' not found")
    
    keywords = df_kw.loc[df_kw['stack']==stack_name, 'keywords'].squeeze()
    
    result = {
        "stack": stack_name,
        "keywords": str(keywords),
        "top_techs": top_techs_for_stack(stack_name, topk=10)
    }
    
    # Add meta information if available
    if latest is not None and stack_name in latest.index:
        meta_info = latest.loc[stack_name]
        result.update({
            "A_t_norm": float(meta_info.get('A_t_norm', 0.0)),
            "BarrierScore_norm": float(meta_info.get('BarrierScore_norm', 0.0)),
            "StackScore_norm": float(meta_info.get('StackScore_norm', 0.0))
        })
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)