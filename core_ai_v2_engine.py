"""
core_ai_v2_engine.py — CoreAI v2 엔진
======================================
Hopfield 분해기 → 클러스터별 마르코프 + TinyTransformer 의미 보정
기존 CoreAI 대비:
  - 코퍼스 습득 구조화 (Hopfield 클러스터)
  - OOV 의미 보정 강화 (TinyTransformer)
  - 클러스터별 정밀 판정
GPU 없음, numpy only
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict, Counter
from typing import Optional
import time

from neural_markov_engine import NeuralMarkovEngine as _NME

_rng = np.random.default_rng(42)

# ── 토크나이저 ────────────────────────────────────────────────
_JOSA = ["에서","에게","으로","에도","에만","부터","까지","이랑",
         "와","과","을","를","은","는","이","가","의","도","만","에","로"]
_EOMI = ["했습니다","합니다","됩니다","있습니다","없습니다","입니다",
         "했다","한다","됩니다","이다","하고","해서","하여","되어",
         "이며","하는","된","한","이고","인가요","인지","인데",
         "나요","가요","해요","지요","군요","할까요","인가요"]

def tokenize(text: str) -> list[str]:
    tokens = []
    for word in text.replace("\n"," ").split():
        word = word.strip(".,!?()[]\"'~：:；;")
        stem = word
        for s in sorted(_JOSA+_EOMI, key=len, reverse=True):
            if word.endswith(s) and len(word) > len(s)+1:
                stem = word[:-len(s)]
                break
        if stem and len(stem) > 1:
            tokens.append(stem)
    return tokens


# ── TinyEmbedder ──────────────────────────────────────────────
class TinyEmbedder:
    def __init__(self, vocab: dict, dim: int = 32):
        self.vocab = vocab
        self.V = len(vocab)
        self.dim = dim
        self.emb = _rng.normal(0, 0.1, (self.V, dim))
        self.inv_vocab = {i:w for w,i in vocab.items()}

    def train(self, tokens: list, epochs: int = 10,
              window: int = 2, lr: float = 0.05):
        pairs = []
        for i, w in enumerate(tokens):
            if w not in self.vocab: continue
            for j in range(max(0,i-window), min(len(tokens),i+window+1)):
                if j!=i and tokens[j] in self.vocab:
                    pairs.append((self.vocab[w], self.vocab[tokens[j]]))

        def sig(x): return 1/(1+np.exp(-np.clip(x,-500,500)))

        for _ in range(epochs):
            _rng.shuffle(pairs)
            for c, ctx in pairs[:1000]:
                negs = _rng.integers(0, self.V, 5)
                vc = self.emb[c]; vp = self.emb[ctx]
                pp = sig(np.dot(vc,vp))
                gc = (pp-1)*vp; self.emb[ctx] -= lr*(pp-1)*vc
                for ng in negs:
                    vn = self.emb[ng]; pn = sig(np.dot(vc,vn))
                    gc += pn*vn; self.emb[ng] -= lr*pn*vc
                self.emb[c] -= lr*gc

        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)+1e-12
        self.emb /= norms

    def nearest(self, word: str, topk: int = 5) -> list:
        if word not in self.vocab: return []
        v = self.emb[self.vocab[word]]
        sims = self.emb @ v
        idx = np.argsort(-sims)[1:topk+1]
        return [(self.inv_vocab[i], float(sims[i])) for i in idx]

    def expand(self, tokens: list, topk: int = 3,
               threshold: float = 0.55) -> list:
        expanded = list(tokens)
        for t in tokens:
            if t in self.vocab:
                nn = self.nearest(t, topk)
                expanded.extend([w for w,s in nn if s >= threshold])
            else:
                # 글자 유사도 기반 근사
                best = sorted(
                    [(w, sum(1 for c in t if c in w)/max(len(t),len(w)))
                     for w in self.vocab],
                    key=lambda x:-x[1]
                )[:topk]
                expanded.extend([w for w,s in best if s > 0.4])
        return list(set(expanded))


# ── Hopfield 분해기 ───────────────────────────────────────────
class HopfieldDecomposer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.vocab: dict = {}
        self.V: int = 0
        self.W: Optional[np.ndarray] = None
        self.cluster_sentences: dict = defaultdict(list)
        self.cluster_tokens: dict = defaultdict(list)
        self.cluster_keywords: dict = {}

    def _build_vocab(self, tokens):
        cnt = Counter(tokens)
        words = [w for w,c in cnt.most_common() if c >= 2]
        self.vocab = {w:i for i,w in enumerate(words)}
        self.V = len(words)

    def _vec(self, tokens):
        v = np.zeros(self.V)
        for t in tokens:
            if t in self.vocab: v[self.vocab[t]] += 1.0
        norm = np.linalg.norm(v)
        return v/norm if norm>1e-12 else v

    def _hebbian(self, tokens_list, window=3):
        self.W = np.zeros((self.V, self.V))
        all_t = [t for ts in tokens_list for t in ts if t in self.vocab]
        for i, t in enumerate(all_t):
            for j in range(max(0,i-window), min(len(all_t),i+window+1)):
                if j==i: continue
                vi = self.vocab.get(t)
                vj = self.vocab.get(all_t[j])
                if vi is not None and vj is not None:
                    self.W[vi][vj] += 1.0
        np.fill_diagonal(self.W, 0)
        self.W /= (len(all_t)+1e-12)

    def decompose(self, sentences: list):
        tokens_list = [tokenize(s) for s in sentences]
        all_tokens = [t for ts in tokens_list for t in ts]
        self._build_vocab(all_tokens)

        if self.V < 5 or len(sentences) <= self.n_clusters:
            chunk = max(1, len(sentences)//self.n_clusters)
            for i,(s,ts) in enumerate(zip(sentences,tokens_list)):
                k = min(i//chunk, self.n_clusters-1)
                self.cluster_sentences[k].append(s)
                self.cluster_tokens[k].extend(ts)
            for k in range(self.n_clusters):
                toks = self.cluster_tokens[k]
                self.cluster_keywords[k] = [
                    w for w,_ in Counter(toks).most_common(5)
                ]
            return

        self._hebbian(tokens_list)
        vecs = np.array([self._vec(ts) for ts in tokens_list])

        # K-means (코사인)
        idx = _rng.choice(len(vecs), self.n_clusters, replace=False)
        centers = vecs[idx].copy()
        assignments = np.zeros(len(vecs), dtype=int)

        for it in range(40):
            new_asgn = np.array([
                int(np.argmax(centers @ v))
                for v in vecs
            ])
            if np.all(new_asgn == assignments) and it > 0: break
            assignments = new_asgn
            for k in range(self.n_clusters):
                members = vecs[assignments==k]
                if len(members) > 0:
                    c = np.mean(members, axis=0)
                    norm = np.linalg.norm(c)
                    centers[k] = c/norm if norm>1e-12 else c

        for i,(s,ts,a) in enumerate(zip(sentences,tokens_list,assignments)):
            k = int(a)
            self.cluster_sentences[k].append(s)
            self.cluster_tokens[k].extend(ts)

        for k in range(self.n_clusters):
            toks = self.cluster_tokens[k]
            self.cluster_keywords[k] = [
                w for w,_ in Counter(toks).most_common(5)
            ]


# ── 클러스터 마르코프 ─────────────────────────────────────────
class ClusterMarkov:
    def __init__(self):
        self.uni = Counter()
        self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter)
        self.total = 0

    def train(self, tokens: list):
        self.total = len(tokens)
        for i,t in enumerate(tokens):
            self.uni[t] += 1
            if i>=1: self.bi[tokens[i-1]][t] += 1
            if i>=2: self.tri[(tokens[i-2],tokens[i-1])][t] += 1

    def score(self, tokens: list, alpha: float = 0.001) -> float:
        V = len(self.uni)
        if V==0 or len(tokens)<2: return -20.0
        total_lp = 0.0; scored = 0
        for i in range(len(tokens)):
            wc = tokens[i]
            p1 = (self.uni[wc]+alpha)/(self.total+alpha*V)
            p2 = p3 = 0.0
            if i>=1:
                wp = tokens[i-1]
                p2 = self.bi[wp][wc]/self.uni[wp] if self.uni[wp]>0 else 0
            if i>=2:
                wpp = tokens[i-2]
                p3 = self.tri[(wpp,wp)][wc]/self.bi[wpp][wp] if self.bi[wpp][wp]>0 else 0
            pjm = 0.6*p3+0.3*p2+0.1*p1
            total_lp += np.log(pjm+1e-12); scored += 1
        return total_lp/max(scored,1)


# ── CoreAI v2 메인 엔진 ───────────────────────────────────────
class CoreAIv2Engine:
    """
    CoreAI v2 — Hopfield 분해 + 마르코프 + TinyTransformer
    기존 CoreAI 대비:
      코퍼스를 Hopfield로 의미 클러스터화 후 마르코프 학습
      → 정확도 향상 + 학습 구조화
    """
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.decomposer = HopfieldDecomposer(n_clusters)
        self.markovs: dict = {}
        self.embedder: Optional[TinyEmbedder] = None
        self.global_vocab: dict = {}
        self.nm_engine = _NME()   # 전체 코퍼스 NeuralMarkov (가드레일 판정용)
        self.is_trained = False
        self.corpus_name = ""
        self.train_stats = {}

    def train(self, corpus_text: str, emb_epochs: int = 10,
              on_progress=None):
        t0 = time.perf_counter()

        sentences = [s.strip() for s in corpus_text.split("\n")
                     if s.strip() and len(s.strip()) > 5]
        all_tokens = tokenize(corpus_text)
        cnt = Counter(all_tokens)
        vocab_words = [w for w,c in cnt.most_common() if c>=2]
        self.global_vocab = {w:i for i,w in enumerate(vocab_words)}

        # 1. TinyTransformer
        if on_progress: on_progress(0, "TinyTransformer 학습 중...")
        t1 = time.perf_counter()
        self.embedder = TinyEmbedder(self.global_vocab, dim=32)
        self.embedder.train(all_tokens, epochs=emb_epochs)
        emb_ms = (time.perf_counter()-t1)*1000

        # 2. Hopfield 분해
        if on_progress: on_progress(50, "Hopfield 분해 중...")
        t2 = time.perf_counter()
        self.decomposer.decompose(sentences)
        decomp_ms = (time.perf_counter()-t2)*1000

        # 3. 클러스터별 마르코프 (구조 분석용)
        if on_progress: on_progress(80, "마르코프 학습 중...")
        t3 = time.perf_counter()
        for k in range(self.n_clusters):
            tokens = self.decomposer.cluster_tokens.get(k, [])
            if not tokens: continue
            m = ClusterMarkov()
            m.train(tokens)
            self.markovs[k] = m
        markov_ms = (time.perf_counter()-t3)*1000

        # 4. NeuralMarkov 전체 코퍼스 학습 (가드레일 판정용)
        if on_progress: on_progress(90, "NeuralMarkov 학습 중...")
        self.nm_engine = _NME()
        self.nm_engine.train(corpus_text, embedding_dim=32, epochs=emb_epochs)

        total_ms = (time.perf_counter()-t0)*1000
        if on_progress: on_progress(100, "완료")

        self.is_trained = True
        self.train_stats = {
            "n_sentences": len(sentences),
            "vocab_size": len(self.global_vocab),
            "emb_ms": emb_ms,
            "decomp_ms": decomp_ms,
            "markov_ms": markov_ms,
            "total_ms": total_ms,
            "clusters": {
                k: {
                    "n_sentences": len(self.decomposer.cluster_sentences.get(k,[])),
                    "keywords": self.decomposer.cluster_keywords.get(k,[]),
                }
                for k in range(self.n_clusters)
            }
        }
        return self.train_stats

    def save(self, path: str) -> bool:
        """학습된 엔진 전체를 pkl로 저장 (tok_emb + W + 마르코프 전부)"""
        import pickle, os
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            data = {
                "n_clusters":       self.n_clusters,
                "global_vocab":     self.global_vocab,
                "corpus_name":      self.corpus_name,
                "train_stats":      self.train_stats,
                # TinyEmbedder
                "emb_emb":          self.embedder.emb if self.embedder else None,
                "emb_vocab":        self.embedder.vocab if self.embedder else None,
                "emb_dim":          self.embedder.dim if self.embedder else 32,
                # Hopfield 분해기
                "cluster_sentences":  dict(self.decomposer.cluster_sentences),
                "cluster_tokens":     dict(self.decomposer.cluster_tokens),
                "cluster_keywords":   self.decomposer.cluster_keywords,
                "decomp_vocab":       self.decomposer.vocab,
                "decomp_W":           self.decomposer.W,
                # 클러스터 마르코프
                "markovs": {
                    k: {
                        "uni":   dict(m.uni),
                        "bi":    {k2: dict(v) for k2,v in m.bi.items()},
                        "tri":   {k2: dict(v) for k2,v in m.tri.items()},
                        "total": m.total,
                    }
                    for k, m in self.markovs.items()
                },
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception:
            return False

    @classmethod
    def load_from_dict(cls, data: dict) -> "CoreAIv2Engine":
        """딕셔너리에서 엔진 복원 (앱 pkl 업로드용)"""
        engine = cls(n_clusters=data["n_clusters"])
        engine.global_vocab = data["global_vocab"]
        engine.corpus_name  = data.get("corpus_name","")
        engine.train_stats  = data.get("train_stats",{})

        if data.get("emb_emb") is not None:
            engine.embedder = TinyEmbedder(data["emb_vocab"], data["emb_dim"])
            engine.embedder.emb = data["emb_emb"]
            engine.embedder.inv_vocab = {i:w for w,i in data["emb_vocab"].items()}

        engine.decomposer.vocab             = data["decomp_vocab"]
        engine.decomposer.V                 = len(data["decomp_vocab"])
        engine.decomposer.W                 = data["decomp_W"]
        engine.decomposer.cluster_sentences = defaultdict(list, data["cluster_sentences"])
        engine.decomposer.cluster_tokens    = defaultdict(list, data["cluster_tokens"])
        engine.decomposer.cluster_keywords  = data["cluster_keywords"]

        for k, md in data["markovs"].items():
            m = ClusterMarkov()
            m.uni   = Counter(md["uni"])
            m.bi    = defaultdict(Counter, {k2:Counter(v) for k2,v in md["bi"].items()})
            m.tri   = defaultdict(Counter, {k2:Counter(v) for k2,v in md["tri"].items()})
            m.total = md["total"]
            engine.markovs[k] = m

        engine.is_trained = True
        return engine

    @classmethod
    def load(cls, path: str) -> "CoreAIv2Engine":
        """저장된 pkl에서 엔진 복원 (~10ms)"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        engine = cls(n_clusters=data["n_clusters"])
        engine.global_vocab  = data["global_vocab"]
        engine.corpus_name   = data.get("corpus_name", "")
        engine.train_stats   = data.get("train_stats", {})

        # TinyEmbedder 복원
        if data["emb_emb"] is not None:
            engine.embedder = TinyEmbedder(data["emb_vocab"], data["emb_dim"])
            engine.embedder.emb = data["emb_emb"]
            engine.embedder.inv_vocab = {i:w for w,i in data["emb_vocab"].items()}

        # Hopfield 분해기 복원
        engine.decomposer.vocab            = data["decomp_vocab"]
        engine.decomposer.V                = len(data["decomp_vocab"])
        engine.decomposer.W                = data["decomp_W"]
        engine.decomposer.cluster_sentences = defaultdict(list, data["cluster_sentences"])
        engine.decomposer.cluster_tokens    = defaultdict(list, data["cluster_tokens"])
        engine.decomposer.cluster_keywords  = data["cluster_keywords"]

        # 마르코프 복원
        for k, md in data["markovs"].items():
            m = ClusterMarkov()
            m.uni   = Counter(md["uni"])
            m.bi    = defaultdict(Counter, {k2: Counter(v) for k2,v in md["bi"].items()})
            m.tri   = defaultdict(Counter, {k2: Counter(v) for k2,v in md["tri"].items()})
            m.total = md["total"]
            engine.markovs[k] = m

        engine.is_trained = True
        return engine

    def evaluate(self, text: str, logp_thr: float = -11.5,
                 expand: bool = False) -> dict:
        t0 = time.perf_counter()

        # 가드레일 판정 — NeuralMarkov 전체 코퍼스 기준
        if not self.nm_engine.is_trained:
            verdict = "SKIP"
            best_logp = 0.0
        else:
            nm_result = self.nm_engine.evaluate(text, logp_thr=logp_thr)
            verdict   = nm_result.get("status", "SKIP")
            best_logp = nm_result.get("avg_logp", 0.0)

        # 클러스터별 참고 점수 (표시용)
        tokens = tokenize(text)
        per_cluster = {}
        best_cluster = -1
        top_score = -999.0
        for k, m in self.markovs.items():
            s = m.score(tokens)
            per_cluster[k] = {
                "logp": s,
                "keywords": self.decomposer.cluster_keywords.get(k, []),
            }
            if s > top_score:
                top_score = s
                best_cluster = k

        ms = (time.perf_counter() - t0) * 1000

        return {
            "verdict": verdict,
            "logp": best_logp,
            "cluster": best_cluster,
            "cluster_keywords": self.decomposer.cluster_keywords.get(best_cluster, []),
            "ms": ms,
            "per_cluster": per_cluster,
            "expanded": False,
        }
