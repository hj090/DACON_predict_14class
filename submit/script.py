"""
script.py — 추론 전용 코드 (평가 서버에서 자동 실행)

제출 zip 구조:
  your_submission.zip
  ├── model/            # 학습된 모델 가중치 (vectorizer.pkl 등 5개 + minilm_local/)
  ├── script.py         # 이 파일
  └── requirements.txt

data/, output/은 서버가 자동으로 추가하므로 zip에 포함하지 않습니다.
학습 과정은 포함하지 않고, 추론만 수행합니다.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import re
import csv
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")

BEST_W = 0.8
BEST_THRESHOLD = 0.8
HISTORY_TURNS = 3

# save_results(predictions)가 인자를 하나만 받는 템플릿 구조를 지키기 위해,
# id 순서를 맞추는 데 필요한 sample_submission 정보는 전역 변수로 공유합니다.
_SUB_FIELDNAMES = None
_SUB_ROWS = None


# ============================================================
# 학습 코드와 동일해야 하는 클래스/함수 (pickle 로드 및 피처 재현에 필요)
# ============================================================

class RuleFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(text) for text in X])

    def _extract(self, text):
        return [
            1 if re.search(r'\b(read|열어|읽어)\b', text, re.I) else 0,
            1 if re.search(r'(grep|검색|찾아).*(코드|텍스트|패턴)', text, re.I) else 0,
            1 if re.search(r'(폴더|디렉토리|directory).*(뭐|목록|구조)', text, re.I) else 0,
            1 if re.search(r'\*\.\w+|글롭|glob|확장자', text, re.I) else 0,
            1 if re.search(r'(수정|고쳐|바꿔|edit)', text, re.I) else 0,
            1 if re.search(r'(새로|생성|만들어|write).*(파일|코드)', text, re.I) else 0,
            1 if re.search(r'(diff|patch|패치|변경사항)', text, re.I) else 0,
            1 if re.search(r'(셸|shell|bash|명령어|터미널)', text, re.I) else 0,
            1 if re.search(r'(테스트|test).*(실행|돌려|run)', text, re.I) else 0,
            1 if re.search(r'(린트|lint|타입체크|맞춤법)', text, re.I) else 0,
            1 if '?' in text else 0,
            1 if re.search(r'(계획|설계|plan)', text, re.I) else 0,
            1 if re.search(r'(웹|인터넷|구글|web)', text, re.I) else 0,
            len(text),
            len(re.findall(r'[a-zA-Z]', text)),
        ]


def extract_session_features(sample):
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    lang_mix = ws.get("language_mix") or {}
    dom_lang, dom_ratio = max(lang_mix.items(), key=lambda kv: kv[1]) if lang_mix else ("none", 0.0)
    return {
        "language_pref": meta.get("language_pref", "unknown"),
        "last_ci_status": ws.get("last_ci_status", "none"),
        "git_dirty": str(ws.get("git_dirty", False)),
        "user_tier": meta.get("user_tier", "unknown"),
        "turn_index": meta.get("turn_index", 0),
        "budget_tokens_remaining_log": np.log1p(meta.get("budget_tokens_remaining", 0)),
        "loc_log": np.log1p(ws.get("loc", 0)),
        "n_open_files": len(ws.get("open_files") or []),
        "dominant_code_lang": dom_lang,
        "dominant_code_ratio": dom_ratio,
        "n_code_langs": len(lang_mix),
        "elapsed_session_sec_log": np.log1p(meta.get("elapsed_session_sec", 0)),
    }


def build_stage2_input_text(sample, history_turns=HISTORY_TURNS):
    history = sample.get("history", []) or []
    hist_text = "\n".join(str(turn) for turn in history[-history_turns:])
    return f"{hist_text}\n{sample.get('current_prompt', '')}".strip()


def route_by_confidence(ids, tfidf_probs, rule_probs, classes, w=BEST_W, threshold=BEST_THRESHOLD):
    ensemble_probs = w * tfidf_probs + (1 - w) * rule_probs
    max_conf = ensemble_probs.max(axis=1)
    pred_idx = ensemble_probs.argmax(axis=1)

    stage1_results = {}
    stage2_input_ids = []
    stage2_probs_list = []

    for sample_id, conf, idx, prob_row in zip(ids, max_conf, pred_idx, ensemble_probs):
        if conf >= threshold:
            stage1_results[sample_id] = classes[idx]
        else:
            stage2_input_ids.append(sample_id)
            stage2_probs_list.append(prob_row)

    stage2_probs = np.array(stage2_probs_list) if stage2_probs_list else np.empty((0, len(ALL_CLASSES)))
    return stage1_results, stage2_input_ids, stage2_probs


def _load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}")
            samples.append(obj)
    return samples


def _validate_samples(samples):
    n_bad = 0
    for s in samples:
        for k in REQUIRED_KEYS:
            if k not in s:
                n_bad += 1
                break
    if n_bad:
        print(f" 경고: 필수 키 누락 샘플 {n_bad}건")
    return n_bad


# ============================================================
# 필수 함수 4개 (권장 구조)
# ============================================================

def load_model():
    """모델 로드 함수"""
    model_dir = 'model'
    minilm_dir = os.path.join(model_dir, 'minilm_local')

    if not os.path.isdir(minilm_dir):
        raise FileNotFoundError(
            f"{minilm_dir} 를 찾을 수 없습니다. model/minilm_local 폴더가 제출 zip의 "
            f"model/ 디렉터리 아래에 포함되어 있는지 확인하세요."
        )

    model = {
        "vectorizer": joblib.load(os.path.join(model_dir, 'vectorizer.pkl')),
        "clf": joblib.load(os.path.join(model_dir, 'tfidf_lgbm.pkl')),
        "rule_clf": joblib.load(os.path.join(model_dir, 'rule_logreg.pkl')),
        "session_encoder": joblib.load(os.path.join(model_dir, 'session_encoder.pkl')),
        "stage2_head": joblib.load(os.path.join(model_dir, 'stage2_head.pkl')),
        "embedder": SentenceTransformer(minilm_dir),
        "rule_extractor": RuleFeatureExtractor(),
    }
    return model


def load_data():
    """평가 데이터 로드 함수"""
    global _SUB_FIELDNAMES, _SUB_ROWS

    data_dir = 'data'
    test_path = os.path.join(data_dir, 'test.jsonl')
    sample_sub_path = os.path.join(data_dir, 'sample_submission.csv')

    samples = _load_jsonl(test_path)
    _validate_samples(samples)

    with open(sample_sub_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _SUB_FIELDNAMES = reader.fieldnames
        _SUB_ROWS = list(reader)
    if _SUB_FIELDNAMES is None or _SUB_FIELDNAMES[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission 컬럼이 (id, action)이 아님: {_SUB_FIELDNAMES}")

    data = samples
    return data


def predict(model, data):
    """추론 수행 함수"""
    samples = data
    samples_by_id = {s["id"]: s for s in samples}

    ids = [s.get("id", "") for s in samples]
    texts = [s.get("current_prompt", "") or "" for s in samples]

    # ---- Stage1 특징 ----
    X_vec = model["vectorizer"].transform(texts)
    rule_features = model["rule_extractor"].transform(texts)

    session_df = pd.DataFrame([extract_session_features(s) for s in samples])
    session_enc = model["session_encoder"].transform(session_df)

    X_final = hstack([X_vec, session_enc])

    tfidf_probs = model["clf"].predict_proba(X_final)
    rule_probs = model["rule_clf"].predict_proba(rule_features)

    # ---- Stage1 라우팅 ----
    stage1_results, stage2_input_ids, stage2_probs = route_by_confidence(
        ids=ids, tfidf_probs=tfidf_probs, rule_probs=rule_probs, classes=model["clf"].classes_,
    )
    stage1_ratio = len(stage1_results) / len(ids) if ids else 0.0
    print(f"Stage1 확정: {len(stage1_results)}건 ({stage1_ratio:.1%}) | Stage2로: {len(stage2_input_ids)}건")

    # ---- Stage2 처리 (확신도 낮은 샘플만, 시간 제한 고려) ----
    stage2_results = {}
    if stage2_input_ids:
        stage2_texts = [build_stage2_input_text(samples_by_id[sid]) for sid in stage2_input_ids]

        stage2_session_df = pd.DataFrame(
            [extract_session_features(samples_by_id[sid]) for sid in stage2_input_ids]
        )
        stage2_session_enc = model["session_encoder"].transform(stage2_session_df)
        if hasattr(stage2_session_enc, "toarray"):
            stage2_session_enc = stage2_session_enc.toarray()

        stage2_emb = model["embedder"].encode(stage2_texts, show_progress_bar=True)
        stage2_feat = np.hstack([stage2_emb, stage2_probs, stage2_session_enc])
        stage2_preds = model["stage2_head"].predict(stage2_feat)

        stage2_results = dict(zip(stage2_input_ids, stage2_preds))
        print(f"Stage2 예측 완료: {len(stage2_results)}건")

    # ---- 결과 합치기 (id -> action) ----
    final_results = {**stage1_results, **stage2_results}
    predictions = {i: str(final_results.get(i, "")) for i in ids}
    return predictions


def save_results(predictions):
    """결과 저장 함수"""
    # output/submission.csv로 저장 (필수)
    if _SUB_FIELDNAMES is None or _SUB_ROWS is None:
        raise RuntimeError("load_data()가 먼저 실행되어야 합니다 (sample_submission 정보 없음).")

    n_missing = 0
    for row in _SUB_ROWS:
        p = predictions.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 id {n_missing}건")

    os.makedirs('output', exist_ok=True)
    out_path = os.path.join('output', 'submission.csv')
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUB_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_SUB_ROWS)
    print(f"Saved: {out_path} (rows={len(_SUB_ROWS)})")


if __name__ == "__main__":
    # 메인 실행 코드
    model = load_model()
    data = load_data()
    predictions = predict(model, data)
    save_results(predictions)
    print("추론 완료!")
