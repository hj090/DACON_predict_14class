"""
script.py — 추론 전용 코드 (평가 서버에서 자동 실행)

제출 zip 구조:
  your_submission.zip
  ├── model/            # 학습된 모델 가중치 (vectorizer.pkl 등 + minilm_local/)
  ├── script.py         # 이 파일
  └── requirements.txt

data/, output/은 서버가 자동으로 추가하므로 zip에 포함하지 않습니다.
학습 과정은 포함하지 않고, 추론만 수행합니다.

이번 버전 변경점:
  - Stage1도 이제 history를 사용 (extract_history_features 추가)
  - session_encoder는 session+history를 합친 df로 transform (학습 때와 동일)
  - BEST_W = 1.0, BEST_THRESHOLD = 0.9 (학습 검증 결과 반영)
  - Stage1 분류기가 LightGBM(.pkl)에서 ONNX(.onnx)로 변경됨 (경량화).
    onnxruntime.InferenceSession으로 로드하고, run()의 두 번째 출력
    (outputs[1], shape=(N,14))이 클래스별 확률입니다. 실제 로컬
    검증으로 확인된 구조입니다 (outputs[0]=예측 라벨 문자열, 미사용).
"""

import os

# 평가 서버는 인터넷이 안 되므로, transformers/huggingface_hub가 모델 정보를
# 확인하러 인터넷에 접속을 시도하다 타임아웃으로 시간을 낭비하는 것을 방지
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re
import csv
import json

import joblib
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

# sklearn/lightgbm이 학습 시 클래스를 알파벳순으로 정렬해서 저장하는데,
# ONNX 세션에는 원본처럼 .classes_ 속성이 없으므로 그 순서를 직접 고정해둠
# (실제 clf_final.classes_/rule_only_clf_final.classes_ 출력으로 확인된 순서:
#  ['apply_patch' 'ask_user' 'edit_file' 'glob_pattern' 'grep_search'
#   'lint_or_typecheck' 'list_directory' 'plan_task' 'read_file' 'respond_only'
#   'run_bash' 'run_tests' 'web_search' 'write_file'])
CLASSES_SORTED = sorted(ALL_CLASSES)

REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")

BEST_W = 1.0            # Stage1 앙상블 가중치 (학습 검증 결과 반영: w=1.0 -> rule_probs 미반영)
BEST_THRESHOLD = 0.9    # 이 확신도 이상이면 Stage1에서 바로 확정
HISTORY_TURNS = 3       # Stage2 입력 텍스트에 포함할 최근 history 턴 수
EMBED_BATCH_SIZE = 128  # Stage2 임베딩 배치 크기 (기본 32보다 키움)
# GPU가 있으면 자동으로 사용, 없으면 CPU로 안전하게 대체
# (device="cuda"로 하드코딩하면 GPU가 없는 환경에서 바로 에러가 남 — 실제로 발생했던 문제)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    """session_meta, workspace에서 파생 피처 dict 생성 (학습 코드 #5와 동일 로직)."""
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


def extract_history_features(sample):
    """history에서 직전 action 관련 파생 피처 dict 생성 (학습 코드 #6과 동일 로직).
    학습 코드는 samples_by_id[sample_id]로 다시 조회하는 구조였는데, 여기서는
    샘플 dict를 바로 받도록 정리함 (extract_session_features와 동일한 방식)."""
    history = sample.get("history", []) or []

    last_action = None
    turns_since = 0
    for turn in reversed(history):
        if turn.get("role") == "assistant_action":
            last_action = turn
            break
        turns_since += 1

    if last_action is None:
        return {
            "last_action_name": "none",
            "last_action_result_status": "none",
            "n_actions_so_far": 0,
            "last_action_ext": "none",
            "turns_since_last_action": len(history),
        }

    result_summary = (last_action.get("result_summary") or "").lower()
    if re.search(r"fail|error|exception|traceback", result_summary):
        result_status = "fail"
    elif re.search(r"\bok\b|success|passed", result_summary):
        result_status = "ok"
    else:
        result_status = "unknown"

    path = (last_action.get("args") or {}).get("path", "")
    ext_match = re.search(r"\.(\w+)$", path)
    ext = ext_match.group(1) if ext_match else "none"

    n_actions = sum(1 for turn in history if turn.get("role") == "assistant_action")

    return {
        "last_action_name": last_action.get("name", "none"),
        "last_action_result_status": result_status,
        "n_actions_so_far": n_actions,
        "last_action_ext": ext,
        "turns_since_last_action": turns_since,
    }


def build_combined_features_df(samples):
    """session_meta + history 파생 피처를 합쳐 하나의 DataFrame으로 (학습 코드의
    combined_train_df/combined_val_df와 동일한 방식: pd.concat)."""
    session_df = pd.DataFrame([extract_session_features(s) for s in samples])
    history_df = pd.DataFrame([extract_history_features(s) for s in samples])
    return pd.concat(
        [session_df.reset_index(drop=True), history_df.reset_index(drop=True)], axis=1
    )


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


def _run_onnx_probs(session, X_final):
    """ONNX로 변환된 Stage1 LightGBM 세션을 실행해 클래스별 확률만 반환.
    실제 로컬 검증 결과: outputs[0]=예측 라벨(문자열, 미사용),
    outputs[1]=확률(shape=(N,14), float32) — 반드시 이 순서 그대로."""
    input_name = session.get_inputs()[0].name
    X_dense = X_final.toarray().astype(np.float32)
    outputs = session.run(None, {input_name: X_dense})
    return outputs[1]


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
        "clf": ort.InferenceSession(os.path.join(model_dir, 'tfidf_lgbm.onnx')),
        "rule_clf": joblib.load(os.path.join(model_dir, 'rule_logreg.pkl')),
        "session_encoder": joblib.load(os.path.join(model_dir, 'session_encoder.pkl')),
        "stage2_head": joblib.load(os.path.join(model_dir, 'stage2_head.pkl')),
        "embedder": SentenceTransformer(minilm_dir, device=DEVICE),
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

    # ---- Stage1 특징: 텍스트(TF-IDF+chi2+규칙) + (session+history) ----
    X_vec = model["vectorizer"].transform(texts)
    rule_features = model["rule_extractor"].transform(texts)

    combined_df = build_combined_features_df(samples)
    session_enc = model["session_encoder"].transform(combined_df)

    X_final = hstack([X_vec, session_enc])

    tfidf_probs = _run_onnx_probs(model["clf"], X_final)
    rule_probs = model["rule_clf"].predict_proba(rule_features)

    # ---- Stage1 확신도 라우팅 ----
    stage1_results, stage2_input_ids, stage2_probs = route_by_confidence(
        ids=ids, tfidf_probs=tfidf_probs, rule_probs=rule_probs, classes=CLASSES_SORTED,
    )
    stage1_ratio = len(stage1_results) / len(ids) if ids else 0.0
    print(f"Stage1 확정: {len(stage1_results)}건 ({stage1_ratio:.1%}) | Stage2로: {len(stage2_input_ids)}건")

    # ---- Stage2: 확신도 낮은 샘플만 history+임베딩으로 재판단 ----
    stage2_results = {}
    if stage2_input_ids:
        stage2_samples = [samples_by_id[sid] for sid in stage2_input_ids]
        stage2_texts = [build_stage2_input_text(s) for s in stage2_samples]

        stage2_combined_df = build_combined_features_df(stage2_samples)
        stage2_session_enc = model["session_encoder"].transform(stage2_combined_df)
        if hasattr(stage2_session_enc, "toarray"):
            stage2_session_enc = stage2_session_enc.toarray()

        stage2_emb = model["embedder"].encode(
            stage2_texts,
            batch_size=EMBED_BATCH_SIZE,
            device=DEVICE,
            show_progress_bar=False,
        )
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
