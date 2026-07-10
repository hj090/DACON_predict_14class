"""
학습(train) + 추론(inference)을 하나로 합친 스크립트.

1) train.jsonl + train_labels.csv로 모델을 학습하고 Macro-F1로 검증
2) 전체 데이터로 재학습 후 모델 저장
3) 같은 실행 안에서 바로 test.jsonl에 대해 예측
4) sample_submission.csv 형식에 맞춰 submission.csv 생성
"""

import csv
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 예측 대상 14개 클래스 (Macro-F1 계산에 사용)
ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")

DATA_DIR = "./data"
MODEL_DIR = "./model"
OUT_DIR = "./output"

TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg.pkl")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")


# ============================================================
# 공통 유틸 (학습/추론 양쪽에서 재사용)
# ============================================================

def load_jsonl(path):
    """jsonl 로드. 한 줄당 샘플 하나."""
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


def validate_samples(samples):
    """필수 키 존재 여부 검증."""
    n_bad = 0
    for s in samples:
        for k in REQUIRED_KEYS:
            if k not in s:
                n_bad += 1
                break
    if n_bad:
        print(f" 경고: 필수 키 누락 샘플 {n_bad}건 (빈 텍스트로 처리)")
    return n_bad


def extract_text(sample):
    """모델 입력 텍스트 추출 — current_prompt만 사용."""
    text = sample.get("current_prompt", "")
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text


def build_features(samples):
    """샘플 리스트 → (ids, 모델 입력 텍스트 리스트)."""
    ids = []
    texts = []
    for s in samples:
        ids.append(s.get("id", ""))
        texts.append(extract_text(s))
    n_empty = sum(1 for t in texts if not t.strip())
    if n_empty:
        print(f" 경고: current_prompt가 비어있는 샘플 {n_empty}건")
    return ids, texts


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 id 순서/컬럼 기준."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission 컬럼이 (id, action)이 아님: {fieldnames}")
    return fieldnames, rows


def merge_predictions(sub_rows, ids, preds):
    """sample_submission의 id 순서에 맞춰 예측값 병합."""
    pred_map = dict(zip(ids, preds))
    n_missing = 0
    for row in sub_rows:
        p = pred_map.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 id {n_missing}건")
    return sub_rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 1. 학습
# ============================================================

def train():
    # train.jsonl: 한 줄 = 샘플 하나
    samples = [json.loads(line)
               for line in open(TRAIN_PATH, encoding="utf-8")
               if line.strip()]

    # train_labels.csv: id -> action 매핑
    labels = {row["id"]: row["action"]
              for row in csv.DictReader(open(TRAIN_LABELS_PATH, encoding="utf-8"))}

    # 입력 X = current_prompt, 정답 y = action
    X = [s["current_prompt"] for s in samples]
    y = [labels[s["id"]] for s in samples]
    print("samples:", len(X), "| classes:", len(set(y)))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    print("train:", len(X_train), "| val:", len(X_val))

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=80_000,
            sublinear_tf=True, lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=500, class_weight="balanced", C=2.0,
        )),
    ])

    pipe.fit(X_train, y_train)
    print("학습 완료")

    # 5. 검증 — Macro-F1
    # 검증 세트로 성능을 확인합니다. Macro-F1은 14개 클래스 각각의 F1 점수를
    # 동일한 가중치로 평균하므로, 자주 등장하지 않는 클래스도 똑같이 중요하게 평가합니다.
    val_pred = pipe.predict(X_val)
    macro_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)
    print(f"Validation Macro-F1: {macro_f1:.4f}")

    # 전체 학습 데이터로 재학습
    pipe.fit(X, y)

    # 저장
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    print(f"저장 완료: {MODEL_PATH}")

    return pipe


# ============================================================
# 2. 추론
# ============================================================

def infer(pipe):
    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    validate_samples(samples)
    print(f" samples={len(samples)}")

    print("Build features...")
    ids, texts = build_features(samples)
    print(f" texts={len(texts)}")

    print("Inference model...")
    preds = pipe.predict(texts) if texts else []
    preds = [str(p) for p in preds]
    print(f" preds={len(preds)}")

    print("Build submission...")
    fieldnames, sub_rows = load_sample_submission(SAMPLE_SUB_PATH)
    sub_rows = merge_predictions(sub_rows, ids, preds)
    save_submission(OUT_PATH, fieldnames, sub_rows)
    print(f"Saved: {OUT_PATH} (rows={len(sub_rows)})")


# ============================================================
# 실행
# ============================================================

def main():
    pipe = train()
    infer(pipe)


if __name__ == "__main__":
    main()
