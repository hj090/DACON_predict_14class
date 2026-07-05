# Agent Action Classifier — Stage 1 (Text-only Baseline)

AI 코딩 에이전트가 대화 도중 특정 시점에서 다음에 취할 행동(action)을
14개 클래스 중 하나로 예측하는 문제의 베이스라인 구현입니다.

현재 실제로 운영 중인 모델은 `current_prompt`(사용자의 현재 발화) 텍스트만
사용하는 **TF-IDF + LogisticRegression** 파이프라인입니다.

> ⚠️ 이 문서는 실제 코드를 기준으로 작성되었습니다.
> 과거 논의 과정에서 LightGBM/문자 n-gram 병행/언어별 평가 등의 대안이
> 검토된 적이 있으나, **실제로 채택되어 코드에 반영된 적은 없습니다.**
> 이 문서에서는 실제 코드와 향후 검토 사항을 명확히 구분해서 표기합니다.
>
> **2026-07-05 업데이트**: 학습(`script_1.1.1.py`)과 추론(`script.py`)이
> `script_full.py` 하나로 통합되었습니다. 기존 두 파일은 더 이상 사용하지
> 않습니다 (레거시로만 보관).

## 목차

- [문제 정의](#문제-정의)
- [데이터 구조](#데이터-구조)
- [현재 구현 범위](#현재-구현-범위)
- [설치](#설치)
- [사용법](#사용법)
- [코드 구조](#코드-구조)
- [하이퍼파라미터](#하이퍼파라미터)
- [확인이 필요한 부분](#확인이-필요한-부분)
- [레거시 파일](#레거시-파일)
- [로드맵 (검토/제안 단계, 미반영)](#로드맵-검토제안-단계-미반영)

## 문제 정의

한 에이전트 세션의 특정 시점 상태(`session_meta`, `history`, `current_prompt`)가
주어졌을 때, 에이전트가 다음에 수행할 행동을 아래 14개 클래스 중 하나로 예측합니다.
(`ALL_CLASSES` 리스트로 코드에 명시되어 있으며 Macro-F1 계산에 사용됩니다.)

| 카테고리 | 행동(action) |
|---|---|
| 탐색 | `read_file`, `grep_search`, `list_directory`, `glob_pattern` |
| 수정 | `edit_file`, `write_file`, `apply_patch` |
| 실행/검증 | `run_bash`, `run_tests`, `lint_or_typecheck` |
| 소통/계획 | `ask_user`, `plan_task`, `web_search`, `respond_only` |

## 데이터 구조

```
data/
├── train.jsonl            # 학습 입력 (70,000건, JSON Lines)
├── train_labels.csv       # 학습 정답 (id, action)
├── test.jsonl             # 평가 입력 형식 확인용 샘플 (5건)
└── sample_submission.csv  # 제출 양식 (id, action)
```

`train.jsonl` / `test.jsonl`의 한 줄(샘플)은 다음 구조를 가집니다.

```json
{
  "id": "sess_sim_20260522_028750-step_02",
  "session_meta": {
    "user_tier": "pro",
    "language_pref": "ko",
    "budget_tokens_remaining": 12000,
    "turn_index": 3,
    "elapsed_session_sec": 145,
    "workspace": {
      "language_mix": {"py": 0.45, "sql": 0.30},
      "loc": 5200,
      "git_dirty": true,
      "open_files": ["app/main.py"],
      "last_ci_status": "failed"
    }
  },
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant_action", "name": "read_file", "args": {...}, "result_summary": "..."}
  ],
  "current_prompt": "이 파일 배송지 로직 좀 고쳐줘"
}
```

## 현재 구현 범위

| 정보 | 상태 |
|---|---|
| `current_prompt` (텍스트) | ✅ 사용 중 (유일한 모델 입력) |
| `session_meta`, `workspace` | ❌ 미사용 |
| `history` | ❌ 미사용 |

`current_prompt` 하나만으로 TF-IDF(단어 단위) + LogisticRegression을
학습하는 것이 현재 코드의 전부입니다. 언어별(한국어/영어/혼합) 분리나
문자 n-gram 병행 등은 **논의만 되었을 뿐 코드에는 없습니다.**

## 설치

```bash
pip install scikit-learn joblib
```

## 사용법

```
project/
├── data/
│   ├── train.jsonl
│   ├── train_labels.csv
│   ├── test.jsonl
│   └── sample_submission.csv
├── model/                  # 학습 후 자동 생성
├── output/                 # 추론 후 자동 생성
└── script_full.py
```

```bash
python script_full.py
```

한 번 실행으로 다음이 순서대로 수행됩니다.

1. `train.jsonl` + `train_labels.csv`로 학습 (`train()` 함수)
   → 검증 세트 Macro-F1 출력 → 전체 데이터로 재학습 → `model/tfidf_logreg.pkl` 저장
2. 방금 학습한 모델로 바로 `test.jsonl` 추론 (`infer()` 함수)
   → `output/submission.csv` 생성

## 코드 구조

`script_full.py`는 크게 세 부분으로 구성됩니다.

| 구역 | 함수 | 내용 |
|---|---|---|
| 공통 유틸 | `load_jsonl`, `validate_samples`, `extract_text`, `build_features`, `load_sample_submission`, `merge_predictions`, `save_submission` | 학습/추론 양쪽에서 재사용하는 데이터 입출력 함수 |
| 학습 | `train()` | 아래 표 참고 |
| 추론 | `infer(pipe)` | `test.jsonl` 로드 → 예측 → `submission.csv` 생성 |

`train()` 내부 단계:

| 단계 | 내용 |
|---|---|
| 1. 데이터 로드 | `train.jsonl` + `train_labels.csv` → `current_prompt`, `action` 추출 |
| 2. train/val 분할 | `stratify=y`, `test_size=0.2`, `random_state=42`로 80/20 분할 |
| 3. TF-IDF 벡터화 | 단어 n-gram(1,2), `min_df=2`, `max_features=80,000` |
| 4. 분류기 | `LogisticRegression(max_iter=500, class_weight="balanced", C=2.0)` |
| 5. 1차 학습 | `pipe.fit(X_train, y_train)` — train(80%)으로만 학습 |
| 6. 검증 | `f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)` 로 Macro-F1 출력 |
| 7. 재학습 | `pipe.fit(X, y)` — **전체 데이터(100%)로 다시 학습** |
| 8. 저장 | `joblib.dump(pipe, "./model/tfidf_logreg.pkl", compress=3)` |

> 5번과 7번이 서로 다른 학습이라는 점에 주의하세요. 5번은 검증(val)용으로
> train(80%)만 학습한 모델이고, 7번은 실제로 저장/배포되는 모델로
> 전체 데이터(100%)를 다시 학습시킨 것입니다. **6번에서 확인한 Macro-F1은
> 5번 모델 기준이며, 7번(최종 저장 모델)의 성능을 그대로 보장하지는
> 않습니다** (데이터가 늘었으므로 비슷하거나 더 나을 가능성이 높지만
> 별도로 검증된 사실은 아님).

`infer(pipe)`는 `train()`이 반환한 모델 객체를 그대로 받아서 예측하므로,
같은 실행 안에서는 `tfidf_logreg.pkl`을 다시 읽어올 필요가 없습니다.
(저장된 `.pkl`은 재사용/배포용으로 남겨두는 것입니다.)

## 하이퍼파라미터

### TF-IDF

| 파라미터 | 값 |
|---|---|
| `ngram_range` | (1, 2) |
| `min_df` | 2 |
| `max_features` | 80,000 |
| `sublinear_tf` | True |
| `lowercase` | True |

### LogisticRegression

| 파라미터 | 값 |
|---|---|
| `max_iter` | 500 |
| `class_weight` | balanced |
| `C` | 2.0 |

## 확인이 필요한 부분

- **언어별 성능 확인 로직 없음**: 데이터에 `session_meta.language_pref`(ko/en/mixed)
  필드가 존재하지만, 현재 코드는 이를 전혀 참조하지 않습니다. 혼합 문장(전체의 약 10%)
  에서 성능이 유독 낮은지 여부는 아직 실제로 측정된 적이 없습니다.

## 레거시 파일

`script_1.1.1.py`(학습 전용), `script.py`(추론 전용)는 `script_full.py`로
통합되면서 더 이상 사용하지 않습니다. 참고용으로만 남겨두었으며,
실제 실행은 `script_full.py` 하나로 하시면 됩니다.

## 로드맵 (검토/제안 단계, 미반영)

아래 항목들은 작업 중 논의되었으나 **실제 코드에는 반영되지 않은 제안 사항**입니다.
채택 여부는 위 "확인이 필요한 부분"의 baseline 수치를 먼저 확보한 뒤 결정 권장.

- [x] val 평가 코드 추가/확인 (Macro-F1 출력 확인됨, script_full.py 반영 완료)
- [ ] `session_meta.language_pref` 기준 언어별 macro F1 분리 확인
- [ ] 문자(char) n-gram 병행 검토 (혼합 문장 대응 목적)
- [ ] `session_meta` / `workspace` 구조화 피처 추가 검토
- [ ] `history`에서 직전 action 등 파생 피처 추가 검토
- [ ] LightGBM 등 다른 분류기로 교체 검토 (LogisticRegression과 비교 실험 필요)
- [ ] Stage 2: multilingual MiniLM 기반 정밀 분류기
- [ ] ONNX 변환 + INT8 양자화 (Stage 2 모델 대상)
