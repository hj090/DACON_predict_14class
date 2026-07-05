# Agent Action Classifier — Stage 1 (Text-only Baseline)

AI 코딩 에이전트가 대화 도중 특정 시점에서 다음에 취할 행동(action)을
14개 클래스 중 하나로 예측하는 문제의 베이스라인 구현입니다.

현재 단계는 `current_prompt`(사용자의 현재 발화) 텍스트만 사용하는
**TF-IDF + LightGBM** 파이프라인입니다.

## 목차

- [문제 정의](#문제-정의)
- [데이터 구조](#데이터-구조)
- [현재 구현 범위](#현재-구현-범위)
- [설치](#설치)
- [사용법](#사용법)
- [코드 구조](#코드-구조)
- [추론 (Inference)](#추론-inference)
- [하이퍼파라미터](#하이퍼파라미터)
- [평가 지표](#평가-지표)
- [로드맵](#로드맵)

## 문제 정의

한 에이전트 세션의 특정 시점 상태(`session_meta`, `history`, `current_prompt`)가
주어졌을 때, 에이전트가 다음에 수행할 행동을 아래 14개 클래스 중 하나로 예측합니다.

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

> **이 저장소는 아직 `current_prompt`만 사용하는 1단계 베이스라인입니다.**

| 정보 | 상태 |
|---|---|
| `current_prompt` (텍스트) | ✅ 사용 중 |
| `session_meta.language_pref` | 평가(언어별 F1 확인)에만 사용, 모델 입력 아님 |
| `session_meta`, `workspace` (그 외 필드) | ❌ 미사용 (다음 단계 예정) |
| `history` | ❌ 미사용 (다음 단계 예정) |

언어별(한국어/영어/혼합) 분리 모델 대신, 단어 n-gram과 문자(char) n-gram을 함께
쓰는 **단일 파이프라인**으로 혼합 문장에 대응하는 전략을 택했습니다.

## 설치

```bash
pip install scikit-learn lightgbm joblib numpy
```

## 사용법

```
project/
├── data/
│   ├── train.jsonl
│   └── train_labels.csv
├── model/                  # 학습 후 자동 생성
└── script_1.1.1.py
```

```bash
python script_1.1.1.py
```

실행하면 `model/tfidf_lgbm.pkl`(학습된 파이프라인)과
`model/test_set.pkl`(재현 가능한 테스트셋)이 생성됩니다.

## 코드 구조

| 단계 | 내용 |
|---|---|
| 1. 데이터 로드 | `train.jsonl` + `train_labels.csv` → `current_prompt`, `action` 추출 |
| 2. train/test 분할 | `stratify=y`로 클래스 비율 유지, 80/20 분할 |
| 3. TF-IDF 벡터화 | 단어 n-gram(1,2) + 문자 n-gram(2,4)을 `FeatureUnion`으로 결합 |
| 4. 분류기 설정 | `LGBMClassifier(class_weight="balanced")` |
| 5. 학습 | `Pipeline.fit()` |
| 6. 평가 | 전체 macro F1 + 언어별(ko/en/mixed) macro F1 |
| 7. 저장 | 파이프라인 및 테스트셋을 `joblib`으로 직렬화 |

## 추론 (Inference)

학습된 모델을 실제 평가/제출에 사용하는 스크립트는 `script.py`입니다.
`test.jsonl`을 읽어 `sample_submission.csv` 형식에 맞춰 `submission.csv`를 생성합니다.

```bash
python script.py
```

```
project/
├── data/
│   ├── test.jsonl
│   └── sample_submission.csv
├── model/
│   └── tfidf_lgbm.pkl      # 학습 스크립트가 생성한 파일
└── output/
    └── submission.csv      # 자동 생성됨
```

> **변경 이력 (2026-07-05)**: 분류기를 `LogisticRegression` → `LightGBM`으로
> 교체하면서 학습 스크립트의 저장 파일명이 `tfidf_logreg.pkl`에서
> `tfidf_lgbm.pkl`로 바뀌었습니다. `script.py`의 `MODEL_PATH`도 이에 맞춰
> `tfidf_lgbm.pkl`을 참조하도록 함께 수정했습니다. `model/` 폴더에
> 예전 `tfidf_logreg.pkl`이 남아있다면 실제로는 더 이상 사용되지
> 않으니, 혼동을 막기 위해 삭제하거나 `tfidf_logreg_deprecated.pkl`
> 처럼 이름을 구분해서 보관하는 것을 권장합니다.

## 하이퍼파라미터

### TF-IDF

| 파라미터 | 값 | 설명 |
|---|---|---|
| `ngram_range` (word) | (1, 2) | 단어 유니그램+바이그램 |
| `ngram_range` (char) | (2, 4) | 문자 2~4-gram, 언어 경계 무관 |
| `max_features` (word / char) | 60,000 / 20,000 | 어휘 예산 배분 |
| `min_df` | 2 | 2회 미만 등장 토큰 제외 |
| `sublinear_tf` | True | 로그 스케일링 |

### LightGBM

| 파라미터 | 값 | 설명 |
|---|---|---|
| `num_leaves` | 63 | 트리 복잡도 |
| `learning_rate` | 0.05 | 학습률 |
| `n_estimators` | 500 | 트리 개수 |
| `min_child_samples` | 20 | 과적합 억제 |
| `class_weight` | balanced | 클래스 불균형 보정 |

## 평가 지표

- **전체 macro F1**: 14개 클래스를 동일 가중치로 평균
- **언어별 macro F1**: `session_meta.language_pref` 기준 ko/en/mixed 각각 확인
  (모델이 혼합 문장에서 유독 성능이 낮은지 모니터링 목적)

## 로드맵

- [ ] `session_meta` / `workspace` 구조화 피처 추가 (`DictVectorizer`)
- [ ] `history`에서 직전 action, 행동 시퀀스 등 파생 피처 추가
- [ ] Stage 2: multilingual MiniLM 기반 정밀 분류기
- [ ] ONNX 변환 + INT8 양자화 (Stage 2 모델 대상)
- [ ] 언어별 그룹 성능 격차 해소 (특히 혼합 문장)
