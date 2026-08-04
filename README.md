# UCI SECOM 비용 민감 불량 스크리닝 & XAI 가이던스 PoC

> 공개 반도체 센서 데이터를 활용해 **데이터 누수 없는 모델 검증**, **운영 비용을 반영한 판단 기준 선정**, **엔지니어가 확인할 수 있는 예측 근거 제공**까지 구현한 개인 프로젝트입니다.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-7B61FF)](https://shap.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Data-UCI%20SECOM-555555)](https://archive.ics.uci.edu/dataset/179/secom)

## 1. 프로젝트 개요

반도체 공정은 불량을 놓치지 않는 것이 중요하지만, 모든 샘플을 검사하면 검사량과 운영 비용이 증가합니다. 이 프로젝트는 UCI SECOM 데이터로 다음 질문을 검토했습니다.

1. 불균형 데이터에서 불량 검출 성능을 신뢰할 수 있게 평가하려면 어떻게 해야 하는가?
2. 미검출 비용과 검사량을 함께 고려해 배포 판단 기준을 어떻게 정할 것인가?
3. 모델의 예측 결과를 엔지니어가 검토 가능한 근거와 확인 항목으로 어떻게 연결할 것인가?

### 구현 범위

- 5×5 Repeated Nested CV 기반의 leakage-free 검증
- None·SMOTE·GMM+PCA 증강 기법 비교 및 통계 검정
- Inner OOF 결과를 이용한 Cost Ratio와 Threshold 자동 선정
- Global·Local SHAP 기반 예측 근거 설명
- Risk Screening → XAI Guidance → Engineer Check 형태의 오프라인 Agent PoC

### 이 프로젝트가 주장하지 않는 것

- 실제 SK하이닉스 또는 Etch 공정 데이터를 사용한 프로젝트가 아닙니다.
- 실제 FDC 시스템이나 Virtual Metrology 모델을 구현한 것이 아닙니다.
- MES·설비 API 연동, Recipe 변경, Interlock 실행 등 생산 제어 기능은 포함하지 않습니다.
- LLM은 계산된 결과를 요약할 뿐, 검사 여부를 판단하거나 설비를 제어하지 않습니다.

## 2. 핵심 결과

운영 조건을 **Recall ≥ 65%, 검사 대상 감소 ≥ 55%**로 정의하고, Inner OOF에서 조건을 만족하는 최소 Cost Ratio를 선택했습니다.

| 구분 | 결과 |
|---|---:|
| 선택된 증강 기법 | SMOTE |
| 배포 Operating Point | FN:FP = **13:1** |
| 대표 Inner OOF Threshold | **0.00571** |
| Outer Test Recall | **68.7%** |
| Outer Test Precision | **11.7%** |
| Outer Test AUC | **0.719** |
| 검사 대상 감소 | **60.1%** |

> 성능 수치는 25개 Outer Test fold의 평균입니다. 모델과 Threshold는 각 Outer Train 내부 OOF 결과만으로 선택했으며, Outer Test는 최종 평가에만 사용했습니다. `0.00571`은 fold별 선정값을 설명하기 위한 대표값으로, 모든 데이터에 고정 적용하는 범용 Threshold가 아닙니다.

## 3. 데이터와 문제 정의

- **데이터:** [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom)
- **규모:** 1,567개 공정 샘플, 590개 익명 센서 변수
- **목표:** 정상/불량 이진 분류 및 검사 대상 선별
- **특성:** 결측치, 고차원 센서 변수, 심한 클래스 불균형

전처리 과정에서는 결측률이 높은 변수와 분산이 없는 변수를 제거하고, 학습 fold 내부에서만 결측치 대치·스케일링·Feature Selection·증강을 수행했습니다.

## 4. Leakage-free 검증 설계

```text
Outer Train
  └─ Inner 5-Fold CV
       ├─ 전처리·Feature Selection
       ├─ 증강 및 모델 학습
       ├─ Inner OOF 예측 생성
       └─ 모델·Cost Ratio·Threshold 선정

Outer Test
  └─ Outer Train에서 확정한 모델과 Threshold로 최종 평가
```

위 과정을 5개 fold × 5개 seed로 반복해 총 25개의 독립적인 Outer Test 결과를 얻었습니다. Feature Selection, SMOTE, 모델 학습, Threshold 탐색은 모두 Outer Train 안에서 수행해 평가 데이터가 선택 과정에 섞이지 않도록 했습니다.

## 5. 증강 기법 비교

먼저 동일한 데이터 분할과 seed를 사용하고, Cost Ratio를 15:1로 고정한 상태에서 세 가지 방법을 비교했습니다.

| 방법 | Recall | Precision | AUC | 검사 대상 감소 |
|---|---:|---:|---:|---:|
| None | 69.8% | 12.3% | 0.723 | 61.3% |
| GMM + PCA | 62.7% | 13.1% | 0.732 | 67.4% |
| **SMOTE** | **71.1%** | 11.6% | 0.721 | 58.4% |

Wilcoxon signed-rank test 결과, GMM+PCA는 Recall이 열위여서 제외했습니다. None과 SMOTE의 차이는 통계적으로 유의하지 않았지만, 미검출 최소화라는 목적에 따라 평균 Recall이 가장 높은 SMOTE를 후속 실험에 사용했습니다.

> 이 표는 **증강 기법 비교를 위한 고정 15:1 결과**입니다. 위의 최종 성능은 자동 선정된 **13:1 Operating Point**를 Outer Test에서 평가한 결과이므로 두 수치를 구분해야 합니다.

## 6. Operating Point 자동 선정

단순히 0.5 Threshold를 사용하는 대신, 불량 미검출과 추가 검사의 상대 비용을 반영했습니다.

```text
Total Cost = Cost Ratio × FN + 1 × FP
```

선정 절차는 다음과 같습니다.

1. 각 Outer Train의 Inner OOF 예측으로 Cost Ratio 후보별 Threshold를 탐색합니다.
2. Recall ≥ 65%, 검사 대상 감소 ≥ 55%를 만족하는지 확인합니다.
3. 조건을 만족하는 **최소 Cost Ratio**를 선택합니다.
4. 선택된 Ratio에서 Total Cost가 가장 낮은 Threshold를 확정합니다.
5. 확정된 모델과 Threshold를 해당 Outer Test에 한 번만 적용합니다.

그 결과 배포 후보 Operating Point는 **13:1**, 대표 Threshold는 **0.00571**로 선정됐습니다.

## 7. XAI 기반 엔지니어 의사결정 지원

Agent는 모델의 판단을 대신하는 자동 제어기가 아니라, 예측 결과를 엔지니어가 검토할 수 있는 형태로 정리하는 오프라인 의사결정 지원 PoC입니다.

```text
Risk Screening
  → Threshold Rule로 검사 대상 선별
  → Global·Local SHAP으로 주요 센서 근거 확인
  → 동일 설비·시간대 데이터 비교 등 확인 항목 제시
  → LLM이 검증 결과를 현장 문장으로 요약
  → 엔지니어가 최종 확인 및 조치 판단
```

### 역할 구분

| 구성 요소 | 역할 |
|---|---|
| Model | 불량 Risk Score 산출 |
| Rule | Threshold에 따라 PASS/INSPECT 후보 분류 |
| SHAP | 전체 모델 경향과 개별 샘플의 판단 근거 설명 |
| LLM | 계산·검증된 결과만 자연어로 요약 |
| Engineer | 센서 이상 여부 확인 및 최종 조치 결정 |

## 8. 주요 발견

### 데이터 누수 방지

전처리와 증강을 전체 데이터에 먼저 적용하면 검증 성능이 과대평가될 수 있습니다. 모든 학습 절차를 CV fold 내부로 제한하고, Inner OOF와 Outer Test의 역할을 분리했습니다.

### GMM+PCA artifact 확인

GMM+PCA 합성 샘플에서 일부 실제 데이터 범위를 벗어난 값이 관찰됐습니다. 단순 성능 비교에 그치지 않고 합성 데이터의 분포를 확인했으며, Recall 열위와 함께 현장 해석 위험을 고려해 제외했습니다.

### 예측과 조치의 분리

익명화된 공개 데이터만으로 실제 Recipe 처방을 제안하는 것은 적절하지 않습니다. 따라서 Agent는 예측·설명·확인 항목까지만 제공하고, 실제 조치는 공정 지식을 가진 엔지니어가 결정하도록 역할을 분리했습니다.

## 9. 실행 방법

```bash
git clone https://github.com/heejaelee712/secom-vmfdc-agent.git
cd secom-vmfdc-agent
jupyter notebook main_pipeline.ipynb
```

주요 라이브러리:

```text
numpy, pandas, scipy, scikit-learn, imbalanced-learn,
matplotlib, seaborn, shap, openai
```

OpenAI API Key는 LLM 요약 예시를 실행할 때만 필요합니다. 모델 학습·검증·SHAP 분석은 API Key 없이 실행할 수 있습니다.

## 10. Repository 구조

```text
secom-vmfdc-agent/
├── main_pipeline.ipynb   # 데이터 분석, 검증, XAI·Agent PoC
├── README.md
└── LICENSE
```

## 11. 한계와 다음 단계

- 센서명이 익명화되어 있어 실제 공정 변수와의 매핑이 필요합니다.
- 공개 소규모 데이터에 대한 결과이므로 실제 Fab 데이터에서 외부 검증이 필요합니다.
- Precision이 낮아 엔지니어의 추가 검토 부담을 줄이기 위한 후속 개선이 필요합니다.
- Cost Ratio와 운영 조건은 실제 미검출 비용·검사 Capacity를 반영해 다시 산정해야 합니다.
- 생산 적용 전 Shadow Test, 공정 전문가 검토, 데이터·모델 모니터링 체계가 필요합니다.

## 용어 안내

기존 Notebook 일부에는 `VM`, `EtchProcessAgent` 같은 초기 실험 단계의 이름이 남아 있습니다. 본 프로젝트의 실제 범위는 **UCI SECOM 기반 불량 스크리닝 및 엔지니어 의사결정 지원 PoC**이며, 해당 명칭이 실제 Virtual Metrology 또는 Etch 제어 기능을 의미하지는 않습니다.

## Reference

- [UCI Machine Learning Repository: SECOM](https://archive.ics.uci.edu/dataset/179/secom)
- [Project Repository](https://github.com/heejaelee712/secom-vmfdc-agent)
