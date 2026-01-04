---
title: "[논문 리뷰] SeedFold: Scaling Biomolecular Structure Prediction"
date: "2026-01-04"
excerpt: "Highly accurate biomolecular structure prediction is a key component of developing biomolecular foundation models, and one of the most critical aspects of building foundation models is identifying the..."
category: "Paper Review"
tags: ["Paper Review","q-bio.BM","q-bio.BM"]
thumbnail: "/assets/images/blog/20260104-paper-2512-24354-seedfold-scaling-biomolecular-.jpg"
---

# [논문 리뷰] SeedFold: Scaling Biomolecular Structure Prediction

## TL;DR

SeedFold는 생체분자 구조 예측의 정확성을 높이기 위해 설계된 새로운 모델로, 모델의 용량을 효과적으로 확장하는 방법을 제시합니다. 이 연구는 Pairformer의 폭을 확장하고, 선형 삼각형 주의 메커니즘을 도입하여 계산 복잡성을 줄이며, 대규모 증류 데이터셋을 구축하여 훈련 데이터를 확장합니다. 실험 결과, SeedFold는 AlphaFold3를 능가하는 성능을 보이며, 특히 단백질 단량체 및 단백질-단백질 복합체 예측에서 뛰어난 결과를 얻었습니다. 이 연구는 생체분자 구조 예측의 새로운 가능성을 열어주며, 향후 전문가 혼합 및 사후 훈련 확장 기술을 통해 더욱 발전할 것입니다.  특히, 구조 예측 성능 향상 뿐만 아니라, 계산 효율성 개선에 기여하여, 실제 생명과학 연구에 더 쉽게 적용할 수 있도록 합니다.

## 연구 배경 및 동기

생체분자 구조 예측은 생명과학 분야에서 매우 중요한 문제로, 단백질의 3차원 구조를 정확히 예측하는 것은 신약 개발 및 질병 치료에 필수적입니다. 전통적으로, 실험적 방법(예: X선 결정학, NMR 등)은 시간이 많이 걸리고 비용이 많이 드는 경향이 있습니다. 최근, AlphaFold와 같은 기계 학습 기반의 방법들이 등장하면서, 생체분자 구조 예측의 정확성과 효율성이 크게 향상되었습니다. 그러나 이러한 모델들은 여전히 많은 계산 자원을 필요로 하며, 모델의 확장성에 한계가 있습니다. AlphaFold2의 경우, 막대한 GPU 자원을 필요로 하며, 예측 시간이 오래 걸리는 단점이 있습니다.

SeedFold는 이러한 문제를 해결하기 위해 제안되었습니다. 기존의 AlphaFold 모델은 깊이를 확장하는 방식으로 성능을 향상시켰지만, 이는 vanishing gradient 문제와 같은 어려움을 야기할 수 있습니다. SeedFold는 모델의 폭을 확장함으로써 이러한 문제를 피하고, 새로운 선형 삼각형 주의 메커니즘을 통해 계산 효율성을 높입니다. 또한, 대규모 증류 데이터셋을 활용하여 더 적은 계산 자원으로도 높은 성능을 달성합니다. 이 연구는 생체분자 구조 예측에서 모델 확장의 새로운 방향을 제시하며, 더 나은 성능과 효율성을 목표로 합니다.  특히, 단백질-단백질 상호작용 (PPI) 예측은 신약 개발에 매우 중요하며, SeedFold는 이 분야에서 뛰어난 성능을 보여줍니다.

## 관련 연구

1. **AlphaFold**: DeepMind에서 개발한 AlphaFold는 단백질 구조 예측의 정확성을 크게 향상시켰습니다. 그러나, 높은 계산 비용이 단점으로 작용합니다. AlphaFold2의 경우, MSA (Multiple Sequence Alignment) 생성에 많은 시간이 소요됩니다.

2. **RoseTTAFold**: 이 모델은 AlphaFold와 유사한 접근 방식을 사용하지만, 더 적은 계산 자원을 요구합니다. 그러나 성능 면에서 AlphaFold에 미치지 못합니다. RoseTTAFold는 end-to-end 학습 방식을 사용하여, AlphaFold보다 학습 속도가 빠릅니다.

3. **Liteformer**: 경량화된 트랜스포머 구조로, 계산 자원이 제한된 환경에서의 활용 가능성을 높입니다. 하지만, 특정 작업에 최적화되어 있어 일반화에 한계가 있습니다.  예를 들어, 모바일 환경에서 단백질 구조 예측을 수행하는 데 적합합니다.

4. **Vision Transformer**: 이미지 인식에 적합한 트랜스포머 구조로, 다양한 시각적 작업에 적용 가능합니다. 단백질 구조 예측에는 직접적인 적용이 어렵습니다. 하지만, 단백질 구조를 이미지로 표현하여 Vision Transformer를 적용하는 연구도 진행되고 있습니다.

5. **Lightning Attention**: 주의 메커니즘의 효율성을 높이기 위해 제안된 방법으로, 계산 복잡도를 줄이는 데 중점을 둡니다. 그러나 학습 안정성 문제를 완전히 해결하지 못합니다.  Sparse Attention과 같은 기법을 사용하여 계산량을 줄입니다.

| 연구 | 차별점 | 장점 | 단점 |
| --- | --- | --- | --- |
| AlphaFold | 높은 정확성, MSA 기반 | 최고 수준의 정확도 | 높은 계산 비용, 긴 예측 시간 |
| RoseTTAFold | 적은 계산 자원, end-to-end 학습 | 빠른 학습 속도, 비교적 낮은 계산 비용 | AlphaFold 대비 낮은 정확도 |
| Liteformer | 경량화, 모바일 환경 최적화 | 매우 낮은 계산 비용, 모바일 기기에서 실행 가능 | 낮은 정확도, 특정 작업에 최적화 |
| Vision Transformer | 이미지 인식 최적화 | 이미지 기반 단백질 구조 분석 가능 | 단백질 구조 예측에 직접적인 적용 어려움 |
| Lightning Attention | 계산 효율성 | 계산량 감소, 빠른 추론 속도 | 학습 안정성 문제, 정확도 감소 가능성 |

## 핵심 기여

1. **폭 확장 전략 제안**: Pairformer의 폭을 확장하여 모델의 표현력을 높임.  이는 모델이 더 많은 정보를 저장하고 처리할 수 있게 합니다.
2. **선형 삼각형 주의 메커니즘 도입**: 계산 복잡도를 줄여 더 긴 서열을 처리 가능하게 함.  기존의 $O(n^2)$ 복잡도를 가지는 attention 메커니즘을 $O(n)$으로 줄입니다.
3. **대규모 증류 데이터셋 구축**: 더 작은 모델이 강력한 모델의 지식을 학습하도록 하여 계산 자원을 절약함.  Knowledge Distillation을 통해, 큰 모델의 지식을 작은 모델로 전달합니다.

각 기여는 생체분자 구조 예측의 정확성과 효율성을 동시에 높이는 데 중점을 두고 있습니다.

## 제안 방법론

SeedFold는 생체분자 구조 예측의 정확성을 높이기 위해 설계된 모델로, 다음과 같은 주요 아이디어와 이론적 근거를 바탕으로 합니다.

### 모델 아키텍처

SeedFold는 Pairformer 아키텍처를 기반으로 하며, 모델의 폭을 확장하여 표현력을 높입니다. Pairformer는 단백질 구조 예측에 사용되는 Transformer의 한 종류로, 단백질의 복잡한 패턴을 학습하는 데 효과적입니다. SeedFold는 hidden dimension을 늘려 더 많은 정보를 학습할 수 있도록 설계되었습니다.  Pairwise representation을 효과적으로 학습하기 위해, attention 메커니즘을 사용합니다.

### 핵심 수식

1. **폭 확장 수식**: 모델의 hidden dimension을 늘려 표현력을 높이는 수식입니다.

   $$
   H' = H \times w
   $$

   여기서 $H'$는 확장된 hidden dimension, $H$는 기존의 hidden dimension, $w$는 확장 비율입니다.  예를 들어, $H = 1024$이고 $w = 2$이면, $H' = 2048$이 됩니다.

2. **선형 삼각형 주의 메커니즘 수식**: 기존의 삼각형 주의 메커니즘을 개선한 수식으로, 계산 복잡도를 줄입니다.

   $$
   \text{Attention}(Q, K, V, B) = \text{softmax}(Q K^T + B) V
   $$

   여기서 $Q$는 query, $K$는 key, $V$는 value 행렬을 나타내며, $B$는 편향 행렬입니다.  기존의 attention 메커니즘은 $Q K^T$를 계산하는 데 $O(n^2)$의 복잡도를 가지지만, 선형 attention은 이를 $O(n)$으로 줄입니다.

3. **GatedLinearTriangularAttention 수식**: GatedLinearTriangularAttention 모듈의 수식으로, Q, K, V 행렬과 B 편향 행렬을 사용하여 주의 메커니즘을 구현합니다.

   $$
   \text{GatedAttention}(Q, K, V, B) = \sigma(Q K^T) \odot \text{softmax}(B) V
   $$

   여기서 $\sigma$는 시그모이드 함수, $\odot$는 요소별 곱셈을 나타냅니다.  시그모이드 함수는 attention score를 0과 1 사이의 값으로 제한하여, 학습 안정성을 높입니다.

이러한 수식들은 SeedFold의 계산 효율성과 표현력을 동시에 높이는 데 기여합니다.  특히, GatedLinearTriangularAttention은 long-range dependency를 효과적으로 학습하는 데 도움을 줍니다.

## 실험 설정

SeedFold의 성능을 평가하기 위해 FoldBench라는 표준 벤치마크를 사용하였습니다. FoldBench는 다양한 단백질 구조 예측 작업을 포함하며, 모델의 일반화 성능을 측정하는 데 적합합니다.  FoldBench는 CASP, CAMEO 데이터셋을 포함하고 있습니다.

### 데이터셋

- **AFDB (AlphaFold Database)**: AlphaFold가 예측한 단백질 구조 데이터베이스.  SeedFold는 AFDB를 사용하여 knowledge distillation을 수행합니다.
- **MGnify**: 환경 메타게놈 데이터베이스로, 다양한 단백질 서열을 포함합니다.  MGnify는 새로운 단백질 서열을 학습하는 데 사용됩니다.

### 평가 지표

- **RMSE (Root Mean Square Error)**: 예측된 구조와 실제 구조 간의 차이를 측정.  낮을수록 좋은 성능을 나타냅니다.
- **MAE (Mean Absolute Error)**: 예측된 구조와 실제 구조 간의 평균 절대 차이.  낮을수록 좋은 성능을 나타냅니다.
- **GDT_TS (Global Distance Test - Total Score)**: 0부터 100까지의 값으로, 예측된 구조와 실제 구조의 유사성을 측정합니다. 높을수록 좋은 성능을 나타냅니다.

### 베이스라인

- **AlphaFold3**: 최신 AlphaFold 모델로, 높은 정확성을 자랑합니다.  AlphaFold3는 현재 비공개 모델입니다.
- **RoseTTAFold**: 적은 계산 자원을 요구하는 모델로, SeedFold와 비교하여 성능을 평가합니다.

### 하이퍼파라미터

| 하이퍼파라미터 | 값 | 설명 |
| --- | --- | --- |
| 학습률 | 0.001 | Adam optimizer 사용 |
| 배치 크기 | 32 | GPU 메모리에 따라 조정 가능 |
| 에폭 수 | 100 | Early stopping 적용 |
| 드롭아웃 비율 | 0.1 | Overfitting 방지 |
| Hidden dimension | 2048 | 모델의 표현력 조절 |
| Attention heads | 32 | Attention 메커니즘의 병렬 처리 |

## 실험 결과 분석

SeedFold는 다양한 단백질 구조 예측 작업에서 AlphaFold3 및 다른 오픈 소스 모델보다 우수한 성능을 보였습니다.

### 주요 결과

| 모델 | RMSE | MAE | GDT_TS |
| --- | --- | --- | --- |
| SeedFold | 0.85 | 0.45 | 85.0 |
| AlphaFold3 | 0.90 | 0.50 | 82.0 |
| RoseTTAFold | 1.10 | 0.60 | 75.0 |

SeedFold는 AlphaFold3 대비 RMSE에서 5.6%, MAE에서 10%, GDT_TS에서 3.7%의 성능 향상을 보였습니다.  특히, 단백질-단백질 복합체 예측에서 더 큰 성능 향상을 보였습니다.

### Ablation Study

폭 확장 전략과 선형 삼각형 주의 메커니즘의 효과를 검증하기 위해 Ablation Study를 수행하였습니다. 폭 확장 전략을 제거한 경우 RMSE는 0.95로 증가하였고, 선형 삼각형 주의 메커니즘을 제거한 경우 RMSE는 0.90으로 증가하였습니다. 이는 두 가지 기법 모두 SeedFold의 성능 향상에 기여하고 있음을 보여줍니다.  Ablation study 결과는 각 모듈이 모델 성능에 미치는 영향을 명확하게 보여줍니다.

## 비판적 평가

### 강점

1. **효율적인 계산**: 선형 삼각형 주의 메커니즘을 통해 계산 복잡도를 획기적으로 줄였습니다.  이를 통해, 더 긴 단백질 서열을 예측할 수 있습니다.
2. **높은 정확성**: AlphaFold3를 능가하는 정확성을 보여주었습니다.  특히, 단백질-단백질 복합체 예측에서 뛰어난 성능을 보입니다.
3. **확장성**: 대규모 증류 데이터셋을 통해 모델의 확장성을 높였습니다.  Knowledge distillation은 모델의 일반화 성능을 향상시킵니다.

### 한계점과 개선 방향

1. **데이터셋의 다양성 부족**: 특정 데이터셋에 최적화되어 일반화에 한계가 있을 수 있습니다.  더 다양한 데이터셋을 사용하여 모델을 학습해야 합니다.
2. **복잡한 아키텍처**: 모델의 복잡성이 증가하여 구현이 어려울 수 있습니다.  모델을 단순화하는 연구가 필요합니다.
3. **긴 서열 예측**: 매우 긴 서열의 단백질 구조 예측에는 여전히 어려움이 있습니다.  더 효율적인 attention 메커니즘이 필요합니다.

### 재현성 평가

재현성을 높이기 위해 상세한 실험 설정과 하이퍼파라미터 정보를 제공하였으며, 코드 저장소를 공개하여 연구의 투명성을 높였습니다.  코드 저장소에는 학습 및 평가에 필요한 모든 코드가 포함되어 있습니다.

## 향후 연구 방향

SeedFold의 성능을 더욱 향상시키기 위해 전문가 혼합(Mixture of Experts) 기법과 사후 훈련 확장(Post-training Scaling) 기술을 탐구할 예정입니다. 전문가 혼합은 여러 개의 작은 모델을 결합하여 더 강력한 모델을 만드는 방법이며, 사후 훈련 확장은 훈련된 모델의 크기를 늘려 성능을 향상시키는 방법입니다. 이러한 기술들을 SeedFold에 적용하면 더욱 뛰어난 성능을 기대할 수 있습니다.  또한, self-supervised learning을 통해 모델의 성능을 향상시키는 연구도 진행할 예정입니다.

## 실무 적용 가이드

SeedFold를 실무에 적용할 때는 다음과 같은 고려사항이 필요합니다.

1. **계산 자원**: SeedFold는 효율적인 계산을 목표로 하지만, 여전히 상당한 계산 자원을 필요로 할 수 있습니다. 적절한 하드웨어 환경을 준비하는 것이 중요합니다.  최소 16GB 이상의 GPU 메모리를 가진 GPU가 필요합니다.
2. **데이터셋 준비**: 대규모 증류 데이터셋을 활용하여 모델의 성능을 극대화할 수 있습니다. 다양한 서열 데이터를 포함하는 데이터셋을 준비하는 것이 좋습니다.  Uniprot, PDB 데이터베이스를 활용할 수 있습니다.
3. **모델 튜닝**: 하이퍼파라미터 튜닝을 통해 모델의 성능을 최적화할 수 있습니다. 특히, 학습률과 드롭아웃 비율은 모델의 성능에 큰 영향을 미칠 수 있습니다.  Bayesian optimization을 사용하여 하이퍼파라미터를 튜닝할 수 있습니다.

## 결론

SeedFold는 생체분자 구조 예측의 정확성을 높이기 위한 효과적인 모델 확장 전략을 제시합니다. 폭 확장 전략과 선형 삼각형 주의 메커니즘을 통해 계산 효율성과 정확성을 동시에 달성하였으며, 대규모 증류 데이터셋을 통해 모델의 확장성을 높였습니다. 이 연구는 생체분자 구조 예측의 새로운 가능성을 열어주며, 향후 전문가 혼합 및 사후 훈련 확장 기술을 통해 더욱 발전할 것입니다.  SeedFold는 신약 개발, 질병 치료 등 다양한 분야에 기여할 수 있을 것으로 기대됩니다.

## 참고 자료

- [논문 링크](https://arxiv.org/abs/2512.24354)
- [코드 저장소](https://github.com/seedfold/seedfold)
- [관련 자료](https://www.deepmind.com/alphafold)
- [FoldBench](https://github.com/FoldBench/FoldBench)