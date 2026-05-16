---
title: "[논문 리뷰] Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space"
subtitle: "트랜스포머의 효율성을 높이는 압축 컨볼루션 어텐션"
date: "2026-05-16"
excerpt: "Multi-headed Attention's (MHA) quadratic compute and linearly growing KV-cache make long-context transformers expensive to train and serve. Prior works such as Grouped Query Attention (GQA) and Multi-..."
category: "Paper Review"
tags: ["Paper Review","cs.CL","cs.AI","cs.CL"]
thumbnail: "/assets/images/blog/20260516-paper-2510-04476-compressed-convolutional-atten.jpg"
---

# [논문 리뷰] Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space

## TL;DR

대규모 언어 모델(LLM)의 핵심인 Multi-Head Attention(MHA)은 시퀀스 길이가 길어질수록 계산량과 메모리 사용량이 크게 증가하는 병목 현상을 가집니다. 이를 해결하기 위해 **Compressed Convolutional Attention(CCA)**와 **Compressed Convolutional Grouped Query Attention(CCGQA)**가 제안되었습니다. 이 방법들은 쿼리, 키, 값을 저차원 잠재 공간으로 '압축'한 뒤 어텐션을 수행하여 계산 효율과 메모리 사용량을 획기적으로 개선합니다. 실험 결과, CCGQA는 동일한 KV 캐시 압축률에서 GQA, MLA보다 일관되게 우수한 성능을 보였으며, 특히 MoE 모델에서 기존 방식의 절반 수준의 KV 캐시를 사용하면서도 8배의 압축률을 달성했습니다. 이는 긴 컨텍스트를 처리하는 LLM의 서빙 비용을 크게 절감할 수 있는 강력한 대안임을 시사합니다.

## 1. 연구 배경 및 동기

대규모 언어 모델(LLM)의 성공은 트랜스포머 아키텍처, 그중에서도 Multi-Head Attention(MHA) 메커니즘에 크게 의존합니다. MHA는 입력 시퀀스 내 토큰 간의 관계를 동적으로 파악하는 데 매우 효과적이지만, 두 가지 주요한 확장성 문제를 안고 있습니다.

1.  **계산 복잡도**: 어텐션 스코어를 계산하는 과정은 시퀀스 길이 $N$에 대해 $O(N^2)$의 시간 복잡도를 가집니다. 이는 입력이 길어질수록 프리필(prefill) 단계의 연산량이 기하급수적으로 증가함을 의미합니다.
2.  **메모리 병목**: 추론(inference) 시, 이전에 계산된 키(Key)와 값(Value)을 저장하는 **KV 캐시**의 크기는 시퀀스 길이 $N$에 비례하여 선형적으로($O(N)$) 증가합니다. 긴 컨텍스트를 처리할 때 이 KV 캐시가 GPU 메모리를 대부분 차지하게 되어, 메모리 대역폭 병목을 유발하고 전체 처리 속도를 저하시킵니다.

이러한 문제를 완화하기 위해 Grouped-Query Attention(GQA)나 Multi-Query Attention(MQA) 같은 변형들이 제안되었지만, 여전히 성능과 효율성 사이의 트레이드오프가 존재합니다. 본 연구는 이 한계를 극복하고자, 압축된 잠재 공간에서 어텐션을 수행하는 새로운 접근법인 CCA와 CCGQA를 제안합니다.

## 2. 관련 연구

기존의 효율적인 어텐션 메커니즘들은 주로 KV 헤드의 수를 줄이는 데 초점을 맞추었습니다.

| 연구 | 주요 아이디어 | 장점 | 단점 |
| :--- | :--- | :--- | :--- |
| **GQA** | 여러 쿼리 헤드가 하나의 키/값 헤드 그룹을 공유 | KV 캐시 크기 감소, MQA보다 성능 저하 적음 | 여전히 성능과 효율 간의 트레이드오프 존재 |
| **MLA** | 학습 시 MHA, 추론 시 MQA처럼 작동하는 하이브리드 구조 | 추론 시 KV 캐시 최소화 | 텐서 병렬 처리 등 복잡한 환경에서 유연성 부족 |
| **CCA/CCGQA** | **쿼리, 키, 값을 저차원 잠재 공간으로 압축 후 어텐션 수행** | 파라미터, KV 캐시, 연산량(FLOPs) 동시 감소 | 기존 방식보다 구조가 다소 복잡함 |

CCA/CCGQA는 헤드 수를 줄이는 기존 방식과 달리, 헤드의 '차원'을 압축하는 새로운 접근법을 취합니다. 이를 통해 GQA와 같은 헤드 공유 기법과 결합하여 효율성을 극대화할 수 있습니다.

## 3. 핵심 기여

1.  **Compressed Convolutional Attention (CCA) 제안**: 쿼리, 키, 값을 압축된 잠재 공간으로 투영하여 계산량과 메모리 사용량을 획기적으로 줄이는 새로운 어텐션 메커니즘을 제안합니다.
2.  **Compressed Convolutional Grouped Query Attention (CCGQA) 제안**: CCA의 아이디어에 GQA의 헤드 공유 방식을 결합하여, 성능 저하를 최소화하면서 더 큰 KV 캐시 압축률을 달성합니다.
3.  **실험적 성능 검증**: 다양한 모델과 데이터셋에서 CCGQA가 GQA와 MLA보다 일관되게 더 나은 성능(낮은 손실)을 보임을 실험적으로 입증합니다.
4.  **뛰어난 하드웨어 효율성**: H100 GPU에서 최적화된 CCA/CCGQA 커널이 MHA 대비 약 **1.7배 빠른 프리필 속도**를 기록하여, 긴 시퀀스 처리 시의 실질적인 효율성 개선을 보여줍니다.

## 4. 제안 방법론

CCA와 CCGQA의 핵심 아이디어는 **'압축 후 연산'**입니다. 전체 차원에서 어텐션을 계산하는 대신, 쿼리(Q), 키(K), 값(V)을 더 작은 차원의 **압축된 잠재 공간(compressed latent space)**으로 투영(projection)한 후, 이 효율적인 공간에서 어텐션 연산을 수행합니다.

### Compressed Convolutional Attention (CCA)

CCA는 다음 세 단계로 구성됩니다.

#### 1. 압축 (Compression)
먼저, 각 어텐션 헤드의 Q, K, V 벡터를 학습 가능한 선형 레이어($W_Q, W_K, W_V$)를 통해 더 낮은 차원 $d_{comp}$로 투영합니다.

$$
Q' = Q \cdot W_Q, \quad K' = K \cdot W_K, \quad V' = V \cdot W_V
$$

-   $Q, K, V \in \mathbb{R}^{N \times d_{head}}$
-   $W_Q, W_K, W_V \in \mathbb{R}^{d_{head} \times d_{comp}}$
-   $Q', K', V' \in \mathbb{R}^{N \times d_{comp}}$ (여기서 $d_{comp} < d_{head}$)

이 단계를 통해 파라미터 수, KV 캐시 크기, 그리고 후속 연산의 FLOPs를 동시에 줄일 수 있습니다.

#### 2. 지역 정보 보강 (Convolution)
압축 과정에서 손실될 수 있는 지역적(local) 정보를 보강하고 표현력을 높이기 위해, 압축된 Q와 K에 **깊이별 컨볼루션(depth-wise convolution)**을 적용합니다.

$$
Q_{comp} = \text{Conv}(\text{RoPE}(\text{L2Norm}(Q')))
$$
$$
K_{comp} = \text{Conv}(\text{RoPE}(\text{L2Norm}(K')))
$$

-   **L2Norm**: 벡터를 정규화하여 학습 안정성을 높입니다.
-   **RoPE (Rotary Positional Embedding)**: 회전 행렬을 이용해 절대적인 위치 정보를 주입합니다.
-   **Conv**: 작은 커널(e.g., 3x1)을 가진 깊이별 컨볼루션이 시퀀스 차원을 따라 적용되어, 인접 토큰 간의 지역적 패턴을 포착합니다.

#### 3. 잠재 공간 어텐션 (Latent Space Attention)
마지막으로, 처리된 압축 Q, K와 압축 V를 사용하여 표준 스케일드 닷-프로덕트 어텐션을 수행합니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q_{comp} K_{comp}^T}{\sqrt{d_{comp}}}\right) V'
$$

모든 핵심 연산이 더 작은 차원 $d_{comp}$에서 이루어지므로 계산 효율이 크게 향상됩니다.

### Compressed Convolutional Grouped Query Attention (CCGQA)

CCGQA는 CCA의 효율성을 한 단계 더 끌어올린 버전입니다. CCA의 압축된 잠재 공간 내에서 GQA 스타일의 **K, V 헤드 공유**를 적용합니다.

-   여러 개의 압축된 쿼리 헤드 그룹이 하나의 압축된 키/값 헤드를 공유합니다.
-   예를 들어, 8개의 쿼리 헤드가 2개의 키/값 헤드를 공유하는 방식입니다.
-   이를 통해 KV 캐시를 추가로 대폭 감소시키면서도 성능 저하를 최소화하여, 계산량과 메모리 대역폭 사이의 **파레토 최적(Pareto-optimal)** 경계를 더욱 개선합니다.

## 5. 실험 설정

CCA와 CCGQA의 성능은 다양한 규모의 MoE(Mixture of Experts) 모델과 Dense 모델에서 검증되었습니다. 실험은 NVIDIA H100 GPU 클러스터에서 BF16 정밀도로 수행되었습니다.

| 모델 | 파라미터 | 데이터셋 | 평가 지표 | 주요 비교 대상 |
| :--- | :--- | :--- | :--- | :--- |
| MoE | 350M, 1.5B | Zyda2 | 학습 손실(Loss), 처리량(Throughput), 지연 시간(Latency) | GQA, MLA |
| Dense | 1B | Zyda2 | 학습 손실(Loss), 처리량(Throughput), 지연 시간(Latency) | GQA, MLA |

## 6. 실험 결과 분석

### 모델 성능 (학습 손실)

-   MoE와 Dense 모델 모두에서, 동일한 KV 캐시 압축률을 가질 때 **CCA/CCGQA가 GQA, MLA보다 일관되게 더 낮은 학습 손실(loss)을 달성**했습니다. 이는 CCA/CCGQA가 더 적은 자원으로 더 높은 모델 품질을 달성할 수 있음을 의미합니다.
-   특히 CCGQA는 8배의 KV 캐시 압축률에서도 성능 저하가 거의 없어, 극적인 효율성 개선이 가능함을 보여주었습니다.

### 계산 효율성 (지연 시간)

-   프리필(prefill) 단계의 순방향/역방향 전파에서 **CCA는 MHA, GQA, MLA보다 훨씬 낮은 지연 시간(더 빠른 속도)을 기록**했습니다.
-   시퀀스 길이가 4K에서 16K로 길어질수록 속도 향상 폭이 더욱 커졌으며, 이는 CCA가 긴 컨텍스트 처리에 특히 효과적임을 입증합니다.

### MLA에 대한 비판적 분석

-   논문은 MLA가 단일 GPU 디코딩에서는 효율적일 수 있으나, 텐서 병렬화(Tensor Parallelism)나 추론적 디코딩(speculative decoding)과 같은 최신 추론 최적화 기법과 함께 사용될 때 비효율적이고 유연성이 떨어진다고 지적합니다.
-   반면, CCA/CCGQA는 다양한 병렬화 전략에 구애받지 않는 더 범용적이고 유연한 구조임을 강조합니다.

## 7. 비판적 평가

### 강점

1.  **획기적인 효율성**: 계산량, 메모리 사용량, 파라미터 수를 동시에 줄여 LLM의 학습 및 추론 비용을 크게 절감합니다.
2.  **우수한 모델 성능**: 효율성을 높이면서도 기존 방식들보다 더 나은 모델 품질(낮은 손실)을 달성하여 성능-효율 트레이드오프를 개선했습니다.
3.  **구조적 유연성**: 다양한 병렬화 전략 및 추론 최적화 기법과 쉽게 통합될 수 있는 범용적인 구조를 가집니다.

### 한계점 및 개선 방향

-   **구현 복잡성**: CCA/CCGQA는 표준 어텐션에 비해 컨볼루션과 추가적인 선형 레이어가 포함되어 있어 구현이 더 복잡합니다. 최고 성능을 위해서는 커스텀 CUDA 커널 구현이 필요할 수 있습니다.
-   **하이퍼파라미터 민감도**: 압축 차원($d_{comp}$), 컨볼루션 커널 크기 등 새로운 하이퍼파라미터가 추가되어, 최적의 조합을 찾기 위한 추가적인 튜닝이 필요할 수 있습니다.

### 재현성 평가

-   논문에서 제안된 알고리즘과 수식이 명확하게 설명되어 있으며, 핵심 아이디어가 직관적이어서 재현성이 높은 것으로 평가됩니다.

## 8. 향후 연구 방향

-   **다양한 모델 아키텍처 적용**: CCA/CCGQA의 개념을 비전 트랜스포머(ViT)나 멀티모달 모델 등 다른 유형의 모델에 적용하여 그 범용성을 검증할 수 있습니다.
-   **압축 기법 고도화**: 선형 투영 외에 더 정교한 비선형 압축 기법을 탐구하여 정보 손실을 최소화하고 성능을 더욱 향상시키는 연구가 가능합니다.
-   **자동화된 하이퍼파라미터 튜닝**: 모델 아키텍처와 태스크에 맞춰 최적의 압축률과 커널 크기를 자동으로 결정하는 연구가 필요합니다.

## 9. 실무 적용 가이드

-   **긴 컨텍스트 모델 서빙**: 수십만 토큰 이상의 긴 컨텍스트를 처리해야 하는 LLM 서비스에서 KV 캐시 병목을 해결하기 위한 효과적인 솔루션이 될 수 있습니다.
-   **온디바이스 AI**: 제한된 메모리와 연산 자원을 가진 엣지 디바이스에 LLM을 배포할 때, CCA/CCGQA를 통해 모델의 크기와 연산량을 크게 줄일 수 있습니다.
-   **구현 팁**: 구현 시 압축된 잠재 공간의 크기($d_{comp}$)와 컨볼루션 커널의 크기를 적절히 조정하여 원하는 성능과 효율성 사이의 균형점을 찾아야 합니다. 일반적으로 압축률이 높을수록 속도는 빨라지지만 성능이 저하될 수 있습니다.

## 10. 결론

Compressed Convolutional Attention (CCA)와 CCGQA는 기존 어텐션 메커니즘의 계산 및 메모리 병목 현상을 해결하는 강력하고 실용적인 대안을 제시합니다. 단순히 헤드 수를 줄이는 것을 넘어, 연산 자체를 더 효율적인 저차원 공간에서 수행한다는 아이디어는 LLM의 확장성을 한 단계 끌어올립니다. 이러한 혁신은 대규모 언어 모델을 더 적은 자원으로 더 빠르게 구동할 수 있게 하여, AI 기술의 접근성을 높이고 새로운 응용 분야를 개척하는 데 크게 기여할 것입니다.

## 참고 자료

-   [논문 원문 (arXiv:2405.04476)](https://arxiv.org/abs/2405.04476)
-   [코드 저장소 (공개 예정)](https://github.com/example/cca-ccgqa)