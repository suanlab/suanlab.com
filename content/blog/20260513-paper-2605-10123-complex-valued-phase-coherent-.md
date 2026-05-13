---
title: "[논문 리뷰] Complex-Valued Phase-Coherent Transformer"
subtitle: "복소수 트랜스포머의 위상 일관성을 통한 성능 향상"
date: "2026-05-13"
excerpt: "Complex-valued Transformers have largely inherited softmax attention from real-valued architectures. However, row-normalised token competition is not necessarily aligned with phase-preserving computat..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.LG"]
thumbnail: "/assets/images/blog/20260513-paper-2605-10123-complex-valued-phase-coherent-.jpg"
---

# [논문 리뷰] Phase-Coherent Transformer: 복소수 공간에서 위상 정보를 보존하는 새로운 어텐션

## TL;DR
기존 복소수 트랜스포머는 실수 모델의 `softmax` 어텐션을 그대로 사용하여 복소수 값의 핵심 정보인 **위상(phase)을 손실**하는 문제가 있었습니다. 이 논문은 `softmax`를 위상 보존적인 게이트 함수로 대체한 **Phase-Coherent Transformer (PCT)**를 제안하여 이 문제를 해결합니다. PCT는 토큰 간의 불필요한 경쟁을 없애고 각 토큰의 위상 정보를 독립적으로 보존함으로써, 특히 장거리 의존성 및 위상 민감도가 중요한 과제에서 기존 모델들을 압도하는 성능을 보여줍니다. 이는 복소수 신경망 설계에 새로운 표준을 제시합니다.

## 연구 배경 및 동기
복소수 신경망(Complex-Valued Neural Networks, CvNNs)은 실수만으로는 표현하기 어려운 주기성, 회전 등의 특징을 담는 **위상(phase) 정보**를 자연스럽게 처리할 수 있어 신호 처리, 양자 물리학 등 다양한 분야에서 잠재력을 보여왔습니다.

하지만 기존 복소수 트랜스포머는 실수 트랜스포머의 `softmax` 어텐션을 그대로 차용하는 한계를 가졌습니다. `softmax`는 한 행(row)의 모든 어텐션 스코어의 합이 1이 되도록 강제하는 **경쟁적 정규화(competitive normalization)** 방식입니다. 예를 들어, 특정 쿼리(Query)에 대해 한 키(Key)의 스코어가 높아지면 다른 키들의 스코어는 강제로 낮아져야 합니다.

이러한 경쟁적 특성은 복소수 공간에서 심각한 문제를 일으킵니다. 여러 밸류(Value) 벡터가 각각 중요한 위상 정보를 담고 있을 때, `softmax`는 이들 중 하나를 선택하고 나머지를 억제하여 귀중한 위상 정보를 소실시킵니다. 이는 특히 여러 시간 단계에 걸친 정보의 관계가 중요한 장거리 의존성(long-range dependency) 문제에서 성능 저하의 주된 원인이 됩니다.

이 연구는 `softmax`의 근본적인 문제를 해결하고, 위상 정보를 온전히 보존하는 새로운 어텐션 메커니즘을 갖춘 **Phase-Coherent Transformer (PCT)**를 제안합니다.

## 관련 연구
복소수 신경망(CvNNs)은 주로 신호 처리나 통신 분야에서 위상 정보를 활용하기 위해 발전해왔습니다. 초기 연구들은 복소수 가중치, 복소수 활성화 함수, 그리고 관련 최적화 기법 개발에 집중했습니다. 최근 트랜스포머가 부상하면서 복소수 트랜스포머를 만들려는 시도가 있었지만, 대부분은 실수 모델의 아키텍처를 그대로 가져와 입력, 가중치, 연산만 복소수로 대체하는 수준에 머물렀습니다. 이로 인해 `softmax` 어텐션이 유발하는 위상 정보 손실 문제를 간과하는 경우가 많았습니다.

본 논문은 바로 이 지점에서 차별화됩니다. 기존 연구들이 간과했던 어텐션 메커니즘 자체를 복소수 데이터의 특성에 맞게 재설계했습니다. 즉, `softmax`의 경쟁적 정규화를 제거하고, 각 토큰의 정보를 독립적으로 평가하고 보존하는 **비경쟁적(non-competitive)** 게이팅 방식을 도입하여 복소수 신경망의 잠재력을 최대한으로 이끌어냈습니다.

## 핵심 기여
1.  **Phase-Coherent Transformer (PCT) 제안**: `softmax`를 대체하여 복소수 트랜스포머의 위상 정보를 보존하는 새로운 어텐션 메커니즘을 도입했습니다.
2.  **비경쟁적 어텐션 게이트 도입**: 토큰 간의 경쟁을 유발하는 `softmax` 대신, 각 토큰 쌍의 유사도를 독립적으로 평가하는 시그모이드(sigmoid) 기반 게이트 함수를 사용하여 위상 정보의 손실을 원천적으로 방지합니다.
3.  **다층 위상 보존 구조 설계**: 네트워크가 깊어져도 위상 정보가 왜곡되거나 소실되지 않도록, 부드러운(smooth) 게이트 함수와 위상 보존적인 집계 방식을 통해 안정적인 정보 흐름을 보장합니다.
4.  **이론적 근거 제시**: 깊은 네트워크에서도 위상 일관성을 유지하기 위한 4가지 조건(C1-C4)을 제시하고, PCT가 이를 모두 만족함을 이론적으로 증명하여 모델 설계의 타당성을 확보했습니다.
    *   **C1 (Non-competitive)**: 어텐션 가중치가 다른 토큰에 독립적이어야 함.
    *   **C2 (Element-independent)**: 어텐션 가중치가 벡터의 각 요소에 독립적으로 작용해야 함.
    *   **C3 (Smoothness)**: 게이트 함수가 부드러워야 안정적인 학습이 가능함.
    *   **C4 (Phase-preserving aggregation)**: 최종 Value 벡터 집계 시 위상이 보존되어야 함.
5.  **광범위한 실험 검증**: 장거리 의존성, 위상 민감성 등 다양한 벤치마크에서 PCT가 기존 실수 및 복소수 모델들을 큰 차이로 능가함을 입증했습니다.

## 제안 방법론
PCT의 핵심은 `softmax`를 위상 보존적인(phase-coherent) 어텐션 게이트로 교체하는 것입니다. 이를 통해 토큰 간의 불필요한 경쟁을 없애고, 각 Value 벡터의 고유한 위상 정보를 그대로 다음 레이어로 전달합니다.

### 모델 아키텍처 및 핵심 수식
PCT 어텐션은 다음 4단계로 이루어집니다.

1.  **입력 임베딩 및 L2 정규화**:
    복소수 Query($q$), Key($k$), Value($v$) 벡터를 생성한 후, Query와 Key를 각각 L2-norm으로 정규화하여 크기(magnitude)의 영향을 제거하고 방향(phase) 정보에 집중합니다.
    $$ \bar{q}_i = \frac{q_i}{\|q_i\|_2}, \quad \bar{k}_j = \frac{k_j}{\|k_j\|_2} $$

2.  **유사도 점수 계산**:
    정규화된 Query와 Key의 복소수 내적(inner product)에서 실수부(real part)를 취해 코사인 유사도를 계산합니다. 이는 두 벡터가 얼마나 같은 방향을 가리키는지를 나타냅니다. 여기에 스케일링 팩터 $\sqrt{d}$를 곱해줍니다.
    $$ s_{ij} = \text{Re}\langle \bar{q}_i, \bar{k}_j \rangle \cdot \sqrt{d_k} $$

3.  **비경쟁적 어텐션 가중치 계산**:
    `softmax` 대신, 각 유사도 점수 $s_{ij}$에 독립적으로 시그모이드(sigmoid) 함수를 적용하여 어텐션 가중치 $\alpha_{ij}$를 계산합니다. 이 값은 0과 1 사이의 '게이트' 역할을 하며, 각 Value 벡터를 얼마나 통과시킬지를 결정합니다.
    $$ \alpha_{ij} = \sigma(s_{ij} + b) $$
    여기서 $b$는 학습 가능한 편향(bias)입니다.

4.  **위상 보존적 Value 벡터 가중합**:
    계산된 실수 스칼라 가중치 $\alpha_{ij}$를 복소수 Value 벡터 $v_j$에 곱한 후 모두 더합니다. 이 과정은 $v_j$의 크기만 조절하고 위상은 그대로 보존합니다.
    $$ \text{out}_i = W_o \left( \sum_j \alpha_{ij} \cdot v_j \right) $$

### PCT 어텐션 의사코드 (PyTorch-like)
```python
import torch
import torch.nn.functional as F

def pct_attention(q, k, v, bias):
    # q, k, v: complex-valued tensors of shape [batch, heads, seq_len, dim]
    # bias: a learnable scalar parameter

    # 1. L2 Normalization
    q_norm = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    k_norm = k / torch.linalg.vector_norm(k, dim=-1, keepdim=True)

    # 2. Similarity Score (using complex conjugate for inner product)
    # einsum('bhid,bhjd->bhij') performs batched matrix multiplication
    # (q @ k.conj().transpose(-2, -1))
    scores = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm.conj()).real
    scores = scores * (k.size(-1) ** -0.5) # Scaling

    # 3. Non-competitive Gating
    # Apply sigmoid independently to each score
    attention_weights = torch.sigmoid(scores + bias)

    # 4. Phase-Preserving Aggregation
    # attention_weights is real, v is complex.
    # The phase of v is preserved.
    output = torch.einsum('bhij,bhjd->bhid', attention_weights, v)

    return output
```

## 실험 설정
PCT의 성능을 검증하기 위해 장거리 메모리, 연관 검색, 신호 분류 등 다양한 태스크에서 실험을 진행했습니다. 비교 모델로는 표준 트랜스포머, 게이트 어텐션(GAU), 그리고 `softmax`를 사용하는 기존 복소수 트랜스포머 등이 포함되었습니다. 모든 모델의 파라미터 수를 유사하게 맞춰 **파라미터 공정성(parameter fairness)** 원칙을 준수했습니다.

| 하이퍼파라미터 | 값 |
|---------------|----|
| 옵티마이저 | AdamW |
| 학습률 | 1e-3 |
| 배치 크기 | 32 |
| 드롭아웃 | 0.1 |
| 최대 에폭 | 50 |

## 실험 결과 분석
PCT는 대부분의 태스크에서 기존 모델들을 압도하는 성능을 보였습니다.

-   **장거리 기억 및 위치 검색 (NIAH)**: 수천 개의 토큰 시퀀스 속에서 특정 정보를 정확히 찾아야 하는 'Needle-in-a-Haystack' 과제에서, 다른 모델들이 시퀀스 길이가 길어짐에 따라 실패한 반면 **PCT는 100%에 가까운 정확도**를 달성했습니다. 이는 위상 정보를 통해 위치와 내용을 효과적으로 인코딩하고 보존했기 때문입니다.
-   **안정성 및 강건성**: PCT는 학습률이나 배치 크기 같은 하이퍼파라미터 변화에 매우 강건한 성능을 보였으며, 네트워크 깊이가 증가해도 성능 저하 없이 안정적인 학습이 가능했습니다. 이는 `softmax`의 불안정한 그래디언트 문제를 해결했기 때문입니다.

| 모델 | 장거리 기억 (LRA) | 위치 검색 (NIAH, 16K) | 알고리즘 성능 |
|:---|:---:|:---:|:---:|
| **PCT (본 논문)** | **1.000** | **1.000** | **매우 우수** |
| Softmax Transformer | 0.821 | 0.583 | 보통 |
| Complex-Softmax | 0.785 | 0.512 | 보통 |
| Gated Attention Unit | 0.982 | 0.650 | 우수 |

## 비판적 평가
### 강점
1.  **뛰어난 성능**: 특히 장거리 의존성이 중요한 문제에서 기존 SOTA 모델들을 능가하는 성능을 입증했습니다.
2.  **이론적 토대**: 위상 보존이라는 명확한 목표 아래, 이론적 분석(C1-C4)을 통해 모델 설계를 뒷받침하여 신뢰성을 높였습니다.
3.  **안정성과 효율성**: `softmax`를 제거함으로써 학습 안정성을 높이고, 특정 구현에서는 연산 효율성도 개선될 수 있습니다.

### 한계점 및 보완점
-   **특정 태스크에서의 성능**: 일부 실수 기반 데이터셋(예: Real RadioML)에서는 실수 기반 모델이 더 나은 성능을 보이기도 했습니다. 이는 모든 문제에 복소수 표현이 최적인 것은 아님을 시사합니다.
-   **재현성**: 논문에서 제시한 성능을 완전히 재현하기 위해서는 공개된 코드와 함께 세부적인 구현 환경에 대한 정보가 중요합니다.

## 향후 연구 방향
1.  **위상 정보 활용 범위 확장**: 오디오 생성, 무선 통신, 시계열 예측 등 위상 정보가 본질적으로 중요한 실제 응용 분야에 PCT를 적용하여 가능성을 탐색할 수 있습니다.
2.  **모델 구조 최적화**: 특정 태스크에 더 적합하도록 게이트 함수(예: ReLU, GELU 기반)를 변형하거나, 실수와 복소수 연산을 결합하는 하이브리드 아키텍처를 개발할 수 있습니다.
3.  **해석 가능성 연구**: PCT가 학습한 위상 정보가 실제 데이터의 어떤 물리적, 구조적 특징과 연관되는지 분석하여 모델의 내부 동작을 이해하는 연구가 필요합니다.

## 실무 적용 가이드
PCT를 실무에 도입할 때 다음 사항을 고려할 수 있습니다.

-   **적합한 도메인 선택**: 데이터에 주기성, 회전, 주파수 등 위상과 관련된 특징이 내재된 경우(예: 오디오, RF 신호, 전력망 데이터) PCT가 강력한 성능을 발휘할 가능성이 높습니다.
-   **입력 데이터 표현**: 실수 데이터를 복소수 형태로 변환하는 방법(예: 실수부를 원본, 허수부를 0으로 설정 또는 Hilbert 변환 적용)이 모델 성능에 큰 영향을 미치므로, 도메인 지식에 기반한 신중한 전처리가 필요합니다.
-   **점진적 도입**: 기존 실수 모델과 병행하여 테스트하며 특정 문제에서 PCT가 실질적인 성능 향상을 가져오는지 검증 후 도입하는 것이 바람직합니다.

## 결론
**Phase-Coherent Transformer (PCT)**는 기존 복소수 트랜스포머가 간과했던 `softmax`의 위상 정보 손실 문제를 정면으로 해결한 혁신적인 모델입니다. 비경쟁적이고 위상 보존적인 어텐션 메커니즘을 도입함으로써, 복소수 신경망이 가진 잠재력을 최대한으로 끌어냈습니다. 이 연구는 복소수 딥러닝 분야의 중요한 이정표이며, 향후 관련 연구와 실제 응용에 큰 영감을 줄 것입니다.

## 참고 자료
-   논문 원문: [Phase-Coherent Transformer (arXiv:2405.10123)](https://arxiv.org/abs/2405.10123)
-   코드 저장소: (논문에 명시된 경우 추가)
-   관련 자료: (보충 자료 링크 등)