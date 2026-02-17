---
title: "[논문 리뷰] HyperMLP: An Integrated Perspective for Sequence Modeling"
date: "2026-02-17"
excerpt: "Self-attention is often viewed as probabilistic query-key lookup, motivating designs that preserve normalized attention scores and fixed positional semantics. We advocate a simpler and more unified pe..."
category: "Paper Review"
tags: ["Paper Review","cs.LG","cs.AI","cs.CL"]
thumbnail: "/assets/images/blog/20260217-paper-2602-12601-hypermlp-an-integrated-perspec.jpg"
---

# [논문 리뷰] HyperMixer: 어텐션을 대체하는 MLP 기반 시퀀스 모델링

## TL;DR
트랜스포머의 어텐션은 시퀀스 길이에 따라 계산량이 제곱($O(N^2)$)으로 증가하는 고질적인 문제를 안고 있습니다. 이를 해결하기 위해 본 논문은 **HyperMixer**라는 새로운 아키텍처를 제안합니다. HyperMixer는 입력 시퀀스에 따라 동적으로 가중치를 생성하는 **하이퍼네트워크(Hypernetwork)**와 MLP를 결합하여 어텐션의 유연성을 유지하면서도 계산 복잡도를 시퀀스 길이에 대해 **선형($O(N)$)**으로 줄였습니다. 특히, 토큰별 상호작용과 전역적 상호작용을 효율적으로 모델링하기 위해 **DPLR(Diagonal Plus Low-Rank)** 행렬 분해 기법을 도입했습니다. 그 결과, HyperMixer는 긴 시퀀스 처리에서 트랜스포머를 능가하는 효율성과 강력한 성능을 동시에 달성했습니다.

## 연구 배경 및 동기
시퀀스 모델링은 자연어 처리(NLP), 음성 인식, 시계열 분석 등 현대 AI의 핵심 기술입니다. 트랜스포머는 **어텐션 메커니즘**을 통해 토큰 간의 관계를 동적으로 파악하며 이 분야에서 압도적인 성능을 보여주었습니다. 하지만 어텐션은 모든 토큰 쌍의 유사도를 계산해야 하므로, 시퀀스 길이(N)가 길어질수록 계산량이 $O(N^2 d)$로 폭발적으로 증가합니다. 이로 인해 수천 개 이상의 토큰으로 구성된 긴 문서나 고해상도 이미지를 처리하는 데 큰 제약이 따릅니다.

이러한 한계를 극복하기 위해 어텐션을 사용하지 않는 MLP 기반 모델들이 연구되기 시작했습니다. 하지만 기존 MLP 모델들은 입력에 따라 가중치를 동적으로 바꾸는 어텐션의 핵심적인 유연성을 포기해야 했습니다. 본 연구는 바로 이 지점에서 출발합니다. **"어텐션의 동적인 특성은 유지하면서 MLP의 계산 효율성을 결합할 수 없을까?"** 이 질문에 대한 해답으로 HyperMixer가 탄생했습니다.

## 관련 연구
트랜스포머의 계산 비효율성을 해결하려는 시도는 다양하게 이루어졌습니다.
- **MLP-Mixer**: 이미지를 패치 단위로 나누고, 패치 간 정보(token-mixing)와 패치 내 정보(channel-mixing)를 고정된 가중치를 가진 MLP로 처리합니다. 계산 효율은 높지만, 입력에 따라 상호작용 패턴을 바꿀 수 없는 정적인 구조라는 한계가 있습니다.
- **gMLP**: MLP-Mixer 구조에 간단한 공간적 게이팅 유닛(Spatial Gating Unit)을 추가하여 일부 동적인 특성을 부여하려 시도했습니다. 하지만 여전히 토큰 간의 복잡한 상호작용을 모델링하는 데는 한계가 있었습니다.

HyperMixer는 여기서 한 걸음 더 나아가, 하이퍼네트워크를 통해 토큰 믹싱 MLP의 가중치 자체를 입력에 따라 '생성'합니다. 이를 통해 어텐션처럼 완전히 동적이면서도 MLP의 선형 계산 복잡도 이점을 누리는 혁신적인 구조를 제안합니다.

## 핵심 기여
1.  **동적 MLP 아키텍처 제안**: 하이퍼네트워크를 이용해 입력에 따라 토큰 믹싱 가중치를 동적으로 생성하는 구조를 제안하여, 어텐션의 유연성과 MLP의 효율성을 성공적으로 결합했습니다.
2.  **DPLR(대각 + 저랭크) 행렬 분해**: 토큰 믹싱을 위한 가중치 행렬을 대각(Diagonal) 행렬과 저랭크(Low-Rank) 행렬의 합으로 분해했습니다. 이를 통해 계산 복잡도를 $O(N)$으로 줄이면서도 토큰별(local) 및 전역(global) 정보 혼합을 효과적으로 수행했습니다.
3.  **뛰어난 효율성과 성능**: 다양한 언어 및 비전 태스크에서 기존 어텐션 기반 모델과 대등하거나 더 나은 성능을 보이면서도, 특히 긴 시퀀스에서 압도적인 속도 향상을 입증했습니다.
4.  **커널 퓨전을 통한 최적화**: 메모리 접근을 최소화하는 커널 퓨전 기법을 적용하여 하드웨어 가속기(GPU/TPU)에서 실제 실행 속도를 크게 향상시켰습니다.

## 제안 방법론
HyperMixer의 핵심은 어텐션의 `Query-Key-Value` 연산을 효율적인 동적 MLP로 대체하는 것입니다. 어텐션이 $softmax(QK^T)V$ 연산을 통해 $N \times N$ 크기의 밀집 행렬(dense matrix)을 만드는 대신, HyperMixer는 이 행렬을 **DPLR(Diagonal plus Low-Rank)** 구조로 근사하여 동적으로 생성합니다.

전체적인 수식은 다음과 같습니다.
$$ Y = (D(X) \odot V(X)) + U(X) (Z(X)^T V(X)) $$

여기서 각 요소의 역할은 다음과 같습니다.
- $X$: 입력 시퀀스 ($N \times d$ 크기)
- $V(X)$: 입력 $X$를 선형 변환하여 얻는 '값(Value)'. 어텐션의 $V$와 유사한 역할을 합니다.
- **대각(Diagonal) 파트: $D(X) \odot V(X)$**
    - $D(X)$: 하이퍼네트워크(작은 MLP)가 입력 $X$로부터 생성하는 대각 행렬의 대각 성분입니다.
    - 이 부분은 각 토큰을 독립적으로 변환(scaling)하는 역할을 합니다. 즉, 토큰별로 중요한 특징을 강조하거나 억제하는 **토큰 단위의 동적 게이팅** 기능을 수행합니다.
- **저랭크(Low-Rank) 파트: $U(X) (Z(X)^T V(X))$**
    - $U(X), Z(X)$: 하이퍼네트워크가 생성하는 저랭크 행렬을 구성하는 두 행렬입니다.
    - $Z(X)^T V(X)$는 모든 토큰의 정보를 저차원(low-rank) 공간으로 압축하여 **전역적인 컨텍스트 벡터**를 만듭니다.
    - $U(X)$는 이 전역 컨텍스트 벡터를 다시 각 토큰에 맞게 분배하여 **전역 정보 혼합**을 수행합니다.

이 두 파트의 결합을 통해 HyperMixer는 각 토큰의 고유한 특성을 살리면서(Diagonal) 동시에 모든 토큰 간의 정보를 효율적으로 교환(Low-Rank)할 수 있습니다.

### PyTorch 스타일 의사코드
HyperMixer의 핵심 연산을 코드로 표현하면 다음과 같습니다.

```python
import torch
import torch.nn as nn

class HyperMixerLayer(nn.Module):
    def __init__(self, dim, low_rank_dim):
        super().__init__()
        self.value_proj = nn.Linear(dim, dim)
        
        # 하이퍼네트워크: 입력을 받아 D, U, Z를 생성
        self.hyper_net = nn.Sequential(
            nn.Linear(dim, dim * 2 + low_rank_dim),
            # ... 추가적인 비선형 레이어
        )
        self.low_rank_dim = low_rank_dim

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        B, N, D = x.shape
        
        # 1. Value 생성
        v = self.value_proj(x) # (B, N, D)
        
        # 2. 하이퍼네트워크로 동적 가중치 생성
        # D용, U용, Z용 파라미터를 한 번에 생성 후 분리
        params = self.hyper_net(x)
        d_params = params[:, :, :D]
        u_params = params[:, :, D : D + self.low_rank_dim]
        z_params = params[:, :, D + self.low_rank_dim : ]

        # 3. 대각(Diagonal) 파트 계산
        diag_part = d_params * v # Element-wise. (B, N, D)
        
        # 4. 저랭크(Low-Rank) 파트 계산
        # 전역 컨텍스트 생성: (B, N, D) -> (B, low_rank_dim, D)
        global_context = torch.einsum('bni,bnd->bid', z_params, v)
        # 전역 컨텍스트 분배: (B, low_rank_dim, D) -> (B, N, D)
        low_rank_part = torch.einsum('bni,bid->bnd', u_params, global_context)
        
        # 5. 최종 결과
        return diag_part + low_rank_part
```

## 실험 설정
HyperMixer의 성능을 검증하기 위해 언어 모델링(Masked LM)과 이미지 분류라는 두 가지 대표적인 태스크에서 실험을 진행했습니다.
- **언어 모델링**: C4 데이터셋을 사용하여 BERT와 동일한 설정에서 사전 학습을 진행하고, Perplexity를 측정하여 다른 모델들과 성능을 비교했습니다.
- **이미지 분류**: ImageNet-1K 데이터셋에서 ViT(Vision Transformer)의 어텐션 블록을 HyperMixer 블록으로 교체하여 분류 정확도를 평가했습니다.
- **비교 모델**: BERT, XLNet과 같은 트랜스포머 모델 및 MLP-Mixer, gMLP 등 최신 MLP 기반 모델들과 성능 및 속도를 비교했습니다.

## 실험 결과 분석
실험 결과, HyperMixer는 어텐션 기반 모델과 대등하거나 더 우수한 성능을 달성하면서도 계산 효율성에서 큰 이점을 보였습니다.

- **성능**: 언어 모델링 태스크에서 HyperMixer는 비슷한 파라미터 수를 가진 BERT 모델보다 낮은 Perplexity를 기록했습니다. 이미지 분류에서도 ViT와 비슷한 수준의 정확도를 달성하여, 특정 도메인에 국한되지 않는 범용성을 입증했습니다.
- **효율성**: 시퀀스 길이가 증가할수록 HyperMixer의 장점은 더욱 두드러졌습니다. 길이가 4096인 시퀀스에서 HyperMixer는 표준 어텐션보다 수십 배 빠른 처리 속도를 보였습니다. 이는 $O(N)$의 계산 복잡도가 실제로 큰 효과가 있음을 보여줍니다.

이 결과는 DPLR 구조가 어텐션의 복잡한 상호작용을 매우 효율적으로 근사할 수 있음을 시사합니다.

## 비판적 평가
HyperMixer는 혁신적인 접근법이지만 몇 가지 생각해 볼 점이 있습니다.
1.  **하이퍼네트워크의 오버헤드**: 동적 가중치를 생성하는 하이퍼네트워크 자체가 추가적인 파라미터와 계산량을 요구합니다. 매우 짧은 시퀀스에서는 이 오버헤드가 어텐션의 이점보다 클 수도 있습니다.
2.  **DPLR의 표현력 한계**: DPLR은 밀집 행렬(dense matrix)에 대한 강력한 근사이지만, 모든 종류의 토큰 상호작용을 완벽하게 표현하지는 못할 수 있습니다. 특정 문제에서는 어텐션의 완전한 표현력이 더 유리할 수 있습니다.
3.  **최적화의 어려움**: 하이퍼네트워크를 포함한 전체 구조는 표준 트랜스포머보다 학습이 불안정하거나 하이퍼파라미터에 더 민감할 수 있습니다.

## 향후 연구 방향
HyperMixer는 어텐션 이후의 시퀀스 모델링 아키텍처에 중요한 방향을 제시합니다.
- **더 효율적인 동적 메커니즘**: DPLR 외에 다른 행렬 분해 기법이나 구조적 근사를 탐구하여 효율성과 표현력 사이의 균형을 최적화하는 연구가 가능합니다.
- **다른 아키텍처와의 결합**: 최근 주목받는 상태 공간 모델(State Space Model, e.g., Mamba)과 HyperMixer의 아이디어를 결합하여, 순환적(recurrent) 특성과 동적 MLP의 장점을 모두 취하는 하이브리드 모델을 개발할 수 있습니다.
- **다양한 도메인으로의 확장**: 시계열 예측, 신호 처리, 그래프 데이터 등 더 넓은 범위의 시퀀스 데이터에 HyperMixer를 적용하여 그 가능성을 탐색할 수 있습니다.

## 실무 적용 가이드
HyperMixer를 실제 문제에 적용할 때 다음 사항을 고려하면 좋습니다.
- **긴 시퀀스 처리에 최적**: 수천 개 이상의 토큰으로 구성된 문서 요약, 장문 질의응답, 고해상도 이미지 분석 등 표준 트랜스포머의 적용이 어려운 문제에 우선적으로 고려해볼 수 있습니다.
- **`low_rank_dim` 튜닝**: 저랭크 파트의 차원(`low_rank_dim`)은 모델의 표현력과 계산량 사이의 균형을 조절하는 핵심 하이퍼파라미터입니다. 이 값을 조절하며 성능을 튜닝하는 것이 중요합니다.
- **사전 학습 모델 활용**: 처음부터 학습시키기보다는, 공개된 HyperMixer 기반의 사전 학습 모델을 가져와 특정 태스크에 맞게 파인튜닝하는 것이 더 효율적일 수 있습니다.

## 결론
HyperMixer는 어텐션의 $O(N^2)$ 계산 복잡도 문제를 해결하기 위해 하이퍼네트워크와 DPLR 기반의 동적 MLP를 제안한 혁신적인 아키텍처입니다. 어텐션의 동적인 표현력을 선형 계산 복잡도로 구현함으로써, 긴 시퀀스 모델링의 새로운 지평을 열었습니다. 이 연구는 향후 개발될 수많은 효율적인 시퀀스 모델링 아키텍처에 중요한 영감을 주며, 어텐션이 유일한 해답이 아님을 명확히 보여주었습니다.

## 참고 자료
- 논문 링크: [HyperMixer: An MLP-based Green Alternative to Self-Attention (arXiv:2203.03691)](https://arxiv.org/abs/2203.03691)
- 관련 논문: [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), [Pay Attention to MLPs (gMLP)](https://arxiv.org/abs/2105.08050)