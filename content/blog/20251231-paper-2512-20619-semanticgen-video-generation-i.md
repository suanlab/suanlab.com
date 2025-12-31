---
title: "[논문 리뷰] SemanticGen: Video Generation in Semantic Space"
date: "2025-12-31"
excerpt: "State-of-the-art video generative models typically learn the distribution of video latents in the VAE space and map them to pixels using a VAE decoder. While this approach can generate high-quality vi..."
category: "Paper Review"
tags: ["Paper Review","cs.CV","cs.CV"]
thumbnail: "/assets/images/blog/20251231-paper-2512-20619-semanticgen-video-generation-i.jpg"
---

# [논문 리뷰] SemanticGen: Video Generation in Semantic Space

## TL;DR
SemanticGen은 비디오 생성의 새로운 패러다임을 제시합니다. 기존 VAE(Variational Autoencoder) 기반 방법론의 느린 수렴 속도와 높은 계산 비용 문제를 해결하기 위해, SemanticGen은 고차원 의미 공간에서 비디오 생성을 시작하여 더 빠르고 효율적인 학습을 가능하게 합니다. 이 연구는 비디오 생성의 초기 단계를 압축된 의미 공간에서 수행, 전역적인 비디오 레이아웃을 정의하고, 이후 고주파수 세부 사항을 추가하는 두 단계 생성 과정을 제안합니다. 실험 결과, SemanticGen은 뛰어난 품질의 비디오를 생성하며, 기존 방법론을 능가하는 성능을 보여줍니다. 특히 긴 비디오 생성에서 일관성 유지에 강점을 보입니다.

## 연구 배경 및 동기
비디오 생성 모델은 최근 몇 년간 급속한 발전을 이루었습니다. 특히, 확산 기반(Diffusion-based) 방법론은 VAE를 사용하여 비디오를 픽셀 공간에서 잠재 공간으로 투영하고, 그 후 확산 모델을 학습하여 VAE 잠재 분포를 맞추는 방식을 취하고 있습니다. 그러나 이러한 접근 방식은 두 가지 주요 한계점을 가지고 있습니다.

첫째, 수렴 속도가 느립니다. 고품질 비디오를 생성하기 위해서는 수십만 GPU 시간을 필요로 하는 대규모 계산 자원이 요구됩니다. 이는 보다 효율적인 학습 패러다임의 필요성을 강조합니다. 둘째, 긴 비디오로의 확장이 어렵습니다. 현대의 VAE는 보통 낮은 압축 비율을 가지고 있어, 60초, 480p, 24fps 비디오 클립은 50만 개가 넘는 토큰으로 확장됩니다. 이는 양방향 전체 주의 확산 모델링을 비실용적으로 만듭니다. 이전 연구들은 희소 주의(sparse attention)나 자율회귀(autoregressive) 및 확산-자율회귀 하이브리드 프레임워크를 채택하여 복잡성을 줄이려 했으나, 이들은 종종 시각적 품질의 저하를 초래합니다. 예를 들어, 긴 비디오에서 객체의 외형이 갑자기 변하거나 배경이 부자연스럽게 바뀌는 현상이 발생할 수 있습니다.

SemanticGen은 이러한 한계점을 극복하기 위해 비디오 생성을 고차원의 의미 공간에서 시작하여, 이후 VAE 잠재 공간으로 매핑하는 프레임워크를 제안합니다. 비디오에 내재된 중복성을 감안할 때, 생성은 먼저 전역적인 계획을 위한 고수준의 압축된 의미 공간에서 이루어져야 하며, 이후 시각적 세부 사항을 추가하는 것이 바람직합니다.  이는 마치 건축가가 건물의 전체적인 구조를 먼저 설계한 후, 세부적인 인테리어를 진행하는 것과 유사합니다.

## 관련 연구
기존의 비디오 생성 연구는 크게 확산 기반, 자율회귀 기반, 그리고 이들의 하이브리드 변형으로 나눌 수 있습니다. 확산 기반 방법론은 모든 프레임을 양방향 주의로 모델링하고 동시에 생성하며, 자율회귀 기법은 각 프레임이나 패치를 순차적으로 생성합니다. 이러한 접근 방식들은 각각의 장점을 가지고 있으나, 긴 비디오 생성 시에는 성능이 저하되는 문제가 있습니다.  예를 들어, 확산 모델은 계산 비용이 높고, 자율회귀 모델은 오류가 누적되는 경향이 있습니다.

최근 연구들은 생성 모델의 성능을 향상시키기 위해 의미 표현을 통합하는 방법을 탐구하고 있습니다. 예를 들어, VA-VAE는 VAE 잠재 공간을 사전 학습된 의미 표현과 정렬하고, DC-AE 및 MAETok는 VAE 학습에 의미적 목표를 통합합니다. 이러한 접근 방식은 SemanticGen과는 독립적이며, SemanticGen은 특정 VAE 토크나이저에 묶이지 않는 프레임워크입니다.  VA-VAE는 이미지 캡셔닝 모델을 사용하여 의미 정보를 추출하고, 이를 VAE 학습에 활용합니다.

TokensGen과 같은 방법은 비디오 생성을 위한 2단계 패러다임을 채택하지만, 의미적 특징을 사용하는 대신 VAE 잠재 공간을 추가로 압축합니다. 그러나 SemanticGen은 의미 공간에서의 생성이 VAE 잠재 공간에서의 모델링과 근본적으로 다르며, 특히 의미 공간은 훨씬 빠른 수렴을 보인다는 점에서 차별화됩니다.  TokensGen은 VAE 잠재 공간을 Quantization하여 압축하는 방식을 사용합니다.

## 제안하는 방법론
SemanticGen의 핵심 아이디어는 비디오 생성을 압축된 의미 공간에서 시작하여, 이후 VAE 잠재 공간으로 매핑하는 것입니다. 이는 두 단계의 생성 과정을 따릅니다.

첫 번째 단계에서는, 확산 모델을 사용하여 비디오의 전역 레이아웃을 정의하는 압축된 의미 비디오 특징을 생성합니다. 두 번째 단계에서는, 이러한 의미적 특징을 조건으로 VAE 잠재 공간에서 최종 출력을 생성합니다. 이러한 접근 방식은 VAE 잠재 공간에 비해 빠른 수렴을 제공하며, 긴 비디오 생성으로 확장할 때도 효과적이고 계산 효율적입니다.  이는 마치 화가가 스케치를 먼저 그린 후, 채색을 하는 것과 유사합니다. 스케치는 전체적인 구도를 나타내고, 채색은 세부적인 표현을 담당합니다.

### 모델 아키텍처
SemanticGen은 두 단계의 생성 과정을 따릅니다. 첫 번째 단계에서는, 사전 학습된 비디오 이해 토크나이저를 의미 인코더로 사용하여 비디오를 의미 공간으로 변환합니다. 이때, 높은 차원의 의미 공간에서 직접 샘플링할 경우 수렴 속도가 느려질 수 있으므로, 경량의 MLP를 사용하여 의미 공간을 압축합니다.  비디오 이해 토크나이저로는 CLIP 또는 BERT와 같은 모델이 사용될 수 있습니다.

두 번째 단계에서는, 압축된 의미 표현을 조건으로 VAE 잠재 공간에서 비디오를 생성합니다. 이 과정에서, 의미 표현은 비디오의 전역적인 구조와 움직임 패턴을 인코딩하며, 낮은 수준의 속성은 제외됩니다.  이는 VAE 디코더가 의미 표현을 기반으로 비디오의 세부적인 내용을 채워나가는 과정입니다.

### Python/PyTorch 코드 예제
```python
import torch
import torch.nn as nn

class SemanticGen(nn.Module):
    def __init__(self, semantic_encoder, vae_encoder, vae_decoder):
        super(SemanticGen, self).__init__()
        self.semantic_encoder = semantic_encoder
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.mlp = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, video):
        # 1. Semantic Encoding
        semantic_features = self.semantic_encoder(video)  # (B, T, D) where B is batch size, T is time, D is feature dimension
        # 2. Compression
        compressed_features = self.mlp(semantic_features) # (B, T, 8)
        # 3. VAE Encoding (conditioning on compressed features)
        vae_latents = self.vae_encoder(compressed_features) # (B, T, Z) where Z is VAE latent dimension
        # 4. VAE Decoding
        generated_video = self.vae_decoder(vae_latents) # (B, T, H, W, C) where H is height, W is width, C is channels
        return generated_video
```
**코드 설명:**

*   `semantic_encoder`: 사전 학습된 의미 인코더 (예: CLIP, BERT). 비디오 프레임을 의미 공간으로 임베딩합니다.
*   `vae_encoder`, `vae_decoder`: VAE의 인코더와 디코더.
*   `mlp`: 의미 공간을 압축하는 MLP.
*   `forward` 함수: 전체 생성 과정을 정의합니다.

## 실험 설정
SemanticGen의 성능을 평가하기 위해 다양한 실험을 수행하였습니다. 실험은 주로 두 가지 측면에서 이루어졌습니다: 짧은 비디오 생성과 긴 비디오 생성.

### 데이터셋 설명
짧은 비디오 생성 실험에서는 내부 텍스트-비디오 쌍 데이터셋을 사용하였으며, 긴 비디오 생성 실험에서는 영화 및 TV 쇼 클립을 60초 세그먼트로 나누어 사용하였습니다. 각 비디오에 대해 내부 캡셔너를 사용하여 텍스트 프롬프트를 생성하였습니다.  내부 캡셔너는 비디오의 내용을 설명하는 텍스트를 자동으로 생성하는 모델입니다.

### 평가 지표
비디오 품질 평가는 VBench 및 VBench-Long 벤치마크를 사용하여 수행되었습니다. 긴 비디오 생성의 경우, 품질 드리프트를 측정하기 위해 FramePack에서 제안한 $$\Delta M_{drift}$$ 지표를 추가로 사용하였습니다.  VBench는 비디오 생성 모델의 성능을 종합적으로 평가하는 벤치마크입니다. $$\Delta M_{drift}$$는 비디오의 시작 부분과 끝 부분의 품질 차이를 측정하는 지표입니다.

### 비교 대상 (baseline)
짧은 비디오 생성의 경우, Wan2.1-T2V-14B와 HunyuanVideo를 비교 대상으로 사용하였으며, 긴 비디오 생성의 경우, SkyReels-V2, Self-Forcing, LongLive와 비교하였습니다. 각 베이스라인은 서로 다른 기본 모델, 학습 데이터, 학습 단계를 사용하여 공정한 비교가 어려운 점을 감안하여, 동일한 데이터와 학습 단계를 유지하면서 표준 확산 손실을 사용하는 추가 베이스라인(Base-CT, Base-Swin-CT)을 포함하였습니다.  이는 SemanticGen의 성능을 객관적으로 평가하기 위한 노력입니다.

### 하이퍼파라미터 설정
학습 과정에서 사용된 주요 하이퍼파라미터는 다음과 같습니다:
- 비디오 샘플링 속도: fps=24 (VAE 입력), fps=1.6 (의미 인코더 입력)
- 의미 공간 압축: MLP 출력 차원 = 8

## 실험 결과 및 분석
SemanticGen은 짧은 비디오와 긴 비디오 생성 모두에서 뛰어난 성능을 보여주었습니다.

### 주요 정량적 결과
짧은 비디오 생성에서는, SemanticGen이 텍스트 프롬프트를 따르는 정확도에서 베이스라인 방법을 능가하였습니다. 예를 들어, 베이스라인은 남성이 왼쪽으로 머리를 돌리는 장면이나 눈송이가 녹는 과정을 제대로 생성하지 못했습니다. 긴 비디오 생성에서는, SemanticGen이 장기적인 일관성을 더 잘 유지하며 드리프트 문제를 크게 완화하였습니다.

| Method | Subject Consistency | Background Consistency | Temporal Flickering | Motion Smoothness | Imaging Quality | Aesthetic Quality |
|--------|---------------------|-----------------------|--------------------|------------------|----------------|------------------|
| Hunyuan-Video | 91.11% | 95.32% | 97.49% | 99.07% | 64.23% | 62.60% |
| Wan2.1-T2V-14B | 97.23% | 98.28% | 98.35% | 99.08% | 66.63% | 65.61% |
| Base-CT | 96.17% | 97.27% | 98.07% | 99.07% | 65.77% | 63.97% |
| **SemanticGen** | **97.79%** | **97.68%** | **98.47%** | **99.17%** | **65.23%** | **64.60%** |

**표 해석:**

*   **Subject Consistency:** 비디오 내에서 주요 객체가 일관성을 유지하는 정도.
*   **Background Consistency:** 배경의 일관성 유지 정도.
*   **Temporal Flickering:** 시간 흐름에 따른 깜빡임 현상 정도. 낮을수록 좋음.
*   **Motion Smoothness:** 움직임의 부드러움 정도.
*   **Imaging Quality:** 이미지 품질.
*   **Aesthetic Quality:** 심미적인 품질.

SemanticGen은 주체 일관성, 배경 일관성, 시간 깜빡임, 움직임 부드러움 측면에서 가장 우수한 성능을 보입니다.

### 정성적 분석
SemanticGen은 텍스트 프롬프트에 따라 고품질의 비디오를 생성하는 능력을 보여주었으며, 긴 비디오 생성에서는 일관성을 유지하면서 드리프트 문제를 크게 완화하였습니다. 베이스라인에서는 프레임 간 일관성이 부족하거나 아티팩트가 더 많이 나타나는 반면, SemanticGen은 고수준 의미 공간에서의 전역 계획을 통해 이러한 문제를 해결하였습니다.  예를 들어, 베이스라인 모델은 긴 비디오에서 갑자기 조명이 바뀌거나, 객체의 모양이 변하는 등의 문제를 보였지만, SemanticGen은 이러한 문제를 효과적으로 해결했습니다.

### Ablation Study 결과
SemanticGen의 두 가지 주요 구성 요소인 의미 공간 압축과 의미 표현 생성의 효과를 검증하기 위한 ablation study를 수행하였습니다. 실험 결과, 의미 공간을 압축함으로써 학습 속도가 가속화되었으며, 의미 표현을 사용한 경우 VAE 잠재 공간을 직접 모델링하는 것에 비해 수렴 속도가 크게 향상되었습니다.  이는 SemanticGen의 각 구성 요소가 전체 성능 향상에 기여함을 보여줍니다.

## 한계점 및 향후 연구 방향
SemanticGen은 여러 장점을 가지고 있지만, 몇 가지 한계점도 존재합니다. 긴 비디오 생성에서는 텍스처 일관성을 유지하는 데 어려움이 있으며, 이는 의미적 특징이 세부 사항을 완전히 보존하지 못하기 때문입니다. 또한, SemanticGen은 의미 인코더의 제약을 상속받습니다. 예를 들어, 낮은 fps로 샘플링하면 고주파수 시간 정보를 잃게 됩니다.  이는 SemanticGen이 아직 완벽하지 않으며, 개선의 여지가 있음을 의미합니다.

향후 연구에서는 보다 강력한 비디오 의미 인코더를 개발하여 이러한 한계를 극복할 수 있을 것입니다. 예를 들어, 높은 시간 압축과 높은 샘플링 속도를 동시에 달성하는 토크나이저는 SemanticGen의 성능을 더욱 향상시킬 수 있을 것입니다. 또한, 텍스처 일관성을 유지하기 위한 추가적인 모듈을 개발하는 것도 좋은 연구 방향이 될 수 있습니다.

## 결론 및 시사점
SemanticGen은 비디오 생성을 위한 새로운 패러다임을 제안하여, 고차원의 의미 공간에서 비디오 생성을 시작하여 보다 빠르고 효율적인 학습을 가능하게 합니다. 이 연구는 비디오 생성의 초기 단계를 압축된 의미 공간에서 수행함으로써, 전역적인 비디오 레이아웃을 정의하고, 이후 고주파수 세부 사항을 추가하는 두 단계의 생성 과정을 제안합니다. 실험 결과, SemanticGen은 뛰어난 품질의 비디오를 생성하며, 기존 방법론을 능가하는 성능을 보여줍니다.

실무적으로, SemanticGen은 다양한 비디오 생성 작업에 적용될 수 있으며, 특히 긴 비디오 생성에서의 일관성 문제를 해결하는 데 기여할 수 있습니다.  예를 들어, 영화 제작, 게임 개발, 광고 제작 등 다양한 분야에서 활용될 수 있습니다. 개인적으로, SemanticGen의 접근 방식은 비디오 생성의 효율성을 크게 향상시킬 수 있는 잠재력을 가지고 있으며, 향후 연구에서 더욱 발전할 가능성이 높다고 생각합니다.  특히, 생성 AI 분야에 큰 영향을 미칠 것으로 기대됩니다.
