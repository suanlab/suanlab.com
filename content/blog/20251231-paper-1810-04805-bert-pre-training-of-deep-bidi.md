---
title: "[논문 리뷰] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
date: "2025-12-31"
excerpt: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed t..."
category: "Paper Review"
tags: ["Paper Review","cs.CL","cs.CL"]
thumbnail: "/assets/images/blog/20251231-paper-1810-04805-bert-pre-training-of-deep-bidi.jpg"
---

# [논문 리뷰] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## TL;DR

BERT(Bidirectional Encoder Representations from Transformers)는 자연어 처리(NLP) 분야에서 혁신적인 도약을 이룬 언어 모델입니다. 이 모델은 양방향 Transformer 아키텍처를 기반으로 하여, 문맥을 양방향으로 고려함으로써 단어의 의미를 더 정확하게 파악할 수 있습니다. BERT는 두 가지 주요 사전 학습 과제인 'Masked Language Model(MLM)'과 'Next Sentence Prediction(NSP)'을 통해 훈련되며, 다양한 NLP 작업에서 최첨단 성능을 달성합니다. 특히, 질의 응답, 텍스트 분류, 언어 추론 등에서 뛰어난 성능을 보이며, 기존 모델 대비 상당한 성능 향상을 이루었습니다. BERT의 등장은 자연어 처리 연구에 큰 영향을 미쳤으며, 이후 RoBERTa, ALBERT, DistilBERT 등 다양한 변형 모델의 기반이 되었습니다.

## 연구 배경 및 동기

자연어 처리 분야는 최근 몇 년간 급격한 발전을 이루어 왔습니다. 특히, 딥러닝을 활용한 언어 모델들이 등장하면서 다양한 자연어 처리 작업에서 성능이 크게 향상되었습니다. 기존의 언어 모델들은 주로 단방향으로 문맥을 고려하였으며, 이는 단어의 의미를 정확히 파악하는 데 한계가 있었습니다. 예를 들어, GPT(Generative Pre-trained Transformer)는 왼쪽에서 오른쪽으로의 문맥만을 고려하여 단어를 예측합니다. 이로 인해 문장의 전체적인 의미를 이해하는 데 어려움이 있었고, 이는 자연어 이해(NLU) 작업에서 성능의 한계로 작용하였습니다.

BERT는 이러한 한계를 극복하기 위해 제안된 모델로, 양방향 문맥을 동시에 고려할 수 있는 능력을 갖추고 있습니다. 이는 단어가 문장에서 어떻게 사용되는지에 대한 더 깊은 이해를 가능하게 하며, 다양한 자연어 처리 작업에서의 성능을 크게 향상시킵니다. 특히, BERT는 사전 학습된 모델을 다양한 작업에 맞게 미세 조정(fine-tuning)함으로써, 적은 양의 레이블된 데이터로도 높은 성능을 달성할 수 있습니다. 이러한 점에서 BERT는 자연어 처리 분야의 새로운 패러다임을 제시하며, 기존 접근법의 한계를 극복하는 데 중요한 역할을 하고 있습니다.  예를 들어, 감성 분석에서 "The food was not good at all."이라는 문장이 있을 때, 단방향 모델은 'good'이라는 단어에 집중하여 긍정적인 의미로 해석할 수 있지만, BERT는 'not'이라는 단어와 함께 고려하여 부정적인 의미로 정확하게 해석할 수 있습니다.

## 관련 연구

BERT의 등장 이전에도 여러 언어 모델들이 자연어 처리 분야에서 사용되었습니다. 대표적인 선행 연구로는 다음과 같은 모델들이 있습니다:

1. **ELMo (Embeddings from Language Models)**: ELMo는 단방향 LSTM을 사용하여 문맥을 고려한 단어 임베딩을 생성하는 모델입니다. ELMo는 각 단어의 문맥적 의미를 반영할 수 있지만, 양방향성을 완전히 활용하지 못한다는 한계가 있습니다. ELMo는 문맥에 따라 단어의 임베딩이 달라지는 'Contextualized Word Embeddings'라는 개념을 도입했습니다.

2. **GPT (Generative Pre-trained Transformer)**: GPT는 Transformer 아키텍처를 기반으로 한 단방향 언어 모델로, 왼쪽에서 오른쪽으로의 문맥을 고려합니다. 이는 문장 생성(task)에 강점을 보이지만, 문장 이해(task)에서는 한계가 있습니다. GPT는 Transformer의 디코더 레이어만을 사용하여 autoregressive 방식으로 텍스트를 생성합니다.

3. **ULMFiT (Universal Language Model Fine-tuning)**: ULMFiT는 사전 학습된 언어 모델을 다양한 작업에 맞게 미세 조정하는 방법론을 제안합니다. 이는 BERT의 fine-tuning 접근법과 유사하지만, 양방향성을 고려하지 않는다는 차이가 있습니다. ULMFiT는 AWD-LSTM 모델을 사용하며, 'discriminative fine-tuning'과 'slanted triangular learning rates'와 같은 효과적인 fine-tuning 기법을 제안했습니다.

4. **Transformer**: Transformer는 인코더-디코더 구조를 통해 병렬 처리가 용이한 언어 모델을 제안하였습니다. BERT는 이 Transformer의 인코더 부분을 기반으로 하고 있습니다. Transformer는 Attention 메커니즘을 통해 장거리 의존성을 효과적으로 학습할 수 있습니다.

5. **Attention Mechanisms**: Attention 메커니즘은 문맥을 고려하는 데 중요한 역할을 하며, BERT는 이러한 메커니즘을 활용하여 단어 간의 관계를 학습합니다. Attention 메커니즘은 Query, Key, Value를 사용하여 입력 시퀀스 내의 각 요소에 대한 중요도를 계산합니다.

| 연구 | 모델 | 주요 특징 | BERT와의 차별점 |
| --- | --- | --- | --- |
| ELMo | LSTM 기반 | 양방향 LSTM 사용 | 완전한 양방향성 부족, feature-based approach |
| GPT | Transformer 기반 | 단방향 문맥 고려 | 양방향 문맥 미고려, fine-tuning approach |
| ULMFiT | LSTM 기반 | 사전 학습 및 미세 조정 | 양방향성 미고려, task-specific fine-tuning |
| Transformer | Transformer 기반 | 인코더-디코더 구조 | BERT는 인코더만 사용, 양방향 학습 |
| Attention Mechanisms | 다양한 모델에 적용 | 문맥적 관계 학습 | BERT는 Transformer의 Attention 사용, self-attention |

## 핵심 기여

BERT 논문은 다음과 같은 주요 기여를 통해 자연어 처리 분야에 큰 영향을 미쳤습니다:

1. **양방향 문맥 학습**: BERT는 Transformer 인코더를 활용하여 양방향 문맥을 동시에 고려하는 언어 모델을 제안하였습니다. 이는 단어의 의미를 더 정확하게 파악할 수 있도록 하였습니다.

2. **사전 학습과 미세 조정**: BERT는 대규모의 비지도 학습을 통해 사전 학습된 모델을 제공하며, 이를 다양한 작업에 맞게 미세 조정할 수 있는 방법론을 제안하였습니다.

3. **최첨단 성능 달성**: BERT는 GLUE, SQuAD 등 다양한 자연어 처리 벤치마크에서 기존의 최첨단 성능을 뛰어넘는 결과를 달성하였습니다.

4. **모델의 확장성**: BERT는 이후 다양한 변형 모델(예: RoBERTa, ALBERT, DistilBERT)의 기반이 되었으며, 자연어 처리 연구의 발전을 이끌었습니다.

5. **간단한 구조와 강력한 성능**: BERT는 복잡한 구조 변경 없이도 다양한 작업에서 높은 성능을 보이며, 자연어 처리 모델의 설계에 새로운 방향성을 제시하였습니다. BERT는 Transformer 인코더 레이어를 쌓아 올린 비교적 간단한 구조를 가지고 있지만, MLM과 NSP라는 효과적인 사전 학습 방법을 통해 강력한 성능을 달성했습니다.

## 제안 방법론

BERT의 제안 방법론은 양방향 Transformer 아키텍처를 기반으로 하여, 문맥을 양방향으로 고려하는 언어 모델을 만드는 것입니다. 이를 위해 BERT는 두 가지 주요 사전 학습 과제인 'Masked Language Model(MLM)'과 'Next Sentence Prediction(NSP)'을 사용합니다.

### 핵심 아이디어와 이론적 근거

BERT의 핵심 아이디어는 양방향 문맥을 고려하여 단어의 의미를 학습하는 것입니다. 기존의 단방향 모델들은 문맥의 한 방향만을 고려하여 단어를 예측하였으나, BERT는 양방향 문맥을 동시에 고려함으로써 단어의 의미를 더 정확하게 파악할 수 있습니다. 이는 자연어 이해 작업에서 특히 유리하게 작용하며, 다양한 NLP 작업에서의 성능 향상으로 이어집니다.  이러한 양방향 학습은 단어의 의미가 주변 단어에 따라 달라지는 자연어의 특성을 더 잘 반영합니다.

### 모델 아키텍처 상세 설명

BERT는 Transformer 인코더를 기반으로 하며, 여러 층의 self-attention과 feed-forward 네트워크로 구성됩니다. BERT-base 모델은 12개의 인코더 층, 768의 hidden size, 12개의 attention heads로 구성되어 있으며, BERT-large 모델은 24개의 인코더 층, 1024의 hidden size, 16개의 attention heads로 구성되어 있습니다.  각 인코더 층은 multi-head self-attention 메커니즘과 feed-forward 네트워크로 구성되어 있으며, layer normalization과 residual connection을 사용하여 학습을 안정화합니다.

### 핵심 수식

1. **Self-Attention**: 각 단어의 표현을 계산하기 위해 self-attention 메커니즘을 사용합니다. self-attention은 다음과 같이 정의됩니다:

   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

   여기서 $Q$, $K$, $V$는 각각 Query, Key, Value 행렬을 나타내며, $d_k$는 키 벡터의 차원입니다.  Self-Attention은 입력 시퀀스 내의 각 단어와 다른 단어 간의 관계를 모델링합니다.

2. **Masked Language Model (MLM)**: 입력 시퀀스의 15%의 토큰을 무작위로 선택하여 마스킹합니다. 마스킹된 토큰 중 80%는 `[MASK]` 토큰으로 대체되고, 10%는 무작위 토큰으로 대체되며, 나머지 10%는 원래 토큰으로 유지됩니다. MLM 손실 함수는 다음과 같이 정의됩니다:

   $$L_{MLM} = - \sum_{i=1}^{m} \log P(x_i | x_{\setminus i})$$

   여기서 $x_{\setminus i}$는 마스크되지 않은 토큰을 나타내고, $m$은 마스크된 토큰의 개수입니다.  MLM은 모델이 문맥을 이해하고 단어의 의미를 추론하는 능력을 향상시킵니다.

3. **Next Sentence Prediction (NSP)**: 두 문장 A와 B를 입력으로 받아, 문장 B가 문장 A 다음에 이어지는 문장인지 (IsNext) 또는 임의의 문장인지 (NotNext)를 예측합니다. NSP 손실 함수는 다음과 같이 정의됩니다:

   $$L_{NSP} = - \log P(IsNext | A, B)$$

   NSP는 모델이 문장 간의 관계를 학습하고, 긴 텍스트의 구조를 이해하는 데 도움을 줍니다. 하지만, 이후 연구에서 NSP의 효과에 대한 논쟁이 있었습니다.

4. **전체 손실 함수**: MLM 손실과 NSP 손실의 합으로 정의됩니다:

   $$L = L_{MLM} + L_{NSP}$$

5. **Feed-Forward Network**: 각 인코더 층에서 self-attention의 출력을 처리하는 feed-forward 네트워크는 다음과 같이 정의됩니다:

   $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

   여기서 $W_1$, $W_2$, $b_1$, $b_2$는 학습 가능한 가중치와 편향입니다.  Feed-Forward Network는 각 단어의 표현을 변환하고, 비선형성을 추가합니다.

### Python/PyTorch 구현 코드

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

# BERT tokenizer와 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 입력 문장
text = "The quick brown [MASK] jumps over the lazy dog."
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])

# [MASK] 토큰의 위치 찾기
mask_token_index = tokenized_text.index('[MASK]')

# 모델 평가 모드로 설정
model.eval()

# 예측 수행
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0, mask_token_index]

# 예측된 토큰 찾기
predicted_index = torch.argmax(predictions).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(f"Predicted token: {predicted_token}") # 출력: fox
```

**코드 설명:**

1.  **라이브러리 임포트**: `transformers` 라이브러리에서 `BertTokenizer`와 `BertForMaskedLM`을 임포트합니다. `torch`는 PyTorch 텐서 연산을 위해 임포트합니다.
2.  **토크나이저 및 모델 로드**: `bert-base-uncased` 사전 학습 모델과 해당 토크나이저를 로드합니다.  `BertTokenizer`는 텍스트를 BERT 모델이 처리할 수 있는 형태로 변환하고, `BertForMaskedLM`은 마스크된 단어를 예측하는 데 사용됩니다.
3.  **입력 문장 준비**: 마스크 토큰 `[MASK]`가 포함된 입력 문장을 정의합니다.
4.  **토큰화 및 인덱싱**: 문장을 토큰화하고, 각 토큰을 해당 ID로 변환합니다.  `tokenizer.tokenize(text)`는 문장을 토큰 리스트로 분할하고, `tokenizer.convert_tokens_to_ids(tokenized_text)`는 각 토큰을 사전 정의된 vocabulary의 해당 ID로 매핑합니다.
5.  **텐서 변환**: 토큰 ID 리스트를 PyTorch 텐서로 변환합니다.
6.  **마스크 토큰 위치 찾기**: 입력 문장에서 `[MASK]` 토큰의 인덱스를 찾습니다.
7.  **모델 평가 모드 설정**: 모델을 평가 모드로 설정하여 추론(inference)에 최적화합니다.
8.  **예측 수행**: `torch.no_grad()` 컨텍스트 내에서 모델을 실행하여 gradient 계산을 비활성화하고 메모리 사용량을 줄입니다. 모델은 입력 텐서를 받아 마스크된 토큰에 대한 예측값을 출력합니다.
9.  **예측된 토큰 추출**: 마스크된 위치에 대한 예측값을 추출하고, 가장 가능성이 높은 토큰의 ID를 찾습니다.  `torch.argmax(predictions)`는 가장 높은 확률을 가진 토큰의 인덱스를 반환하고, `tokenizer.convert_ids_to_tokens([predicted_index])[0]`는 해당 인덱스를 다시 토큰으로 변환합니다.
10. **결과 출력**: 예측된 토큰을 출력합니다.

## 실험 설정

BERT의 성능을 평가하기 위해 다양한 자연어 처리 벤치마크가 사용되었습니다. 대표적인 벤치마크로는 GLUE(General Language Understanding Evaluation)와 SQuAD(Stanford Question Answering Dataset)가 있습니다.

### 데이터셋

- **BooksCorpus**: 800M 단어로 구성된 대규모 텍스트 데이터셋. 주로 소설 장르의 책으로 구성되어 있습니다.
- **Wikipedia**: 2,500M 단어로 구성된 영어 Wikipedia 데이터셋. 다양한 주제에 대한 정보를 포함하고 있습니다.

### 평가 지표

- **GLUE 벤치마크**: 다양한 자연어 이해 작업을 평가하는 종합적인 벤치마크.  GLUE는 텍스트 유사성, 자연어 추론, 감성 분석 등 다양한 task를 포함합니다.
- **SQuAD**: 질의 응답 작업에서의 정확도를 평가하는 데이터셋.  SQuAD는 주어진 문맥(context) 내에서 질문에 대한 답변을 찾는 task입니다.

### 베이스라인

- **ELMo, GPT**: 기존의 단방향 언어 모델들
- **BERT-base**: 12개의 인코더 층, 768의 hidden size, 12개의 attention heads
- **BERT-large**: 24개의 인코더 층, 1024의 hidden size, 16개의 attention heads

### 하이퍼파라미터

| 하이퍼파라미터 | BERT-base | BERT-large |
| --- | --- | --- |
| 인코더 층 수 | 12 | 24 |
| Hidden size | 768 | 1024 |
| Attention heads | 12 | 16 |
| 학습률 | 1e-4 | 1e-4 |
| 배치 크기 | 32 | 32 |
| 학습 에폭 | 3 | 3 |

**참고**: 위 하이퍼파라미터는 일반적인 설정이며, 특정 작업에 따라 최적의 값이 달라질 수 있습니다.

## 실험 결과 분석

BERT는 다양한 자연어 처리 벤치마크에서 최첨단 성능을 달성하였습니다. 특히, GLUE와 SQuAD에서의 성능 향상이 두드러집니다.

### 주요 결과

| 작업 | 기존 SOTA | BERT-base | BERT-large |
| --- | --- | --- | --- |
| GLUE | 72.0 | 80.5 | 82.1 |
| SQuAD v1.1 | 91.7 | 93.2 | 94.6 |
| SQuAD v2.0 | 78.0 | 83.1 | 86.2 |

### 성능 향상률

- **GLUE**: BERT-base는 기존 SOTA 대비 8.5% 향상, BERT-large는 10.1% 향상
- **SQuAD v1.1**: BERT-base는 1.5% 향상, BERT-large는 2.9% 향상
- **SQuAD v2.0**: BERT-base는 5.1% 향상, BERT-large는 8.2% 향상

### Ablation Study

BERT의 성능 향상은 양방향 문맥 학습과 사전 학습된 모델의 미세 조정에 크게 기인합니다. Ablation Study를 통해 MLM과 NSP의 효과를 분석한 결과, 두 과제 모두 성능 향상에 기여함을 확인하였습니다. 특히, MLM은 단어의 문맥적 의미를 학습하는 데 중요한 역할을 하며, NSP는 문장 간의 관계를 이해하는 데 도움을 줍니다.  하지만, 이후 연구에서는 NSP task의 효용성에 대한 의문이 제기되었고, RoBERTa와 같은 모델에서는 NSP task를 제거하거나 다른 방식으로 대체하여 성능 향상을 이루었습니다.

## 비판적 평가

### 강점

1. **양방향 문맥 학습**: BERT는 양방향 문맥을 고려함으로써 단어의 의미를 더 정확하게 파악할 수 있습니다.
2. **사전 학습과 미세 조정**: 대규모 비지도 학습을 통해 사전 학습된 모델을 다양한 작업에 쉽게 적용할 수 있습니다.
3. **최첨단 성능**: 다양한 자연어 처리 벤치마크에서 기존의 최첨단 성능을 뛰어넘는 결과를 달성하였습니다.

### 한계점과 개선 방향

1. **모델 크기**: BERT-large 모델은 많은 파라미터를 가지고 있어, 학습 및 추론 속도가 느립니다. 이를 개선하기 위해 경량화된 모델(예: DistilBERT, MobileBERT)이 필요합니다.
2. **NSP의 효과**: NSP의 효과에 대한 논쟁이 있으며, 일부 연구에서는 NSP를 제거하거나 다른 형태의 문장 관계 예측 방법을 사용하기도 합니다. RoBERTa는 NSP task를 제거하고 더 많은 데이터를 사용하여 학습하여 성능을 향상시켰습니다.
3. **고정된 마스크**: MLM에서 사용되는 `[MASK]` 토큰은 fine-tuning 단계에서는 나타나지 않아 pretrain-finetune discrepancy 문제가 발생할 수 있습니다. 이를 해결하기 위해 SpanBERT와 같은 모델에서는 연속된 토큰을 마스킹하는 방법을 사용합니다.

### 재현성 평가

BERT는 공개된 코드와 사전 학습된 모델을 통해 쉽게 재현할 수 있습니다. 이는 연구의 신뢰성을 높이는 데 기여합니다.  Hugging Face의 `transformers` 라이브러리를 통해 BERT 모델을 쉽게 사용할 수 있으며, 다양한 사전 학습 모델을 다운로드하여 fine-tuning할 수 있습니다.

## 향후 연구 방향

1. **모델 경량화**: DistilBERT와 같은 경량화된 모델을 개발하여, BERT의 성능을 유지하면서도 속도를 개선할 수 있습니다.  모델 경량화는 모바일 기기나 임베디드 시스템과 같이 자원 제약적인 환경에서 BERT 모델을 사용할 수 있도록 합니다.
2. **다양한 언어로의 확장**: BERT를 다양한 언어로 확장하여, 다국어 자연어 처리 작업에서의 성능을 향상시킬 수 있습니다.  Multilingual BERT (mBERT)는 100개 이상의 언어로 학습되었으며, cross-lingual transfer learning에 사용될 수 있습니다.
3. **문장 관계 예측의 개선**: NSP의 한계를 극복하기 위해, 문장 관계 예측을 위한 새로운 방법론을 개발할 수 있습니다.  Sentence-BERT (SBERT)는 문장 임베딩을 생성하고, 문장 간의 유사도를 효율적으로 계산할 수 있도록 설계되었습니다.
4. **지식 통합**: 외부 지식 베이스를 활용하여 BERT의 성능을 향상시킬 수 있습니다.  Knowledge-enhanced BERT (KnowBERT)는 entity embedding을 사용하여 외부 지식을 통합합니다.

## 실무 적용 가이드

BERT를 실무에 적용할 때는 다음과 같은 사항을 고려해야 합니다:

1. **모델 선택**: BERT-base와 BERT-large 중 작업의 복잡도와 리소스를 고려하여 적절한 모델을 선택합니다.  일반적으로 BERT-large 모델이 더 높은 성능을 보이지만, 더 많은 컴퓨팅 자원을 필요로 합니다.
2. **미세 조정**: 사전 학습된 BERT 모델을 작업에 맞게 미세 조정하여, 최적의 성능을 달성합니다.  Fine-tuning은 task-specific 데이터를 사용하여 모델의 가중치를 업데이트하는 과정입니다.
3. **하드웨어 요구사항**: BERT-large 모델은 많은 메모리를 요구하므로, 적절한 하드웨어 환경을 준비합니다.  GPU를 사용하면 학습 및 추론 속도를 크게 향상시킬 수 있습니다.
4. **토큰 길이 제한**: BERT는 입력 토큰의 길이에 제한(일반적으로 512 토큰)이 있습니다. 긴 텍스트를 처리해야 하는 경우, 텍스트를 분할하거나 truncation 기법을 사용해야 합니다.
5. **최적화**: 모델 경량화 기법(예: quantization, pruning)을 사용하여 모델 크기를 줄이고 추론 속도를 높일 수 있습니다.

## 결론

BERT는 양방향 사전 학습을 통해 다양한 자연어 처리 작업에서 뛰어난 성능을 발휘하며, 기존의 단방향 모델의 한계를 극복하였습니다. BERT의 등장 이후, 자연어 처리 분야는 획기적인 발전을 이루었으며, BERT는 이후 등장하는 많은 언어 모델들의 기반이 되었습니다. BERT는 자연어 처리 연구에 큰 영향을 미쳤으며, 다양한 변형 모델의 개발을 촉진하였습니다. BERT는 자연어 이해, 질의 응답, 텍스트 분류 등 다양한 NLP task에서 강력한 성능을 보여주며, 실무에서도 널리 사용되고 있습니다.

## 참고 자료

- [BERT 논문](https://arxiv.org/abs/1810.04805)
- [BERT 코드 저장소](https://github.com/google-research/bert)
- [Transformers 라이브러리](https://github.com/huggingface/transformers)
- [RoBERTa 논문](https://arxiv.org/abs/1907.11692)
- [ALBERT 논문](https://arxiv.org/abs/1909.11942)
- [DistilBERT 논문](https://arxiv.org/abs/1910.01108)
