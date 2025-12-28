---
title: "Transformer 아키텍처 완벽 이해"
date: "2024-12-10"
excerpt: "GPT, BERT의 기반이 되는 Transformer 아키텍처를 상세히 알아봅니다."
category: "Deep Learning"
tags: ["Transformer", "NLP", "Attention", "Deep Learning"]
thumbnail: "/assets/images/research/nlp.jpg"
---

# Transformer 아키텍처 완벽 이해

2017년 Google이 발표한 "Attention is All You Need" 논문에서 소개된 **Transformer**는 현대 NLP의 근간이 되는 아키텍처입니다.

## 왜 Transformer인가?

기존 RNN/LSTM의 한계:
- 순차적 처리로 인한 병렬화 어려움
- 긴 시퀀스에서 정보 손실
- 학습 시간이 오래 걸림

Transformer의 장점:
- 완전한 병렬 처리 가능
- Self-Attention으로 장거리 의존성 포착
- 확장성이 뛰어남

## 핵심 구성요소

### 1. Self-Attention

Self-Attention은 입력 시퀀스 내 각 토큰이 다른 모든 토큰과의 관계를 계산합니다.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **Query (Q)**: 현재 토큰의 질의
- **Key (K)**: 다른 토큰들의 키
- **Value (V)**: 실제 정보를 담은 값

### 2. Multi-Head Attention

여러 개의 Attention을 병렬로 수행하여 다양한 관점에서 정보를 수집합니다.

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

### 3. Position Encoding

Transformer는 순서 정보가 없으므로, 위치 인코딩을 추가합니다.

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 4. Feed-Forward Network

각 위치에 독립적으로 적용되는 완전연결 신경망입니다.

```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

## Encoder와 Decoder

### Encoder
- 입력 시퀀스를 인코딩
- Self-Attention + Feed-Forward
- 6개의 동일한 레이어 스택

### Decoder
- 출력 시퀀스 생성
- Masked Self-Attention + Cross-Attention + Feed-Forward
- Auto-regressive 방식

## 대표적인 모델들

| 모델 | 구조 | 주요 용도 |
|------|------|----------|
| BERT | Encoder only | 분류, NER, QA |
| GPT | Decoder only | 텍스트 생성 |
| T5 | Encoder-Decoder | 범용 |

## 마무리

Transformer는 NLP를 넘어 Computer Vision, Audio 등 다양한 분야로 확장되고 있습니다. 다음 포스트에서는 Vision Transformer를 다루겠습니다.
