---
title: "머신러닝의 수학적 기초"
date: "2024-03-08"
excerpt: "머신러닝 알고리즘의 핵심이 되는 수학적 개념들 - 선형대수, 미적분, 확률/통계를 알아봅니다."
category: "Machine Learning"
tags: ["Machine Learning", "Mathematics", "Linear Algebra", "Calculus", "Statistics"]
thumbnail: "/assets/images/blog/ml-math.jpg"
---

# 머신러닝의 수학적 기초

머신러닝을 깊이 이해하기 위해서는 수학적 기초가 필수적입니다. 이 글에서는 머신러닝에서 자주 사용되는 핵심 수학 개념들을 살펴보겠습니다.

## 1. 선형대수 (Linear Algebra)

### 벡터와 행렬

머신러닝에서 데이터는 주로 벡터와 행렬로 표현됩니다. 벡터 $\mathbf{x}$는 다음과 같이 표현합니다:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$$

### 행렬 곱셈

두 행렬 $\mathbf{A}$와 $\mathbf{B}$의 곱셈은 다음과 같이 정의됩니다:

$$(\mathbf{AB})_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

### 내적 (Dot Product)

두 벡터의 내적은 유사도 측정에 자주 사용됩니다:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

## 2. 미적분 (Calculus)

### 경사하강법

신경망 학습의 핵심인 경사하강법은 손실 함수의 기울기를 이용합니다:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta)$$

여기서 $\eta$는 학습률(learning rate), $\nabla_\theta L$은 손실 함수의 기울기입니다.

### 편미분

다변수 함수에서 각 변수에 대한 편미분:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

### 연쇄 법칙 (Chain Rule)

역전파(backpropagation)의 핵심인 연쇄 법칙:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$$

## 3. 확률과 통계

### 베이즈 정리

머신러닝의 많은 알고리즘이 베이즈 정리를 기반으로 합니다:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

또는 더 일반적으로:

$$P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) P(\theta)}{P(\mathcal{D})}$$

여기서:
- $P(\theta | \mathcal{D})$: 사후 확률 (posterior)
- $P(\mathcal{D} | \theta)$: 가능도 (likelihood)
- $P(\theta)$: 사전 확률 (prior)

### 가우시안 분포

정규 분포(가우시안 분포)의 확률 밀도 함수:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

다변량 가우시안 분포:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

### 엔트로피

정보 엔트로피는 불확실성을 측정합니다:

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

교차 엔트로피 (분류 문제의 손실 함수):

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

## 4. 최적화 (Optimization)

### 손실 함수

선형 회귀의 MSE 손실:

$$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T\theta)^2$$

### 정규화

과적합을 방지하기 위한 L2 정규화 (Ridge):

$$L_{ridge}(\theta) = L(\theta) + \lambda \|\theta\|_2^2 = L(\theta) + \lambda \sum_{j=1}^{p} \theta_j^2$$

L1 정규화 (Lasso):

$$L_{lasso}(\theta) = L(\theta) + \lambda \|\theta\|_1 = L(\theta) + \lambda \sum_{j=1}^{p} |\theta_j|$$

## 5. 신경망에서의 수학

### 활성화 함수

시그모이드 함수:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

소프트맥스 함수 (다중 클래스 분류):

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### Attention 메커니즘

Transformer의 Self-Attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q$, $K$, $V$는 각각 Query, Key, Value 행렬이고, $d_k$는 Key의 차원입니다.

## 코드 예시: 경사하강법 구현

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    """
    Simple gradient descent implementation for linear regression
    Loss function: L(θ) = (1/2n) Σ(y - Xθ)²
    Gradient: ∇L = -(1/n) X^T (y - Xθ)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    for epoch in range(epochs):
        # 예측값 계산: ŷ = Xθ
        y_pred = np.dot(X, theta)

        # 경사 계산: ∇L = -(1/n) X^T (y - ŷ)
        gradient = -(1/n_samples) * np.dot(X.T, (y - y_pred))

        # 파라미터 업데이트: θ = θ - η∇L
        theta = theta - learning_rate * gradient

        if epoch % 100 == 0:
            loss = (1/(2*n_samples)) * np.sum((y - y_pred)**2)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return theta

# 사용 예시
X = np.random.randn(100, 3)
true_theta = np.array([2.0, -1.5, 3.0])
y = np.dot(X, true_theta) + np.random.randn(100) * 0.1

theta_learned = gradient_descent(X, y)
print(f"Learned θ: {theta_learned}")
print(f"True θ: {true_theta}")
```

## 마무리

머신러닝의 수학적 기초를 이해하면:

1. **알고리즘 선택**: 문제에 맞는 적절한 알고리즘을 선택할 수 있습니다
2. **하이퍼파라미터 튜닝**: 학습률, 정규화 강도 등의 의미를 이해하고 조절할 수 있습니다
3. **디버깅**: 모델이 제대로 학습되지 않을 때 원인을 파악할 수 있습니다
4. **새로운 방법 개발**: 기존 방법을 개선하거나 새로운 방법을 개발할 수 있습니다

수학은 머신러닝의 언어입니다. 이 기초를 탄탄히 다지면 더 깊은 이해와 응용이 가능해집니다.
