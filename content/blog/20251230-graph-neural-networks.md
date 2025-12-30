---
title: "Graph Neural Networks 기초"
date: "2025-12-30"
excerpt: "최근 들어 인공지능(AI)과 머신러닝(ML)이 다양한 분야에서 혁신을 이루고 있습니다. 그중에서도 그래프 뉴럴 네트워크(Graph Neural Networks, GNN)는 복잡한 구조적 데이터를 효율적으로 처리할 수 있는 강력한 도구로 주목받고 있습니다. 그래프 데이터는 소셜 네트워크, 추천 시스템, 분자 구조 등 다양한 실세계 문제에서 자연스럽게 발생하며..."
category: "Deep Learning"
tags: []
thumbnail: "/assets/images/blog/20251230-graph-neural-networks.png"
---

# Graph Neural Networks 기초

## 도입부

최근 들어 인공지능(AI)과 머신러닝(ML)이 다양한 분야에서 혁신을 이루고 있습니다. 그중에서도 그래프 뉴럴 네트워크(Graph Neural Networks, GNN)는 복잡한 구조적 데이터를 효율적으로 처리할 수 있는 강력한 도구로 주목받고 있습니다. 그래프 데이터는 소셜 네트워크, 추천 시스템, 분자 구조 등 다양한 실세계 문제에서 자연스럽게 발생하며, 이러한 데이터를 효과적으로 처리하는 것은 매우 중요합니다. 이번 포스트에서는 GNN의 기본 개념을 이해하고 간단한 Python 코드 예제를 통해 실습해보도록 하겠습니다.

## 본문

### 그래프 데이터란?

그래프는 정점(vertex)와 간선(edge)으로 구성된 데이터 구조입니다. 정점은 개별 객체를 나타내고, 간선은 정점 간의 관계를 나타냅니다. 예를 들어, 소셜 네트워크에서 각 개인은 정점으로 표현되고, 친구 관계는 간선으로 표현될 수 있습니다.

### GNN의 기본 개념

그래프 뉴럴 네트워크는 이러한 그래프 구조 데이터를 처리하기 위해 설계된 신경망 모델입니다. GNN의 주요 특징은 그래프의 구조적 정보를 신경망 학습에 활용한다는 점입니다. GNN은 주로 다음과 같은 두 가지 단계로 구성됩니다.

1. **메시지 전달(Message Passing)**: 각 정점은 이웃 정점으로부터 정보를 수집합니다.
2. **갱신(Update)**: 수집한 정보를 바탕으로 정점의 특징 벡터를 갱신합니다.

이 과정을 여러 번 반복하며 그래프의 구조적 정보와 정점의 특징을 학습합니다.

### GNN의 간단한 구현

이제 Python을 사용하여 간단한 GNN을 구현해보겠습니다. 여기서는 PyTorch Geometric 라이브러리를 사용할 것입니다. PyTorch Geometric은 그래프 데이터 처리를 위한 다양한 도구와 기능을 제공합니다.

#### 환경 설정

먼저 필요한 패키지를 설치합니다.

```bash
pip install torch
pip install torch-geometric
```

#### 데이터 준비

그래프 데이터를 준비합니다. 여기서는 간단한 예제로 Cora 데이터셋을 사용하겠습니다. Cora는 논문 간의 인용 관계를 나타내는 그래프 데이터셋입니다.

```python
from torch_geometric.datasets import Planetoid

# Cora 데이터셋 로드
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# 데이터셋 정보 출력
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')
```

#### GNN 모델 정의

GNN 모델을 정의합니다. 여기서는 간단한 그래프 컨볼루션 네트워크(Graph Convolutional Network, GCN)를 구현해보겠습니다.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 모델 초기화
model = GCN(hidden_channels=16)
```

#### 모델 학습

모델을 학습시킵니다. 여기서는 간단한 학습 루프를 구현합니다.

```python
import torch.optim as optim

# 데이터와 모델 준비
data = dataset[0]
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 학습 루프
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

# 평가
def test():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')
```

위의 코드에서 우리는 Cora 데이터셋을 사용하여 GCN 모델을 학습하고, 테스트 세트에서 모델의 정확도를 평가합니다.

## 결론

이번 포스트에서는 그래프 뉴럴 네트워크의 기본 개념과 간단한 Python 구현을 통해 GNN의 동작 방식을 살펴보았습니다. GNN은 그래프 형태의 데이터를 처리하는 데 매우 유용하며, 다양한 분야에서 활용될 수 있습니다. 추가로 학습을 원하는 독자들은 PyTorch Geometric의 공식 문서와 다양한 튜토리얼을 참고하시기 바랍니다. GNN은 여전히 활발히 연구되고 있는 분야로, 앞으로도 많은 발전이 기대됩니다.

### 참고 자료

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/)
- [Deep Learning on Graphs: A Survey](https://arxiv.org/abs/1812.08434)
- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1901.00596)