---
title: "Pandas로 시작하는 데이터 분석"
date: "2024-12-15"
excerpt: "Python Pandas 라이브러리를 활용한 데이터 분석 기초를 알아봅니다."
category: "Data Science"
tags: ["Python", "Pandas", "Data Analysis", "Tutorial"]
thumbnail: "/assets/images/research/datascience.jpg"
---

# Pandas로 시작하는 데이터 분석

데이터 분석을 시작할 때 가장 먼저 배우게 되는 라이브러리가 바로 **Pandas**입니다. 이 글에서는 Pandas의 기본 사용법을 알아보겠습니다.

## Pandas란?

Pandas는 Python에서 데이터 조작과 분석을 위한 핵심 라이브러리입니다. 엑셀과 유사한 테이블 형태의 데이터를 다루기 쉽게 만들어줍니다.

## 설치 방법

```python
pip install pandas
```

## 기본 사용법

### DataFrame 생성

```python
import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Daegu']
}

df = pd.DataFrame(data)
print(df)
```

### CSV 파일 읽기

```python
df = pd.read_csv('data.csv')
df.head()  # 처음 5행 확인
```

### 기본 통계 확인

```python
df.describe()  # 기술 통계량
df.info()      # 데이터 타입 정보
```

## 데이터 선택과 필터링

### 열 선택

```python
df['name']           # 단일 열
df[['name', 'age']]  # 복수 열
```

### 조건 필터링

```python
df[df['age'] > 30]  # 나이가 30 초과인 행
```

## 마무리

Pandas는 데이터 분석의 기초입니다. 다음 포스트에서는 더 고급 기능들을 다루겠습니다.
