---
title: "빅데이터 처리: Hadoop과 Spark 비교"
date: "2024-11-15"
excerpt: "대용량 데이터 처리를 위한 Hadoop과 Spark의 차이점과 선택 기준을 알아봅니다."
category: "Big Data"
tags: ["Big Data", "Hadoop", "Spark", "Data Engineering"]
thumbnail: "/assets/images/research/bigdata.jpg"
---

# 빅데이터 처리: Hadoop과 Spark 비교

빅데이터 처리의 양대 산맥인 **Hadoop**과 **Spark**. 각각의 특징과 적합한 사용 사례를 알아봅니다.

## Hadoop 개요

### 탄생 배경
- 2006년 Yahoo에서 개발
- Google의 MapReduce 논문 기반
- 대용량 데이터의 분산 저장 및 처리

### 핵심 구성요소

```
┌─────────────────────────────────────┐
│           Hadoop Ecosystem          │
├─────────────────────────────────────┤
│  Hive  │  Pig  │  HBase  │  Sqoop  │
├─────────────────────────────────────┤
│           MapReduce (처리)           │
├─────────────────────────────────────┤
│             YARN (리소스)            │
├─────────────────────────────────────┤
│            HDFS (저장)               │
└─────────────────────────────────────┘
```

#### HDFS (Hadoop Distributed File System)
- 대용량 파일의 분산 저장
- 블록 단위 복제 (기본 3 replica)
- 장애 허용(Fault Tolerant)

#### MapReduce
- 분산 데이터 처리 프레임워크
- Map → Shuffle → Reduce 단계
- 배치 처리에 적합

## Spark 개요

### 탄생 배경
- 2009년 UC Berkeley AMPLab에서 개발
- Hadoop MapReduce의 한계 극복
- In-memory 처리로 속도 향상

### 핵심 구성요소

```
┌─────────────────────────────────────┐
│           Spark Ecosystem           │
├─────────────────────────────────────┤
│ Spark SQL │ MLlib │ Streaming │Graph│
├─────────────────────────────────────┤
│           Spark Core (RDD)          │
├─────────────────────────────────────┤
│  YARN / Mesos / Kubernetes / 독립   │
├─────────────────────────────────────┤
│      HDFS / S3 / Cassandra 등       │
└─────────────────────────────────────┘
```

#### RDD (Resilient Distributed Dataset)
- 불변의 분산 데이터셋
- Lazy Evaluation
- 장애 복구 가능

#### DataFrame & Dataset
- 구조화된 데이터 처리
- SQL 쿼리 지원
- 최적화된 실행 계획

## 성능 비교

| 항목 | Hadoop MapReduce | Spark |
|------|------------------|-------|
| 처리 속도 | 느림 (디스크 기반) | 빠름 (메모리 기반) |
| 반복 작업 | 매우 느림 | 매우 빠름 (100x) |
| 실시간 처리 | 불가 | 가능 (Streaming) |
| 메모리 사용 | 적음 | 많음 |
| 비용 | 저렴 | 상대적으로 비쌈 |

### 속도 차이 이유

**Hadoop MapReduce:**
```
Map → 디스크 저장 → Shuffle → 디스크 저장 → Reduce
     (I/O 발생)              (I/O 발생)
```

**Spark:**
```
Transformation → Transformation → Action
        (메모리에서 처리, 필요시에만 디스크)
```

## 사용 사례

### Hadoop이 적합한 경우
- 대용량 데이터 저장이 주목적
- 배치 처리 위주
- 메모리 제약이 있는 환경
- 비용에 민감한 경우

### Spark가 적합한 경우
- 반복적인 머신러닝 알고리즘
- 실시간/준실시간 처리
- 인터랙티브 데이터 분석
- 빠른 응답 속도가 필요한 경우

## 코드 비교

### WordCount - Hadoop (Java)
```java
public class WordCount {
    public static class TokenizerMapper
        extends Mapper<Object, Text, Text, IntWritable> {

        public void map(Object key, Text value, Context context)
            throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }
    }
    // ... Reducer 코드 생략
}
```

### WordCount - Spark (Python)
```python
from pyspark import SparkContext

sc = SparkContext("local", "WordCount")
text_file = sc.textFile("data.txt")

counts = text_file.flatMap(lambda line: line.split(" ")) \
                  .map(lambda word: (word, 1)) \
                  .reduceByKey(lambda a, b: a + b)

counts.saveAsTextFile("output")
```

## 함께 사용하기

실제로는 Hadoop과 Spark를 함께 사용하는 경우가 많습니다:

```
데이터 저장: HDFS (Hadoop)
     ↓
데이터 처리: Spark
     ↓
결과 저장: HDFS or Data Warehouse
```

## 2024년 트렌드

1. **Spark의 지배**: 처리 엔진으로는 Spark가 대세
2. **클라우드 서비스**: AWS EMR, Databricks, GCP Dataproc
3. **HDFS 대안**: 클라우드 스토리지 (S3, GCS)
4. **Delta Lake**: ACID 트랜잭션 지원 저장소

## 마무리

- **저장**: HDFS 또는 클라우드 스토리지
- **처리**: Spark (대부분의 경우)
- **선택 기준**: 예산, 처리 속도, 데이터 크기, 팀 역량

둘 다 알면 더 좋지만, 시작한다면 **Spark**부터 배우는 것을 추천합니다.
