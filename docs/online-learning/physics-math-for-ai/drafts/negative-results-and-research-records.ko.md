## 1. 각 실행에 안정적인 식별자 부여하기

질문, 코드 버전, 데이터 상태, 분할, 설정, 시드, 환경, 결과물 위치를 기록합니다. 지표 정의와 자원 측정 경계를 명시합니다. 같은 식별자로 설정, 원시 출력, 분석, 해석을 연결합니다. 정리된 메타데이터는 과학 결과의 탐색과 재사용을 돕습니다. [Wilkinson 외, FAIR 원칙](https://doi.org/10.1038/sdata.2016.18)

## 2. 가설이 맞지 않은 결과와 실행 오류 구분하기

완료한 평가에서 개선이 없거나 비용이 더 크거나 특정 조건에서 성능이 나쁠 수 있습니다. 이는 과학적 결과입니다. 중단된 실행은 부분 결과를 다루는 규칙을 미리 정하지 않았다면 유효한 최종 점수가 없습니다. 예외, 발생 단계, 남은 결과물을 기록하고 구현을 진단한 뒤 모형 품질을 해석합니다.

## 3. 선택 보고가 요약값에 주는 영향 보기

완료한 검증 실행 10개의 손실이 0.42, 0.55, 0.50, 0.60, 0.48, 0.57, 0.45, 0.65, 0.53, 0.59라고 가정합니다. 최고 실행의 손실은 0.42이고 전체 평균은 0.534입니다. 두 수치 중 어느 것을 보고하든 정확히 이름 붙이고 선택 규칙을 밝힙니다. 최고 실행을 고른 뒤 독립 최종 데이터에서 평가하는 것은 별도의 질문입니다.

## 4. 검증을 통과하지 못한 설명도 보존하기

가설마다 예측, 개입, 결과, 해석을 기록합니다. 특정 예산이나 데이터 조건의 부정적 결과는 그 주장의 범위를 제한합니다. 가정을 바꾼 새 실험을 제안할 수 있습니다. 원래 결과를 수정 가설과 연결하면 분석의 변화 과정을 확인할 수 있습니다.

## 5. 기록과 수치 요약 검산하기

```python
import json
from statistics import mean
from math import isclose

losses = [.42, .55, .50, .60, .48, .57, .45, .65, .53, .59]
record = {
    "run_id": "synthetic-run-01",
    "question": "does the intervention improve held-out episode error?",
    "split_id": "example-machine-split",
    "seed": 2026,
    "status": "completed",
    "metric": {"name": "MAE", "unit": "K", "value": losses[0]},
    "selection": "all completed validation runs retained",
}
restored = json.loads(json.dumps(record))
assert restored == record
assert isclose(mean(losses), .534)
print("best:", min(losses), "mean:", mean(losses), "count:", len(losses))
```

## 6. 다른 연구자가 판단을 복원할 수 있게 쓰기

근거, 경쟁 설명, 확인한 한계, 다음에 구별할 관측을 요약합니다. 각 주장을 해당 실행이나 증명에 연결합니다. 음성 대조와 불리한 조건도 포함합니다. 보존할 연구 기록과 논문에서 전달할 간결한 이야기를 구분하되 같은 결과물로 추적할 수 있게 합니다.
