# 국소 질서변수: 수학 기반과 한영 원고 검증

2026-11-11. `microscopic-blocks-and-local-order-parameters`는 **등록 전 준비 중**이다. 602개 등록 단원, 검증된 입문 원주제316개, 미완료46개는 변하지 않는다. 새 원고 두 개는 활성 미등록 초안 한 단원으로 센다.

## 구현과 독립 검산

- [수학 계약](block-order-model-contract.md): 활동도·counting measure·주기 edge·제한 자유에너지·국소 근사의 의미를 고정한다.
- `src/lib/learning-block-order.ts`: Gray-code로 스핀 하나씩 바꾸고 해당 incident edge와 블록 개수를 갱신한다. 전체 프로필의 불일치 다항식, 정확한 확률·자화 공분산·전체 합 분포, 선택 프로필의 위치별 조건부 평균과 국소 기준 잔차를 계산한다.
- `scripts/learning/verify-block-order-foundation.py`: 짧은 축을 행 상태로 사용한 정수 다항식 transfer로 상태 개수를 독립 계산한다. Fraction으로 가중치를 평가하고 선택한 프로필의 미시 평균은 별도 직접 열거로 확인한다. TypeScript의 Gray 순회나 incident 갱신 함수를 공유하지 않는다.

24개 모형에서 **프로필2,212개·영이 아닌 n(k,d) 계수9,549개**와 모든 반환 확률·모멘트·선택 미시 예를 대조했다. 4×4, 8×2/2×8 전치, 길이2 주기 edge, 블록 전체 주기 wrap, 홀수 부피, 16스핀과 활동도 분자·분모 경계를 포함한다. 선택 예제는 작은 numeric mask 순이며 무작위 표본이 아니다.

열역학 Z 대신 상대 Z만 맞추는 오류를 피하려고 절대 Z²와 log Z의 baseline 피연산자도 반환한다. 로그·반정수 거듭제곱 전체가 유리수라는 주장은 없다. 모든 지원 입력은 유한한 양의 가중치를 가진다. x=0인 영온 극한이나 임계 극한은 계산하지 않는다.

## 설명용 반례

4스핀의 평균0 두 블록에서 미시 확률은 1/6,1/3,1/3,1/6이고 선택 프로필 확률은1/9다. 한 블록의 평균1/2에서도 경계 스핀 평균은1/3이고 내부는2/3이다. 따라서 균일 퇴화도나 블록 평균곱을 정확한 제한 합으로 대체할 수 없다.

6스핀 열린 세 블록에서 중앙 자화0을 고정하면 비이웃 끝 블록의 cross ratio는16/25다. 이는 그 거친 그래프의 양의 최근접 pair-factorization을 배제한다. 같은 세 블록을 주기 연결하면 끝 쌍은 이웃이므로 같은 논증을 적용하지 않는다. 전체 자화를 추가 고정하지 않았다는 조건도 출력에 명시한다.

평균곱 기준과 정확 합의 잔차는 선택 프로필에서 `ln(9/8)/2`다. 포화 프로필의 잔차0과 달라 공통 정규화 인자만의 차이가 아니다. 이 값은 정규화한 분포의 KL이 아니며 일반적인 부호 정리도 주장하지 않는다.

## 상세 원고

[한국어](drafts/microscopic-blocks-and-local-order-parameters.ko.md), [영어](drafts/microscopic-blocks-and-local-order-parameters.en.md) 각각 20절, display 수식9개, 표2개, 해설 문제8개를 작성했다. 동네 평균 비유 → 미시 가중치 → 손계산 → 숨긴 엔트로피 → 경계 평균 → 국소 기준 → 조건부 독립 반례 → Python → 시각화 실험 설계 → AI 연결 순서다.

실제 `getLearningContent`를 임시 콘텐츠 루트에서 실행해 수식 오류·중복 anchor·영어 한글 잔류가 없는지 확인했다. 한영 수식·Python 코드는 일치하며, 원고에서 추출한 Python 자체를 실행해1/9·7/27·16/25를 확인했다. 무결합·반강자성 후속 계산도 실행했다. 새 Colab 강좌를 등록한 것은 아니다.

## 재현과 남은 절차

```bash
node --import tsx --test scripts/learning-block-order-foundation.test.ts scripts/learning-block-order-manuscripts.test.ts
```

위 검사는 8개 테스트다. 독립 참고값은 실행 때 Python에서 새로 생성하며 `.runtime/learning/block-order-foundation/`에 비교 결과·원고 렌더링·예제 실행 기록을 남긴다. 참고값 JSON만 생성하거나 플래그를 바꾼 것으로 완료하지 않는다.

Node20·24에서 각각8개 테스트가 통과했다. 신규 TypeScript 세 파일의 ESLint와 저장소 전체 TypeScript 검사도 통과했다. 앞선 `f04715a`의 전체 `npm run check`는 별도 고정 worktree에서 계속 실행 중이며, 그 결과를 이 새 준비 코드의 전체 CI 통과로 간주하지 않는다.

독립 브라우저 검산기와 bounded 응답 codec, Worker 요청 attribution·취소·오류 처리, 기존 홈페이지 테마의 실제 시각화·조작·해설 문제, 상태 복원, 실제 reader의 한영·두 테마·모바일·접근성 검사가 남았다. 이 단계에서는 생산 브라우저 계산·단원 등록·공개 배포 완료를 주장하지 않는다. 통계장론 원주제와 나머지 교육과정 목표는 계속 진행 중이다.
