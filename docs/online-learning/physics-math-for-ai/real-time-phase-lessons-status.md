# 실시간 위상: 예측 실험과 해설 문제

2026-11-11. 결과 시각화에 이어 원고의 **안내 실험 6개·해설 문제 8개**를 한영 학습 화면에 연결했다. 이 단원의 검증 요건은 **17/23에서 19/23**으로 진행한다. 전체 입력·표시·학습 저장과 접근성 통합, 실제 reader와 생산 자산, 기존 콘텐츠 보존, 단원 범위 감사의 네 요건은 남는다. 신규 단원은 아직 미등록이며 공개 홈페이지에서 제공된다고 주장하지 않는다.

## 학습 흐름

안내를 선택하고 먼저 예측을 적는다. 예측은 안내별로 유지되며 최대 2000자다. 안내 선택·예측 작성·해설 열기·문제 풀기·문제 초기화는 계산을 시작하지 않는다. **적용** 버튼만 해당 예제 입력으로 바꾸어 계산과 독립 검산을 요청한다. 문제는 정답과 오답 모두에 설명을 제공한다. 언어 전환은 학습 선택과 사용자가 작성한 예측을 보존한다.

기존 32예제를 유지하고 안내에 필요한 17개의 명시 준비를 추가했다. 여섯 안내의 버튼 참조는 총18개다.

1. 양·음 곡률: 같은 determinant 크기에서도 정상 경로·작용·음의 모드 위상을 구분한다.
2. 영 pivot: 가역 2×2행렬과 실제 특이행렬을 비교한다.
3. 정확 초점: 호환/비호환 양끝과 정규화된 parity packet을 비교한다.
4. 두 폭 족: 고정 σ²와 σ²=ℏr의 위치·운동량 분산과 단위를 구별한다.
5. chirp 초점: 위치 분산이 사라져도 운동량 분산이 남는 반례를 확인한다.
6. fold 합류: 두 실수 정상점, 퇴화한 원점, 실수 정상점이 없는 비영 Airy 진폭을 비교한다.

여덟 문제는 위상 상쇄·정상성과 최소성·determinant와 branch·영 pivot·분포 초점·고전극한 족·위치/위상공간·실수 정상점 부재를 다룬다. 추가 수치 계산을 reader에 import하지 않는다. 학습 상태 parser는 자체 소유 불변 배열을 만들고 잘못된 값·접근자·추가 필드를 거부한다. 이 parser와 화면 내 상태 유지가 아직 새로고침 저장의 완료를 뜻하지는 않는다.

## 검증

- Node20.20.2/24.13.0 각22검사: 17준비 모두 계산·독립 전체 응답 gate 통과, 별도 Python Fraction 손계산과 공분산 대조, 한영 SSR·상태 검증·의존성 분리. 기존 개념 회귀4검사도 통과했다.
- Chromium 한영 × 명암 ×320/1440px 8조합: 48안내·192선택지 피드백·39실제 예제 적용, 키보드 조작·언어 전환·예측 유지·문제 초기화·접근성8회. 안내 전체 버튼 참조18개를 실제 적용했다. 계산/검산 port94개 생성·94개 종료·활성0이다.
- 기존 홈페이지의 CSS·버튼·HSL 색을 사용한다. 모바일에서 문제 구분이 약한 부분은 기존 information-lab 테두리와 여백으로 보정하고, 보정한 최종 화면도 8조합·접근성8회로 재검사했다. 원고 내용과 계산 값은 바꾸지 않았다.
- 타입 검사·lint 통과. authoring esbuild 화면 검증이며 실제 Next 생산 자산·reader 검증은 후속이다. 기존 out·다른 preview는 보존하고 이번 서버만 종료한다.

## 다음 순서와 전체 범위

입력 form 및 모든 표시·학습 상태 저장 → 실제 Interactive Book reader와 생산 Worker 자산 → 기존600단원 보존과 범위 감사 순서다. 등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362검증·**47미완료**를 유지한다. 과거의48개 목록에서 한 항목이 이미 완료된 결과다. 전체51영역509주제와 연구 심화가 완료된 것은 아니다.

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-phase-lessons.test.ts scripts/learning-real-time-phase-guide-models.test.ts
node --import tsx scripts/preview-real-time-phase-lessons.ts
node --import tsx scripts/check-real-time-phase-lessons-native.ts
REAL_TIME_STYLE_ONLY=1 node --import tsx scripts/check-real-time-phase-lessons-native.ts
```

증거: `.runtime/learning/real-time-phase-lessons/verification.json`. 원격 반영은 먼저7e946c0을 push했고, 이 검증 기록과 스타일 보정은 후속 commit으로 반영한다. push 성공과 공개 배포 성공을 구분한다.
