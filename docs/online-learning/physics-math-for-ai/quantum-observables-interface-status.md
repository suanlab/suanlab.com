# 관측·연산자: 개념 비교와 자유 실험 화면

2026-11-08. [Worker 검증](quantum-observables-worker-status.md)에 두 가족의 교수 화면을 연결했다. **실제 Next.js 교재 읽기는 아직 미등록**이다. 등록552단원·1656안내, 원래301/362입문 검증·61미완료를 유지한다. 전체51영역509주제·연구 깊이·공용 그림 후속 범위도 계속 진행한다.

## 학습 화면

| 가족 | 고정 개념 비교 | 자유 실험과 시각화 |
| --- | --- | --- |
| Hermitian 스펙트럼·Born 확률·기대값 | 같은 평균·다른 분포, 수동 기저와 측정 선택, 축퇴·음수 간격·비Hermitian 반례 | 준비13개·정수 칸16개, 확률 막대·고유값·rank·평균 기여·모멘트·조건부 순수 상태의 Bloch3D/2D |
| 축퇴 사영과 순차 관측 | coarse/세분화 도구·coherence, 순서와 표의 축, 고유값 병합·영 분기 | 준비12개·정수 칸26개, 도구/순서 선택·분기 순서도·결합 확률 heatmap·주변/조건부 법칙·밀도행렬·purity |

각 가족은 개념3단계·예측3개·풀이와 같은 비교를 자유 실습으로 옮기는 안내3개를 갖춘다. 고정 비교는 저장된 자유 준비와 독립적으로 계산한다. 유효한 정수 수정은 Enter/blur로 적용하고 잘못된 초안은 이전 준비를 유지한다. 전체 준비 JSON에서도 정수의 원래 숫자 문자열을 검사한다.

C²의 Bloch 그림은 Worker가 제공한 정규화 진폭의 반올림 좌표에서 만든 표시다. 점을 구 표면으로 다시 정규화하지 않는다. 입력과 조건부 분기는 정적 비교이며 상태 붕괴 애니메이션이 아니다. 확률0 분기에서는 조건부 상태를 그리지 않고 미정의를 표시한다. 정확 진폭·확률의 근거는 별도 표에서 확인한다. 수동 기저 변환은 상태 좌표와 관측량 행렬을 함께 바꾸고 평균 잔차를 보여준다.

C³ 그림은 추상 분기의 도식이며 물리적 삼차원 위치가 아니다. coarse Lüders와 rank1 세분화를 다른 도구로 표시한다. 세분화는 A 다음 B 비교이고 역순 버튼은 coarse 도구끼리의 비교다. 역순 원본의 행 B·열 A를 전치해 비교 heatmap의 행 A·열 B와 맞춘다. 두 관측량이 가환해도 임의 세분화가 후속 B를 보존한다고 설명하지 않는다.

행/단계 강조는 그림을 읽는 안내다. 물리 시간·장치 궤적·양자 상태의 연속 진화를 만들지 않는다. 수동 slider를 제공하고 동작 줄이기·사이트 모션 정지·화면 밖에서 자동 진행을 멈춘다. 버튼 간격·줄바꿈·반응형 스크롤과 색은 기존 홈페이지 스타일에 맞췄다.

## 확인한 근거

- 기존 수치·집필·Worker187검사와 새 교수 화면·SSR·의존성7검사, 합계194검사 통과. 타입·전체 lint 통과.
- 한영×명암×320px/1600px×두 가족의16기본 화면·접근성16조합. 영어 한글 잔류·가로 넘침·페이지 오류 없음.
- 준비50회 선택, 개념 단계·예측 각12회, 잘못된 초안20회 거절, 저장 복원16회. Worker 실패 후 재시도·손상 저장값·저장 쓰기 실패 각4회 확인.
- 실제3D8회,2D 전환과3D 복귀, WebGL 상실 후2D 대체·3D 재시도2회 확인. 기본 검사 Worker256개 생성·256개 종료.
- 추가4흐름에서 안내12회·고정 준비12회, 고정/자유 독립성24회 확인. 조건부 상태 선택4회, 측정 도구 전환2회·관측 순서 전환4회 확인.
- 모바일 실제 터치·키보드 스크롤 각2회. 모션 진행·화면 밖 정지·사이트 일시정지 각4회 확인. 개념 Worker 실패 후 재시도4회. 추가 Worker48개 모두 종료하고 unmount 후 canvas 잔존0개.
- 전체 화면 의존성 그래프에 두 수치 엔진·정확 행렬 기반·정규화 core가 포함되지 않음을 확인했다. 한영·명암·좁은 화면 캡처를 검토했다.

브라우저 검사는 기존 로컬 서버와 같은 출처에서 현재 소스의 컴포넌트·실제 공급자·실제 Worker를 bundle하여 실행했다. **새 Next.js 읽기나 생성된 정적 Worker 자산의 검증은 아니다.** 수치 모형·독립 참조·한영 원고·기존 등록 교재와 정적 출력은 보존했다.

## 다음 순서

한영 본문을 Interactive Book에 등록하고 표지·목차·선수 단원·퀴즈·검색을 연결한다. 실제 한영·명암·모바일·키보드/터치·접근성·저장·오류 복구와 정적 출력을 검증한 뒤에만 관측·연산자 입문 완료를 기록한다. Online Book 목록에는 새 교재 링크를 추가하지 않는다. 그 다음 원래 주제는 unitary evolution·Schrödinger이며 전체 연구 심화 범위도 유지한다.

새 빌드·공개 배포·commit·push는 수행하지 않았다. 근거는 `.runtime/learning/quantum-observables-interface/verification.json`, `browser-verification.json`, `interaction-verification.json`, `browser-dependency-graph.json`과 화면·검사 로그다.

```bash
npx tsx --test scripts/learning-observable-interface.test.ts scripts/learning-observable-worker.test.ts scripts/learning-quantum-observable*.test.ts scripts/learning-quantum-hilbert.test.ts
npx tsx scripts/check-observable-interface.ts
npx tsx scripts/check-observable-interactions.ts
npm run typecheck
npm run lint
```
