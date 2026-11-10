# 대칭·게이지: 국소 기준과 Wilson 루프 기반

2026-11-09. `graph-u1-local-frames-wilson-loops`의 한영 원고와 정확 수치 기반을 준비했다. **실제 교재 등록, 학습 화면, Worker, 독립 응답 검증은 아직 남아 있다.** 등록 수치는 587단원·1776안내, 원래 입문 범위는312/362검증·50미완료를 유지한다. 전체51영역509주제와 영역 감사·연구 심화도 미완료다.

## 원고와 교육 범위

언어별16절에 국소 기준의 직관, 링크 방향 약속, 공변과 불변의 유도, 전역/국소와 물리적 중복의 조건, 에너지, 삼각형 손계산, 열린/닫힌/물질로 감싼 관측량, 불완전한 루프 측정, Hermitian 연산자와 Wirtinger 기울기, AI 공변성의 검증 방법, 해설 문제6개를 작성했다.

- 원고: `content/learning/physics-math-for-ai/graph-u1-local-frames-wilson-loops.{ko,en}.md`
- 입력·계산·예제: `src/lib/learning-graph-gauge{,-input,-examples}.ts`
- 독립 참조: `scripts/learning/graph-gauge-reference.py`
- 검사: `scripts/learning-graph-gauge-foundation.test.ts`

원자료는 David Tong의 [Gauge Theory 4장 §4.2.1](https://davidtong.org/pdfs/teaching/gauge-theory/gauge4.pdf)에서 링크 변환·Wilson 곱·물질 관측량의 틀을 확인했다. 원자료의 SU(N) 격자 이론과 여기의 별도 유한 U(1) 그래프 교육 모형을 구분한다.

## 구체적 검증

13예제와36그래프 사례를 Python `Fraction`으로 별도 계산했다. 연산자는 incidence 행렬 조립을 복제하지 않고 **스칼라 이차형식의 polarization**으로 복원했다. 기울기는 **사차 다항식의 중심 차분과 정확한 삼차 오차 보정**으로 대조했다. 역방향 루프도 실제 경로를 역전해서 계산했다.

56검사는 에너지·전체 행렬·물질 기울기·간선/루프 값과 모든 공변 잔차, 자기 루프·평행 간선·고립 노드·분리 성분, 잘못된 경로/입력 소유권, reader 번들의 계산기 분리, 최대12노드32간선8경로와18자리 분수의 정확 항등식을 확인한다. 한영 본문은 실제 loader의 strict KaTeX로 렌더링한다. 타입과 대상 lint를 별도로 확인한다. 증거는 `.runtime/learning/graph-gauge-foundation/`에 둔다.

기본 삼각형에서 함께 변환한 에너지는0, 물질만 변환하면8이다. 루프 곱+i와−i는 에너지가 모두3이지만 다른 불변량이다. 선택된 왕복 루프가1이어도 다른 순환에 플럭스가 남을 수 있다.

## 대칭·게이지 원주제의 남은 순서

1. 이 단원의 독립 응답 검증 → 실제 Worker → 한영 개념·실험·2D/3D 화면 → 실제 reader·생산 자산 검증.
2. `gauge-orbits-tree-fixing-boundary-anchors`: spanning forest 고정, 독립 순환과 연결 궤도, 연결 성분별 잔여 자유도, 외부 기준과 비식별성.
3. `graph-gauss-constraints-physical-phase-space`: 정준 전기 흐름, Gauss 제약, 전하·경계와 허용 게이지 군. 집필 전에 해당 범위의 일차 자료를 추가 검토한다.
4. 세 단원을 원래 요구사항에 대조한 후 입문 원주제 완료 여부를 판단한다. 이후 장이론의 다른8원주제와 나머지4영역41주제, 전체 감사·연구 심화를 계속한다.

현재 실험에는 공간 배치·연속체 극한·시간 발전·게이지 고정·경계 앵커·Gauss 제약·양자 측도·비가환 군이 없다. 선택된 루프가 완전한 분류량이라는 주장도 없다. commit/push·공개 배포는 수행하지 않았다.
