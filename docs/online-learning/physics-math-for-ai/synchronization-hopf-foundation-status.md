# 동기화·임계 동역학: Hopf 가족의 수치·한영 원고 기반

후속: [두 진동자까지 포함한 두 가족의 수치·원고 기반](synchronization-two-families-foundation-status.md)을 준비했다. 아래는 첫 가족 단계의 기록이다.

2026-11-08. [세 가족 집필 계약](synchronization-critical-dynamics-authoring-plan.md)에 따라 첫 가족을 준비했다. **교재 등록·Worker·교수 화면 전 단계**다. 등록545단원·1635안내, 원래298/362입문 검증·64미완료를 유지한다. 전체51영역509주제·연구 깊이·공용 그림 후속 범위도 미완료다.

## 구현한 내용

`src/lib/learning-hopf-amplitude.ts`는 초임계 Stuart–Landau 모형의 제곱 진폭·적분 진폭·감지 않은 위상·Cartesian 좌표·초기값 민감도를 계산한다. 정확한 정수·유리수 부호로 μ<0,μ=0,μ>0과 불변 진폭을 구별한다. μ=0은 대수 감쇠이며 선형 이완시간과 진폭/제곱 진폭 반감시간이 다르다.

정확한 원점의 위상은 미제공이다. μ>0의 원이라도 각속도가0이면 평형점들의 원이며 주기 궤도로 표시하지 않는다. ω=0의 선형 퇴화도 통상적인 Hopf 횡단과 구별한다. 초기에 정확히0인 상태가 스스로 흔들리도록 잡음을 삽입하지 않는다.

방향성256bit 지수·로그와 정확 유리수 척도, 바깥쪽 제곱근 경계, 각도 이분·Taylor 잔차·배각식으로 계산한다. `learning-directed-phase-rotation.ts`는 큰 누적 각도의 계산을 위한 별도 모듈이며 기존 회전 모듈을 수정하지 않았다. 위상 구간에는 전역 Lipschitz 경계도 적용한다. 상태를 원 위에 투영하거나 반지름을 재정규화하지 않는다. 좌표 반지름과 로그 진폭 수지의 별도 구간 잔차를 대수 항등식과 구분한다.

## 독립 검산과 원고

- 생산 코드를 import하지 않는220자리 역진폭2×2행렬 지수로 **35준비·376시간 기록·3920스칼라** 참조를 만들었다. 이전34준비를 유지하고 원고의 임계·전단 예제1개를 추가했다.
- 추가로 각도0·분수·양/음2⁴⁸경계 등7회전 참조를256/512bit에서 확인한다. 독립 참조14값은 위의 모형3920값과 구별한다.
- 임계 양옆±10⁻⁹, 정확한 원점, 일정 진폭, 회전 없는 원, 역회전, 매우 작은 초기 진폭, 강한 감쇠, 약한 포화와10¹⁰rad이상의 누적 위상, 영 시간·분수 입력·기록 밀도 변화를 검사한다.
- [한국어 원고](hopf-amplitude-critical-relaxation-chapter.ko.md)와 [영어 원고](hopf-amplitude-critical-relaxation-chapter.en.md)는 각각14절·10표시 수식·표2개·풀이6개다. 직관·단위·유도·관측량별 회복·위상 축약의 한계·계수 식별을 설명한다. 실제 Markdown/KaTeX, 한영 수식 일치와 독립 참조에 연결한 예제 표를 검사한다.

**49검사·타입·lint를 통과했다.** 검사: `npx tsx --test scripts/learning-hopf-*.test.ts`. 참조 재생성: `python3 scripts/learning/generate-hopf-amplitude-references.py --output /tmp/hopf-reference.json`. 이 유한 사례 검산을 전 입력에 대한 기계 증명이나 실측계 검증이라고 주장하지 않는다. 최종 검사·타입·lint·참조 재생성·기존 홈페이지 보존 근거는 `.runtime/learning/synchronization-foundation/verification.json`에 기록한다.

## 다음 작업

두 진동자의 정확한 잠김·불안정 위상·임계 병목·미끄러짐, 유한 집단의 질서와 연속 분포 극한을 이어서 준비한다. 세 가족의 Worker·실제 시각화·한영 읽기·접근성·정적 출력이 끝나기 전에는 현재 원래 주제를 완료로 기록하지 않는다. 진입 경로는 **도서 → Interactive Book**이며 Online Book 목록에는 추가하지 않는다.
