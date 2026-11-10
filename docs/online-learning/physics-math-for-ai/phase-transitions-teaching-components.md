# 상전이 교수 화면의 독립 검증

> 현재 단계(2026-11-07): 세 단원의 실제 교재 등록과 검증을 완료했다. 아래는 해당 준비 단계의 기록이며, 최신 결과는 [실제 읽기 화면 연결 검증](phase-transitions-integration-verification.md)을 따른다.


2026-11-07. 세 가족의 고정 개념·실험·퀴즈와 시각화를 작성하고 독립 미리보기에서 검증했다. **실제 교재 등록·읽기 경로·Next 정적 worker 자산 검증은 다음 단계다.** 현재521등록단원·1563안내, 원래286/362입문 검증·76미완료 수를 유지한다.

## 학습 구성

`learning-phase-transitions.ts`와 `learning-phase-transitions-concepts.ts`에 한영 자료를 작성했다. 세 가족에 각각6단계, 안내3개, 최종 질문3개를 제공한다. 총18개 고정 예측 질문은 각3선택지·해설을 가지며 최종 퀴즈는9문제다. 비교 예제는 유한계9개, 변분계9개, Landau12개로 총30개다. 안내9개가 모든 예제를 빠짐없이 다룬다.

- 유한계: 미시 다중도→가중치→부호 평균·절댓값→N Var(m)→거시·조건부·미시 엔트로피→자기 항 관례. 막대 그림과 페이지별 상태 표를 연결한다.
- 변분계: 제약 엔트로피 지형→모든 정지해→인접 장의 근 수→사차 임계 최소→작은 편향의 전역 선택→극한 순서. 퍼텐셜·도함수, 근 확대·역할·장벽을 비교한다.
- Landau: 실수 전체의 정의역→삼중·오중근→세 최소 공존→영장·비영 장 접힘→기울어진 사차 최소→미해결 전역 후보. 모든 근과 중복도·고차 도함수를 보존한다.

## 화면과 계산 연결

`phase-transitions-lab.tsx`, `phase-transitions-concept.tsx`, `phase-transitions-view.tsx`, `use-phase-transitions-series.ts`를 추가했다. 기존 색 변수·명암·표·버튼 스타일을 사용한다. 실험의 저장 상태와 고정 개념 준비는 독립적이며, 모든 정확 계산은 기존 전용 worker를 사용한다. 새 입력의 첫 렌더에서 이전 결과를 숨긴다.

입력은 유효 범위를 확인하고 비영 underflow를0으로 바꾸지 않는다. 초기화와 예제 선택은 잘못 입력한 초안도 지운다. 자동 순회는 물리 시간 진화가 아니며 모션 설정·화면 이탈에 따라 멈춘다. 수동 단계는 계속 사용할 수 있다.

곡선은 반올림된 표본 그림이다. 표시 식의 취소 오차보다 작은 변화를 곡선처럼 그리지 않고 **그림 정밀도 미해결**이라고 표시한다. 이 경계는 근·임계성 판정에 쓰지 않는다. 근의 좌표는 충분한 자릿수로 표시하고 역할·중복도는 인증된 계산에서 읽는다. 미제공 계산값과 접힘의 단일값 미분 부재도 구별한다.

## 실제 확인한 범위

- 새43검사: 모든 예제·고정 준비의 수치 결과·한영 SSR, 안내 범위, 데이터 소유권, 실제 등록 수, 숫자 파싱, 가까운 근의 표시, 계산 누락·그림 잡음의 구별을 확인했다. lint·타입 검사도 통과했다.
- 한영×명암×320/1440px의24조합:144개 개념 단계·432개 고정 선택지·72개 안내·240개 예제 선택·216개 퀴즈 선택·24회 새로고침 복원을 확인했다. worker616개 생성·616개 종료,0개 잔존·페이지 오류0개다.
- 최종 표시48상태: 미제공 값·접힘·상한 후보·가까운 근의 그림 미해결·유한 상태 페이지 이동을 검사했다. 도함수 전환·확대·근 상세도 조작했다.72회의 접근성 검사에서 위반이 없고, 영어 한글 잔존·가로 넘침이 없다. 축 숫자 잘림은 스크린샷으로 확인 후 수정했다.
- 추가 생명주기4검사: 세 가족의 실제 계산 시작 후 중도 종료·재시도·실험/개념 worker 자산 실패 복구, 정상 모션 자동 순회·화면 밖 중지·전체 모션 중지·중지 중 수동 단계를 확인했다.

이 검사는 실제 구성요소·기존 export CSS·별도 브라우저 worker를 사용하는 로컬 harness다. 가벼운 pathname stub을 사용하며 Next가 생성하는 실제 worker URL과 교재 등록 경로를 대신 검증하지 않는다. bundle 의존성에서도 정확 수치 모델은 main에서 제외되고 worker에만 들어감을 확인했다. 이전4130소스·4264정적 출력은 그대로다.

## 재현과 다음 순서

저장소 루트에서 기존 `out/`가 필요하다. 없으면 먼저 `npm run build`한다.

```bash
node scripts/learning/prepare-phase-teaching-preview.mjs
python3 .runtime/learning/phase-transitions-teaching/preview/server.py
# 별도 터미널
npx tsx scripts/check-phase-transitions-teaching.ts
npx tsx scripts/check-phase-transitions-lifecycle.ts
npx tsx scripts/check-phase-transitions-view-states.ts
npx tsx --test scripts/learning-phase-transitions-teaching.test.ts
```

증거는 `.runtime/learning/phase-transitions-teaching/verification.json`에 둔다. 다음은 세 단원·안내의 실제 등록, 개념·실험 dispatcher와 표지 연결, 실제 읽기의 저장·취소·재시도, 전체 export와 링크·worker 번들 검증이다. 진입은 **도서 → Interactive Book**으로 유지하며 Online Book 본문에는 링크를 넣지 않는다. 전체 교육과정과 연구 심화는 계속 진행 중이다.
