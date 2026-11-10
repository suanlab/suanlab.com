# Spin·angular momentum 두 단원 실제 교재 검증

2026-11-08. 스핀 두 단원을 **도서 → Interactive Book**의 실제 읽기로 연결했다. 현재558개 한영 단원·1674개 안내 실험이며, 원래 입문 범위는304/362주제 검증·58미완료다. 전체51영역509주제·연구 깊이·공용 그림 후속은 계속 남는다.

## 연결한 내용

- [스핀1/2 측정과 SU(2) 회전](http://localhost:4175/book/online/physics-math-for-ai/spin-half-measurement-su2-rotation/): 반각·능동 회전·복소 진폭·Born 확률·평균/분산·Bloch3D/2D·Rodrigues 대응·2π 부호·4π 복원을 비교한다. 각도는 물리 시간과 구별한다.
- [유한 스핀 ladder·Casimir·가중 내적](http://localhost:4175/book/online/physics-math-for-ai/finite-spin-ladders-casimir-metric/): 일반 j의 계수 유도와 유한 j=0…4 실험, 기저 metric·물리 확률·원시 작용 norm·정규화 image·영 image·같은 평균의 다른 상태를 다룬다.
- 한영 원고4개를 수학 본문 변경 없이 등록하고 선수 지식·표지·목차·개념·예측·안내·단원 퀴즈·검색·사이트맵에 연결했다. 언어별 합계48절·표시 수식34개·표6개·해설 문제24개다.
- 예제39개·개념/예측6단계·안내6개·각도/적용 횟수 선택·모션·입력·저장·Worker를 연결했다. Online Book 목록에는 해당 교재 링크를 추가하지 않았다.

## 확인한 근거

- 전체10492검사·lint·콘텐츠·정적 build·최종 타입 검사 통과.1796 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- 실제 읽기 두 단원×한영×명암×390/1600px의16조합과 접근성16조합을 확인했다. 예제39회·개념/예측 각24회·고정 준비92회·안내48회·단원 퀴즈48회다.
- 잘못된 정수 초안16회 거절·유효 수정16회 반영·새로고침 저장 복원16회, 표본 Home/End32회·선택 기록16회를 확인했다. 한 바퀴 추가 비교8회·ladder 방향16회·영 image8회, 키보드/실제 터치 pan 각8회를 포함한다. 영어 한글 잔류·페이지 가로 넘침·비유한 SVG 좌표·page error가 없다.
- 스핀1/2의 실제3D8회·시점·2D/3D 전환·WebGL 상실→2D→3D 재시도2회를 확인했다. 높은 j에는 qubit Bloch 구를 제공하지 않는다. 밝은 데스크톱과 어두운 모바일 캡처를 검토했다.
- 생성된 `/_next/static/` Worker로 읽기 계산을 실행하고 별도 의도적 실패 후 재시도2회를 확인했다. 기본 읽기에서 추적한 Worker309개가 모두 종료됐다. 클라이언트 단원 이동3회에서도 Worker18개 종료·목차 복귀 뒤 canvas0개를 확인했다.
- 같은 생산 Worker에 추가로82개 참조 준비·수치 관측166049개를 보내 Node의 검증된 엔진과 대조했다. 생성 factory의 classic 옵션과 동일하게 실행했다. 정확 구간·표시값·부호·해결 상태·구조가 일치하며 모든 포트를 종료했다. log 표시값의 플랫폼 비교 오차는 검사 코드에 명시한다. 독립 기호/고정밀 수학 참조는 전체 수치 검사에서 확인하며 이 transport 비교와 구별한다.
- 한영 진입2회·새 표지4개·읽기 링크4개·검색 결과4개·Online Book 제외2회를 확인했다. 기존556단원·1668안내와51영역의 기존 내용·소속·순서를 의미 hash로 보존했다.

수치 엔진·독립 참조·프로토콜·교수 화면·준비 원고는 이번 등록에서 변경하지 않았다. [수치/원고](quantum-spin-foundation-status.md), [Worker](quantum-spin-worker-status.md), [교수 화면](quantum-spin-teaching-status.md)의 기록도 보존한다. 소스 화면·독립 수치·생산 자산·실제 읽기의 검증 범위를 구별한다.

## 남은 범위

이번에는 spin·angular momentum의 입문 한 주제를 추가 검증했다. 양자역학11주제 중 입문4주제이며 영역 전체·궤도 파동함수·일반 연산자 정의역·각운동량 결합·연구 심화는 미완료다. 다음은 **density matrix**이며 uncertainty/measurement·tensor product/entanglement·변분·다체·회로·정보와 이후 영역을 순서대로 진행한다.

로컬 작업본이다. 공개 배포·commit·push는 수행하지 않았으며 전체 목표는 진행 중이다. 근거는 `.runtime/learning/quantum-spin-integration/verification.json`, `native-reader.json`, `production-worker.json`, `registry-preservation.json`, `discovery.json`, `client-navigation.json`, 검사 로그와 보존 기록이다.

```bash
npm test
npm run lint
npm run validate:content
npm run build
npm run typecheck
npm run check:export
npx tsx scripts/check-spin-reader.ts
npx tsx scripts/check-spin-assets.ts
npx tsx scripts/check-spin-discovery.ts
npx tsx scripts/check-spin-client.ts
```
