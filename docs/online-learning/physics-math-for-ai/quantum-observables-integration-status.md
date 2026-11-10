# 관측·연산자: Interactive Book 실제 읽기 통합

2026-11-08. 두 한영 단원을 본문·표지·목차·선수 단원·개념 예측·안내·퀴즈·검색에 연결하고 생성된 Next.js 읽기와 Worker 자산으로 검증했다. 등록554단원·1662안내, 원래302/362입문 검증·60미완료다. 전체51영역509주제·연구 깊이·공용 그림 후속 범위는 계속 진행한다.

## 실제 읽기

- [Hermitian 스펙트럼·Born 확률·기대값](http://localhost:4175/book/online/physics-math-for-ai/hermitian-spectrum-born-expectation/): 관측량·고유공간·확률·평균/분산·기저 변경·조건부 순수 상태와 Bloch3D/2D를 연결한다.
- [축퇴 사영과 순차 관측](http://localhost:4175/book/online/physics-math-for-ai/degenerate-projectors-sequential-observation/): coarse Lüders·rank1 세분화·coherence·조건부/비선택 상태·관측 순서와 결합/주변 확률을 비교한다.

각 단원은 한영12절·풀이6개, 개념3단계·예측3개·안내3개·단원 퀴즈3개를 갖춘다. 진입은 도서 → Interactive Book이며 Online Book 목록에는 해당 교재 카드를 넣지 않는다. 기존 URL과 북마크를 유지하고 읽기에서도 Interactive Book을 활성 표시한다.

## 확인한 근거

- 전체 회귀10026검사·집중199검사와 타입·lint·콘텐츠·정적 빌드 통과. 콘텐츠1789 URL·출판293·프로젝트51개와1792 HTML의 내부 링크·자산·앵커·사이트맵·canonical 검증 통과.
- 실제 읽기 한영×명암×390px/1600px×두 단원16조합·접근성16조합. 준비25회·개념 단계/예측 각24회·고정 준비84회·안내48회·단원 퀴즈48회 확인.
- 잘못된 소수 초안16회 거절·유효 정수 수정16회 반영·새로고침 저장 복원16회. 수동 slider 선택32회, 키보드·실제 터치 스크롤 각8회. 영어 한글 잔류·가로 넘침·비유한 SVG 좌표·페이지 오류 없음.
- 실제3D8회·카메라 시점·2D/3D 전환, WebGL 상실→2D→3D 재시도2회. 밝은 PC·어두운 모바일 읽기와 Bloch 캡처를 검토했다.
- 실제 읽기에서 측정 도구 전환16회·관측 순서 전환16회·조건부 상태 선택16회 확인. 확률0 분기에는 대체 상태를 그리지 않고 미정의를 표시한다.
- 생성된 정적 Worker 자산으로 계산하고 의도한 Worker 실패 후 재시도2회 확인. 기본 읽기 Worker 생성261개·종료261개. 별도 클라이언트 단원 이동3회에서 Worker16개 모두 종료하고 목차 복귀 뒤 canvas 잔존0개 확인.
- 한영 교재 진입2회·표지4개·읽기 링크4개·검색 결과4개, Online Book 새 교재 링크 제외2회 확인.
- 기존552단원·1656안내와 기존 영역 메타데이터·소속·순서를 의미 hash로 보존했다. 수치 모형·전송·독립 참조·교수 화면은 변경하지 않았다. 원고의 집필 상태 머리말만 등록 완료 문구로 갱신했고 본문은 같은 원고를 사용한다.

[수치 기반](quantum-observables-foundation-status.md)의 독립50준비·정확 유리수12028대조·방향성 좌표369대조, [Worker 검증](quantum-observables-worker-status.md)의 실제50준비·30260수치 항목 대조, [교수 화면](quantum-observables-interface-status.md)의 저장·모션·도구/순서·개념 오류 복구·고정/자유 독립성 근거도 보존한다. 이번 실제 읽기 검증과 앞선 소스 화면 검증의 범위를 구분한다.

## 남은 범위

관측·연산자의 입문 한 주제만 추가 검증했다. 양자역학11주제 중 입문2주제가 검증됐으며 영역 전체와 일반 측정 도구·POVM·혼합 상태·무한 차원 정의역·연속 스펙트럼·연구 심화는 미완료다. 다음은 [unitary evolution·Schrödinger](quantum-unitary-authoring-plan.md)의 스펙트럼 전파와 구간별 Hamiltonian 두 가족이다. 이후 spin·density matrix·측정·얽힘·변분·다체·회로·정보와 나머지 영역을 원래 순서대로 진행한다.

로컬 작업본이며 공개 배포·commit·push는 수행하지 않았다. 근거는 `.runtime/learning/quantum-observables-integration/verification.json`, `native-reader.json`, `discovery.json`, `client-navigation.json`, 전체 검사 로그와 보존 기록이다.

```bash
npm test
npm run lint
npm run validate:content
npm run build
npm run typecheck
npm run check:export
npx tsx scripts/check-observable-reader.ts
npx tsx scripts/check-observable-discovery.ts
npx tsx scripts/check-observable-client.ts
```
