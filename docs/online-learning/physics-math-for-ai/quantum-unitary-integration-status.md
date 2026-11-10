# Unitary evolution 두 단원 실제 교재 검증

2026-11-08. 시간 진화의 두 가족을 **도서 → Interactive Book**의 실제 읽기로 연결했다. 현재556개 한영 단원·1668개 안내 실험이며, 원래 입문 범위는303/362주제 검증·59미완료다. 전체51영역509주제·연구 깊이·공용 그림 후속은 계속 남는다.

## 연결한 내용

- [스펙트럼·Schrödinger 시간 진화](http://localhost:4175/book/online/physics-math-for-ai/two-level-spectral-schrodinger-evolution/): 실제 물리 시간·에너지 위상·보존량·전역/상대 위상·ray와 정규화하지 않은 Euler 오차를 비교한다.
- [Hamiltonian pulse·시간 순서](http://localhost:4175/book/online/physics-math-for-ai/piecewise-hamiltonian-pulse-time-ordering/): 정순·역순·상수 평균의 같은 시각 상태, 전환 경계·비가환성·국소 교환자·상쇄·역전파를 비교한다.
- 원고4개와 단원·선수 지식·표지·목차·개념 단계·예측·안내·퀴즈·검색·사이트맵을 등록했다. 각 원고12절·해설6문제, 전체 표시 수식30개·표6개를 포함한다.
- 준비31개·개념6단계·물리 시간 slider/재생·복소 진폭·Bloch3D/2D·확률/fidelity·Euler·pulse 경로·입력·저장·Worker를 연결했다. Online Book 목록에 교재 링크를 추가하지 않았다.

## 확인한 근거

- 전체10261검사와 관련295검사, lint·콘텐츠·정적 build·최종 타입 검사 통과.1794 HTML의 내부 링크·자산·앵커·사이트맵·canonical을 확인했다.
- 실제 읽기 두 단원×한영×명암×390/1600px의16조합과 접근성16조합. 준비31회·개념/예측 각24회·고정 준비104회·안내48회·단원 퀴즈48회를 확인했다.
- 잘못된 정수 초안16회 거절·유효 수정16회 반영·새로고침 저장 복원16회. 물리 시간 Home/End32회와 선택 기록16회·pulse 경로24회, 키보드/실제 터치 pan 각8회를 확인했다. 영어 한글 잔류·페이지 가로 넘침·비유한 SVG 좌표·page error가 없다.
- 실제3D16회·시점·2D/3D 전환·WebGL 상실→2D→3D 재시도4회. 밝은 PC·어두운 모바일 읽기와 Bloch 캡처를 검토했다. Bloch는 상태 공간의 좌표이며 입자 위치가 아니다.
- 생성된 `/_next/static/` Worker 자산으로 읽기 계산을 실행하고 의도한 실패 후 재시도2회를 확인했다. 기본 읽기의 Worker291개가 모두 종료됐다. 별도 클라이언트 단원 이동3회에서 Worker18개 종료·목차 복귀 뒤 canvas0개를 확인했다.
- 같은 생산 Worker 자산에 대해 추가로66개 독립 참조 입력·수치 관측41049개를 Node의 검증된 모형과 대조했다. 생성된 factory의 classic 실행 옵션을 확인하고 동일 방식으로 실행했다. 정확 구간·표시값·해결 상태·구조가 일치하며 모든 포트를 종료했다. log 표시값의 비교 허용 오차는 검증 코드에 명시한다.
- 한영 진입2회·새 표지4개·읽기 링크4개·검색 결과4개와 Online Book 제외2회를 확인했다. 기존554단원·1662안내와51개 영역의 기존 내용·소속·순서를 의미 hash로 보존했다.

수치 모형·독립 참조·전송·교수 화면은 이번 등록에서 바꾸지 않았다. 원고의 집필 상태 안내4문장만 실제 진입 안내로 바꾸고 수학 본문을 유지했다. [계산 기반](quantum-pulse-foundation-status.md), [한영 원고](quantum-unitary-foundation-status.md), [Worker](quantum-unitary-worker-status.md), [교수 화면](quantum-unitary-interface-status.md)의 근거도 보존한다. 소스 화면·독립 수치·생산 자산·실제 읽기의 검증 범위를 구분한다.

## 남은 범위

이번에는 unitary evolution·Schrödinger의 입문 한 주제를 추가 검증했다. 양자역학11주제 중 입문3주제가 검증됐으며 영역 전체·무한 차원 정의역·일반 시간 의존 H·열린계·연구 심화는 미완료다. 다음은 spin·angular momentum이며, density matrix·uncertainty/measurement·tensor product/entanglement·변분·다체·회로·정보와 이후 영역을 순서대로 진행한다.

로컬 작업본이며 공개 배포·commit·push는 수행하지 않았다. 전체 목표는 진행 중이다. 근거는 `.runtime/learning/quantum-unitary-integration/verification.json`, `native-reader.json`, `production-worker.json`, `registry-preservation.json`, `discovery.json`, `client-navigation.json`, 검사 로그와 보존 기록이다.

```bash
npm test
npm run lint
npm run validate:content
npm run build
npm run typecheck
npm run check:export
npx tsx scripts/check-unitary-reader.ts
npx tsx scripts/check-unitary-assets.ts
npx tsx scripts/check-unitary-discovery.ts
npx tsx scripts/check-unitary-client.ts
```
