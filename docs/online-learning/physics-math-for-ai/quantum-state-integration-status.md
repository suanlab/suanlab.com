# 상태·Hilbert 공간: Interactive Book 실제 읽기 통합

2026-11-08. 두 한영 단원을 표지·목차·선수 단원·안내·개념 예측·퀴즈·검색에 연결하고 생성된 Next.js 읽기와 Worker 자산으로 검증했다. 등록552단원·1656안내, 원래301/362입문 주제 검증·61미완료다. 전체51영역509주제, 연구 깊이와 공용 그림 후속 범위는 계속 진행한다.

## 실제 읽기

- [복소 상태·ray·기저 진폭](http://localhost:4175/book/online/physics-math-for-ai/complex-state-rays-basis-amplitudes/): 복소 내적·정규화·전역/상대 위상·기저와 Gram 기하. 실제 계산 좌표를 Bloch3D/2D, 복소 진폭과 기저 확률로 비교한다.
- [파동함수·L² 사영·완비성](http://localhost:4175/book/online/physics-math-for-ai/wavefunction-l2-projection-completeness/): L² 동치류·밀도와 구간 확률·유한 Legendre 사영. 포착 norm과 재정규화를 분리하고 정확 적분·계수·오차를 표시한다.

각 단원은 한영14절·풀이6개, 개념3단계·예측3개·안내3개·퀴즈3개를 갖춘다. 진입은 **도서 → Interactive Book**이다. Online Book 목록에는 해당 교재 카드를 넣지 않으며 기존 URL과 북마크는 유지한다. 읽기와 목차에서도 Interactive Book이 활성 메뉴다.

## 검증

- 전체 회귀9879검사·집중129검사, 타입·lint·콘텐츠 검사 통과. 콘텐츠1787 URL·출판293·프로젝트51개 검증. 정적 빌드와1790 HTML의 링크·자산·앵커·사이트맵·canonical 검증 통과.
- 새 읽기 한영×명암×390px/1600px×두 단원16조합·접근성16조합 통과. 준비21회, 개념 단계·예측 각24회, 고정 준비92회, 안내48회, 단원 퀴즈48회 확인.
- 잘못된 소수 입력16회 거절·올바른 정수 수정16회 반영, 새로고침 저장 복원16회. 모바일 키보드·실제 터치 스크롤 각8회. 영어 한글 잔류·가로 넘침·비유한 SVG 좌표·페이지 오류 없음.
- 실제3D8회, WebGL 상실→2D 대체→3D 재시도2회. 수동 시점·2D/3D 전환, 동작 줄이기에서 스캔 비활성화 확인. 밝은 PC·어두운 모바일 화면과 Bloch 캡처를 검토했다.
- 생성된 정적 Worker 자산으로 계산하고 의도한 Worker 실패 후 재시도2회 확인. 기본 읽기 검사 Worker 생성236개·종료236개. 별도 클라이언트 단원 이동3회에서 Worker18개 모두 종료하고 목차로 돌아온 뒤 canvas 잔존0개 확인.
- 한영 교재 진입2회·새 표지4개·읽기 링크4개·검색 결과4개 확인, Online Book의 새 교재 링크 제외2회 확인.
- 기존550단원·1650안내와 영역 메타데이터·기존 소속·상대 순서 보존. 과거 Master/Fokker/coarse 보존 검사는 기존 값과 소속·순서를 계속 검사하면서 이후 단원 추가를 허용하도록 수정했다.

수치 기반의 독립46준비·정확 유리수7103대조·방향성 좌표856대조, 실제 Worker46입력과 앞선 화면·저장·모션 복구 근거는 [수치 기반](quantum-state-foundation-status.md)과 [화면 검증](quantum-state-interface-status.md)에 보존했다. 이번 등록에서 수치 엔진과 독립 참조는 변경하지 않았다.

## 남은 범위

상태·Hilbert 공간의 **입문 한 주제**만 검증했다. 양자역학11주제 전체와 일반 Hilbert 공간·무한 차원 연산자 정의역·스펙트럼·rigged Hilbert space·혼합 상태·연구 깊이는 완료되지 않았다. 다음은 [관측·연산자 두 가족](quantum-observables-authoring-plan.md)의 정확 행렬 모형·독립 참조·한영 원고다. 이후 시간 진화·spin·density matrix·측정·얽힘·변분·다체·회로·정보와 나머지 영역을 원래 순서대로 진행한다.

로컬 작업본이며 공개 배포·commit·push는 수행하지 않았다. 근거: `.runtime/learning/hilbert-integration/verification.json`, `native-reader.json`, `discovery.json`, `client-navigation.json`과 전체 검사 로그.

```bash
npm test
npm run lint
npm run validate:content
npm run build
npm run typecheck
npm run check:export
npx tsx scripts/check-hilbert-reader.ts
npx tsx scripts/check-hilbert-discovery.ts
npx tsx scripts/check-hilbert-client.ts
```
