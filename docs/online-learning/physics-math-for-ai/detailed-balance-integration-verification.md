# 상세평형 세 단원 입문 통합 검증

2026-11-07. 정상 순환·Hastings 보정·두 상태 가역성/혼합의 세 단원을 실제 정적 Next 교재에 연결하고 검증했다. **로컬530단원·1,590안내, 원래289/362주제 입문 검증·73미완료**다. 전체51영역509주제·연구 심화는 계속 미완료이며 공개 배포·commit·push는 수행하지 않았다.

## 실제 교재 연결

다음 세 단원의 한영 본문·개념·예측·실습·표지·최종 퀴즈와 선수 링크를 등록했다.

| 단원 ID | 내용 |
| --- | --- |
| `stationary-flux-three-state-cycle` | 정상분포에서도 남는 순환 흐름, 쌍별 상세평형, 시간 반전, 주기·항등과 수축 상한 |
| `metropolis-hastings-proposal-balance` | 제안 비대칭 보정, 거부 체류, 우연한 목표 보존, 실제 정상 법칙과 이동만 남긴 점프 기록 |
| `reversible-two-state-periodicity-mixing` | 부호 있는 오차와 거리, 가역적 교대, 항등·흡수, Cesaro 평균과 최악 초기값 혼합 기준 |

주소는 기존 `/book/online/physics-math-for-ai/{id}/` 규칙을 유지한다. 도서 → Interactive Book에서 교재와 세 표지에 접근하며, 메뉴 활성 표시·breadcrumb·검색·사이트맵도 이를 따른다. Online Book 본문에 새 교재 진입 링크를 추가하지 않았다. 기존 홈페이지의 색·명암·서체·표·모션 설정을 유지했다.

`src/data/interactive-book.ts`와 열역학 domain의 `unitIds`에 세 단원·9안내를 추가했다. `BookArt`, `ConceptWalkthrough`, `InteractiveBookReader`의 전용 분기를 연결했다. 본문 로더·기존 퀴즈·동적 경로 생성·검색·사이트맵은 기존 방식을 사용한다. 새로운 서버 기능이나 API가 필요하지 않다.

## 보존과 전체 검사

변경 전 `scripts/fixtures/pre-detailed-balance-registry-hashes.json`에527단원·안내·영역 metadata와 순서를 기록했다. 기존527개 단원/안내의 모든 값과 순서, 전체 domain 순서와 `unitIds` 외 metadata를 보존했다. 열역학 domain에만 세 ID를 뒤에 추가했다. 이전 주제의 등록 개수 고정 검사는 해당 시점의 기존 prefix를 계속 보존하도록 갱신했고, 새 통합 검사가 현재530단원·1,590안내와 새 꼬리를 정확히 검사한다.

전체 `npm test` **8,207개**가 통과했다. `npm run lint`, `npm run validate:content`, `npm run build`, 빌드 후 `npm run typecheck`, `npm run check:export`도 통과했다. 콘텐츠1,765 URL과 HTML1,768개의 내부 링크·자산·fragment·canonical·사이트맵을 확인했다. 세 본문의 한영16개 대제목, 완전한 전용 컴포넌트와 SSR Worker 접근0도 검사했다.

실제 산출물 `out/_next/static/chunks/detailed-balance-calculation.7c83cad98b36e7d7.js` 하나에 세 수치 모형의 계약이 들어 있다. 이 계약은 다른 JS 자산에서 발견되지 않았고, 실제 reader가 요청한 worker URL도 이 파일과 일치했다. main 의존 그래프에서 새 수치 모듈을 제외하는 독립 준비 단계 검사와 함께 계산 분리를 확인했다. 기존 다른 단원의 main 계산까지 모두 worker로 바꿨다고 주장하지 않는다.

## 실제 페이지 검증

`scripts/check-detailed-balance-readers.ts`는 별도 앱 주입 없이 실제 export와 native worker를 사용한다. 한영×명암×320/1440px×세 가족24조합에서 안내·수동 조작·초기화·퀴즈·새로고침을 검사했다. 개념6단계 전체와 모든36프리셋의 깊은 비교는 한영의 좁은 어두운 화면에서 수행했다. 모션 기본 설정과 저장 경계는 별도 상황으로 검사했다.

| 항목 | 확인 수 |
| --- | ---: |
| 실제 읽기 화면 조합 | 24 |
| 개념 단계 / 예측 선택 / 고정 준비 선택 | 36 / 108 / 90 |
| 안내 해설 / 안내 예제 선택 | 72 / 72 |
| 프리셋 선택 / 최종 퀴즈 선택 | 72 / 216 |
| 기록 표 페이지 이동 / 정확한 유리수 펼침 | 148 / 626 |
| Hastings 행렬 선택 / 로그 표시 전환 | 400 / 52 |
| 수동 단계 / 초기화 / 새로고침 복원 | 24 / 30 / 24 |
| 잘못된 초안 / 자유 입력 복귀 | 30 / 6 |
| 접근성 검사 상태 / 저장 경계 | 40 / 7 |
| 실제 시작 ACK에 연결한 중지 / 재시도 | 3 / 3 |
| 실제 worker 자산 로드 실패 후 복구 | 3 |
| 계산 중 요청 교체·순환/두 상태 | 2 |
| 자동 진행 / 전역 정지 / 화면 밖 정지 | 3 / 3 / 3 |
| reduced-motion 정지 / 입력 초점 정지 | 3 / 3 |
| 목록을 거친 클라이언트 재장착·복원 | 3 |
| reader worker 생성 / 종료 / 관찰한 시작 ACK | 472 / 472 / 469 |
| 관찰한 오래된 결과 / 페이지 오류 | 0 / 0 |

빠른 Hastings의 중지는 실제 시작 ACK 뒤 UI 중지 버튼을 실행한 검사다. 이 사실만으로 수치 커널 내부의 CPU 중단 시점을 증명하지 않는다. 순환·두 상태에서는 계산 중 표시가 나타난 요청을 실제로 교체하는 검사도 수행했다. 영어 페이지의 한글 누출과 문서 가로 넘침은 발견하지 않았다. 세 실제 모바일 스크린샷을 직접 확인했다. 자동 접근성 검사가 모든 가능한 상태와 요구를 보증하지는 않는다.

`scripts/check-detailed-balance-navigation.ts`는 메뉴·breadcrumb·검색·카드·표지 각12개 경우, 교재 목록 진입4개 경우와 Online Book 제외4개 경우를 확인했다. 이전/다음 링크로 네 번 이동했으며, 계산 중인 순환 단원에서 다른 단원으로 이동하는 경우도 포함했다. 이 별도 흐름의 worker17개도 모두 종료됐다. 기존 사용자 미리보기4175는 계속 제공된다.

## 증거와 남은 범위

증거는 `.runtime/learning/detailed-balance-integration/`의 `verification.json`, `registry-verification.json`, `worker-bundle-verification.json`, `worker-urls.json`, `full-tests.log`, `build.log`, `typecheck.log`, `lint.log`, `content-validation.log`, `export-check.log`, `browser-verification.json`, `navigation-verification.json`, 세 모바일 PNG에 있다. 소스4,248개·산출물4,287개의 해시를 기록했다. 이전4,244개 소스 중4,233개는 바이트를 보존했고, 나머지11개는 등록/분기5개와 단계에 맞춘 검사6개다. 새 파일은 보존 fixture와 통합/브라우저 검사4개다.

진행 기록에서 상세평형을 입문 주제 하나로만 올렸다. 원래 남은 입문 주제는73개이며 열역학에서 다음은 **유한계와 열역학 극한**이다. 기존 유한 저장고·Curie–Weiss·상관·유한 사슬과 겹치지 않는 범위를 먼저 설계하고, 열역학 영역 종합 점검 후 비평형·확률열역학·복잡계 등 남은 영역으로 이어간다. 일반 상태 공간의 샘플러 정리·물리적 국소 상세평형·열/엔트로피 생산이나 전체 연구 깊이를 이 유한 모델들로 완료했다고 세지 않는다.
