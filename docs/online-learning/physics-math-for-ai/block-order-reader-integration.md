# 미시상태에서 국소 질서변수로: 실제 reader와 단원 검토

2026-11-11. `microscopic-blocks-and-local-order-parameters`를 **603번째 단원**으로 연결하고 [25요건 행렬](block-order-scope-audit-matrix.json)을 원고·계산·실제 화면·생산 자산·저장·출력과 대조했다. 이 단원의 입문 계약은 충족한다. **통계장론 원주제와 전체 교육과정은 미완료**다.

## 연결한 콘텐츠와 보존

한영 각20절·display 수식9개·손계산·동일 Python 예제를 `content/learning/`에 등록했다. 개념6단계, 안내6개, 공통 문제8개, 전용 표지와 lazy 실험을 연결했다. standalone에서 검증한 실험의 개념/문제는 embedded 모드에서 중복 표시하지 않는다. 선수 단원은 미시 상태 수·조건부 엔트로피·유한 장의 측도와 자유에너지다.

기존 **602단원·안내 데이터·1204원고·메뉴·공유 그래픽**을 등록 전 hash fixture와 대조했다. 기존 객체와 순서를 유지하고 새 항목만 추가했다. curriculum coverage 원장에 단원을 추가한 뒤 생성 스크립트 결과가 후보 소스와 같은지도 확인했다. 등록 전체는 **603단원·1875안내·1206원고**다. 기존 Interactive Book 연결을 유지하며 Online Book 페이지와 메뉴는 보존했다.

## 실제 reader 화면

Node20/Next14의 정적 후보 빌드에서 한영×명암×320/1440px **8상태**를 검사했다. 실제20절과 개념48회·안내48회·64문제의 정오답/해설을 확인했다. 영어 한글 혼입·문서 가로 넘침·pageerror가 없고 axe 위반도0이다. 실제 개념도와 공간 그림 screenshot을 직접 확인했다.

프로필을 고정한 조건부 평균, 블록 평균, 전체 앙상블 평균과 미시 배치를 구별한다. 높이의 스핀 단위를 유지하고 화면 종횡비에 맞춰 카메라를 배치한다. 3D 시점의 키보드 조작과 닫기를 확인했다. 별도 실제 Next 화면에서 WebGL을 막아도2D 지도·정확한 표가 유지되고, 동작 줄이기 설정에서는 자동 재생과 개념 전환 효과가 꺼진다.

## 저장과 수명

적용한 입력과 미완성 편집, 모드·배치·프로필 페이지를 분리해 저장한다. 결과와 실행/재생 플래그는 저장하지 않는다. 실제 페이지에서 미완성 `1e-`, 조건부 모드, 배치1과 두 번째 프로필 페이지가 reload 뒤 복원됐다. 공통 문제의 선택 답변과 완료 표시, 별도 key의 개념 단계도 복원됐다.

손상된 상태는 기본값으로, 저장 차단은 방문 중 실험으로 처리한다. 저장된 x=2와 미완성 편집을 미리 넣은 페이지에서 **첫 저장과 첫 계산 모두 x=2**임을 확인했다. 기본 입력을 먼저 쓰거나 계산하지 않는다.

실제 취소/재시도와 visibilitychange 경로도 확인했다. 가시성 검사는 문서 상태를 모사한 이벤트다. 숨김 시 결과가 사라지고 Worker가 종료되며 복귀 후 현재 입력을 새로 계산한다. 이 별도 수명 검사에서6개 Worker가 종료됐다.8상태의 port 카운터는 **reload 후 마지막 문서의 snapshot**이며 이전 문서의 횟수를 합산하지 않는다.

## 실제 Next Worker 자산

실제 reader factory가 생성한 Worker URL과 옵션을 수집해 그대로 사용했다. 새 Python Fraction24모형, 현재9예제, 최대 유리수 활동도1모형의 **34응답**이 Node의 전체 result hash와 같고 실제 독립 검산 자산을 통과했다. 계산값·구간·귀속 변조3개를 거부했다. probe Worker **71개 모두 종료**했으며 초기 reader의2개는 이 수에 포함하지 않는다. 로드한 스크립트 hash도 기록했다.

esbuild adapter나 수동으로 추측한 Worker 유형으로 대체하지 않았다. 전체 reader graph에서 새 생산/검산 엔진은 제외된다. 이전 free-energy reader가 공유 로그 모듈을 직접 가져오는 기존 경로는 남아 있으므로, 전체 사이트의 모든 수학 코드가 main thread에서 제거됐다는 주장은 하지 않는다. 새 단원 자체의 lab graph에는 그 모듈도 없다.

## 빌드·설치·검증 범위

정적 **1841HTML**의 내부 링크·자산·fragment·sitemap·canonical 검사가 통과했다. 콘텐츠 검사는1838URLs·293publications·51projects를 확인했다. 후보와 현재 checkout의 소스·원고·정적 원본 등4704파일이 byte 단위로 같다.

관련112개 회귀 검사가 Node20에서 통과했다. 기존 thermal reader 보존 검사, 새 단원 계산/프로토콜/수명/화면·상태 검사를 포함한다. 새 reader3검사와 등록 원고 증거는 Node20/24에서 확인했다. 전체 TypeScript와 변경 파일 lint도 통과했다. 앞선 `f04715ac` 전체26380검사 결과는 해당 고정 커밋의 증거이며, 이 새 커밋의 전체 CI 결과와 구별한다.

기존 로컬 홈페이지4818파일을 백업했다. 후보4566파일에 새 출력에 없는 이전 자산317개를 보존하여 설치본은4883파일이다. 후보·설치본·백업 hash와 이전1840HTML 경로 보존을 확인했다. 기존 로컬 서버에서 새 reader HTTP200을 확인했다. 공개 GitHub Pages 배포는 commit/push와 별도로 추적한다.

## 재현과 남은 내용

```bash
node --import tsx --test --test-concurrency=2 scripts/learning-block-order*.test.ts scripts/learning-worker-message-error.test.ts scripts/learning-thermal-field-reader-integration.test.ts
node --import tsx scripts/check-block-order-reader.ts
node --import tsx scripts/check-block-order-production-workers.ts
node --import tsx scripts/check-block-order-reader-fallback.ts
```

증거는 `.runtime/learning/block-order-reader-integration/`에 있다. 이번 처리 뒤 통계장론에는 **상호작용 scalar Gibbs 장**, **연결 생성함수와 Legendre 유효 작용**이 남는다. 원주제 감사·임계성/보편성·재규격화·유효 이론·척도 분리와 연구 심화도 별도다. 원래 입문 원주제의 검증 수316/362·미완료46개는 변하지 않는다.
