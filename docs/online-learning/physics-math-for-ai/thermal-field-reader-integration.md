# 유한 장의 측도와 열 자유에너지: reader 등록과 단원 검토

2026-11-11. `finite-field-measure-and-thermal-free-energy`를 **602번째 단원**으로 등록했다. [24요건 행렬](thermal-field-scope-audit-matrix.json)을 원고·계산·실제 화면·저장·출력과 대조해 이 단원의 입문 계약을 충족한다고 판정했다. **통계장론 원주제 전체와 전체 교육과정은 미완료**다.

## 등록한 내용과 기존 콘텐츠

한영 각21절·display 수식9개·손계산·동일 Python 실습을 `content/learning/`에 등록했다. 개념6단계·안내6개·해설 문제8개와 전용 표지, lazy 실험을 연결했다. 등록 전 원고의 준비 안내를 현재 reader 설명으로 바꾸었고, 실행 코드는 기존 검산한 예제와 같다. 선행 단원은 Gaussian 상관·공간 스펙트럼·정준 에너지 요동이다.

기존 **601단원·안내 데이터·원고1202편·메뉴·공유3D 코드**를 등록 전 hash fixture와 대조했다. 순서와 기존 객체·원고 byte를 유지하고 새 항목만 추가했다. field-multiscale의 연결 단원도 하나 추가했다. Online Book 페이지와 메뉴를 수정하지 않았다. Interactive Book을 통한 기존 연결 구조를 유지한다. 등록 전체는 **602단원·1869안내·1204원고**다.

## 저장과 복원

버전1 실험 상태는 적용한 입력과 미완성 편집, 안내/해설, 화면 선택을 구분해 저장한다. 결과·검산 영수증·실행 플래그·재생 여부는 저장하지 않는다. 적용한 입력은 bounds 검사와 정규화를 다시 거치며 결과는 새 Worker로 계산·독립 검산한다. 미완성 편집은 길이가 제한된 문자열로 복원하고 결과는 이전 적용 입력의 값임을 알린다.

reader의 개념 단계/모션 설정과 공통 문제 답변/완료 표시는 별도 key로 저장한다. embedded 실험에는 개념도와 문제를 중복 표시하지 않는다. 손상된 상태는 기본값으로, 저장 차단은 방문 중 실험으로 처리한다. 저장 복원이 끝나기 전에는 쓰기나 계산 Worker를 시작하지 않는다. T=3과 미완성 편집을 미리 넣은 실제 페이지에서 첫 저장·첫 계산이 모두T=3임을 확인했다.

## 실제 reader와 생산 자산

Node20/Next14 정적 빌드의 실제 페이지를 한영×명암×320/1440px **8상태**에서 확인했다. 48개념·48안내·64문제의 정오답과 해설, 적용 입력·미완성 편집·행렬/성분·답변/완료 표시의 reload 복원을 검사했다. 영어 한글 혼입·문서 가로 넘침·pageerror가 없고, axe 위반도0이다. 대표 생산 reader screenshot을 직접 확인했다.

실제 reader의 계산 취소/재시도와 visibilitychange 경로도 확인했다. 가시성 검사는 문서 상태를 모사해 이벤트를 전달했다. 숨김 시 이전 결과를 숨기고 Worker를 종료하며, 복귀하면 현재 입력을 새로 계산한다. 이 별도 검사에서6개 Worker가 종료됐다. 8상태의 port 카운터는 **reload 후 마지막 문서의 snapshot**으로, 이전 문서의 횟수를 합산하지 않는다.

별도 검사는 실제 reader factory가 만든 Worker URL과 옵션을 수집해 그대로 사용했다. **새 Fraction 기준52모형+현재14예제**, 총66응답이 Node와 전체 hash가 같고, 실제 독립 검산 자산을 통과했다. 수학값·구간·귀속 변조3개를 거부했으며 검사135개 Worker는 모두 종료됐다. 로드한 스크립트 hash도 보존한다. esbuild adapter나 수동으로 고른 Worker 유형으로 대체하지 않았다.

## 출력과 홈페이지 반영

정적 **1840HTML**의 내부 링크·자산·fragment·sitemap·canonical 검사를 통과했다. 발견한 `visual-intuition` 앵커 누락을 수정하고 다시 빌드·검사했다. 최종 빌드의 런타임 소스1709파일은 현재 checkout과 byte가 같다. 입력·정규화·저장·등록의 새6검사는 Node20/24에서 통과했고, 과거 phase의 보존 검사도 통과했다. TypeScript와 ESLint도 통과했다.

직전 준비 커밋의 CI는 공통 `learning-worker-message-error.test.ts`가 새 제어기를 발견했으나 대표 입력 fixture가 없어 실패했다. `worker-messageerror-inputs.json`에 유효한 thermal-field 입력을 추가해 기존 항목을 보존했다. 전체 제어기의 EventTarget·재시도·교체·취소·오류·deadline 공통70검사는 Node20/24에서 통과했다. 새 제어기를 추가할 때 이 공통 목록도 함께 등록해야 한다. 전체 CI와 공개 배포 성공 여부는 이 로컬 검증과 별도로 추적한다.

후속 관찰: `f04715ac`의 고정 worktree에서 실행한 Node20 `npm run check`가 종료 코드0으로 완료됐다. 테스트 **26,380개 전부 통과**했고 lint·콘텐츠·빌드·전체 타입·1840HTML export 검사도 통과했다. 로그는 `.runtime/learning/thermal-field-reader-integration/full-check-isolated-node20.log`다. 이 결과는 해당 고정 커밋의 증거이며, 이후 추가한 국소 질서변수 코드의 전체 CI 통과나 공개 배포 성공을 대신하지 않는다.

기존 홈페이지4754파일을 백업하고 새4559파일을 반영했다. 기존 정적 자산491개 중 새 출력에 없는259개도 보존해 설치본은4818파일이다. 후보/설치본·백업의 hash와 이전 HTML 경로 보존을 확인했다. 기존 로컬 홈페이지 서버에서 새 reader HTTP200도 확인했다. 공개 GitHub Pages 배포 성공은 commit/push와 별도로 확인한다.

## 재현과 다음 순서

```bash
node --import tsx --test scripts/learning-thermal-field-state.test.ts scripts/learning-thermal-field-reader-integration.test.ts
node --import tsx --test scripts/learning-worker-message-error.test.ts
node --import tsx scripts/check-thermal-field-reader.ts
node --import tsx scripts/check-thermal-field-production-workers.ts
```

두 브라우저 명령은 `THERMAL_READER_CANDIDATE`의 `out/`을 사용하며 자체 서버와 브라우저를 종료한다. 근거는 `.runtime/learning/thermal-field-reader-integration/`의 보존·생산 Worker·reader·설치 결과와 빌드 로그다. 원래 서버는 유지한다.

이 단원은 통계장론의 네 처리 중 첫 번째다. 다음은 **미시상태에서 국소 질서변수로**의 범위·모형 검토이며, 상호작용 scalar 장과 Legendre 유효 작용이 뒤따른다. 원주제 감사는 이들을 함께 검토한 뒤 진행한다. 입문 원주제 **316/362검증·46미완료**, 전체51영역509주제와 연구 심화 미완료 상태를 유지한다.
