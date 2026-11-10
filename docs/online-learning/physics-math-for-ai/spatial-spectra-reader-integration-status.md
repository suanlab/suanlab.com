# 공간 상관·스펙트럼: 실제 교재 통합

2026-11-10. `spatial-spectra-decay-and-correlation-length`을 Interactive Book의597번째 단원으로 연결하고 검증한 정적 출력을 로컬 홈페이지에 반영했다. 안내1838개·한영1194원고이며 미등록 초안은0개다. 상관함수 세 처리의 원주제 감사는 아직 남아 입문314/362검증·48미완료를 유지한다.

## 반영한 내용

한영22절·13표시수식·8해설 문제,8개념·6예측 안내·21입력 예제와 실험을 연결했다. 목록 표지와 기준 개념도, 위치/Fourier 행렬 열지도·Three.js, 거리별 유한/무한 상관, 모드 파워와 물리 파수, 고정 길이 정련 표를 제공한다. 기준 막대 그림에는 패널별 최대값을 표시해 다른 높이 정규화를 구별했다.

Interactive Book 진입→교재 목록→영역 검색→새 단원, 통합 검색과 선수/이전 단원 링크를 확인했다. Online Book 목록에는 새 단원 링크를 추가하지 않았다. embedded 실험에서는 별도의 개념·퀴즈를 중복 출력하지 않는다. 기존 홈페이지의 공통 색·스타일을 사용한다.

## 확인한 증거

- 관련117검사와 표시 보완 후15검사, 이전/신규 등록 보존4검사를 통과했다. 기존596단원·안내와51영역 범위의 내용·순서를 해시로 대조했다. 등록된 한영1194원고를 strict 수식·영문 내 한글·중복 h1 검사로 확인했다.
- TypeScript·ESLint·콘텐츠1832URL·정적 빌드와1835HTML의 링크/자산/fragment/sitemap/canonical 검사를 통과했다. 검토 빌드에는 실제 Tailwind 설정과 보관 QT manifest까지 포함했다. 기존 출력4527파일의 백업을 검증했다.
- 실제 Next reader를 한영×명암×320/1440px의8조합에서 확인했다.64개념·64퀴즈의 정답/오답 해설, 입력·퀴즈·완료 저장과 복원,3D 키보드·컨텍스트 손실 복구와 접근성8회를 확인했다. 한영 기준 그림16상태를 기록했다.
- 실제 Next 계산/검산 자산을HTTP와 디스크 바이트로 대조했다. 독립 참조30입력과21예제, 반복 포함51요청을 계산·검산하고 수치 변조51건·요청 귀속 불일치51건을 거부했다. 표시 보완 후에도 이 두 자산의 해시가 같음을 확인했다.
- 최대 입력의320px 실제 reader에서 터치 회전·표시 재생의 비재계산·취소/실패/재시도·새로고침 복원을 확인했다. 높이는43855px이며 새로고침 전6/후2Worker가 모두 종료됐다. 검사 후 브라우저 정리 중 종료143이 발생한 실행은 별도로 보존하고, 같은 최종 화면을 단독 재실행해 정상 종료0을 확인했다.
- 한영 메뉴/검색과 기존 가우스·비가우스·장 모드3reader를 확인했다. 기존 생산Worker60자산을 보존했으며 후보 출력과 채택한4535파일이 동일하다.

## 실행 근거와 재현

`.runtime/learning/spatial-spectrum-reader-integration/verification.json`과 관련 로그·사진·해시에 기록했다. 실제 등록은 `src/data/spatial-spectrum-reader.ts`, dispatch는 기존 교재/개념/표지 컴포넌트와 새 reader wrapper에 있다.

```bash
npx tsx --test --test-concurrency=2 scripts/learning-spatial-spectrum-{numerics,manuscripts,workers,controller,teaching,reader-integration}.test.ts
npx tsx scripts/check-spatial-spectrum-reader.ts
npx tsx scripts/check-spatial-spectrum-production-assets.ts
npx tsx scripts/check-spatial-spectrum-reader-maximum.ts
npx tsx scripts/check-spatial-spectrum-catalog.ts
```

브라우저 검사는Playwright Chromium과 정적 미리보기가 필요하다. 기본 후보 주소는 `http://127.0.0.1:4191`이며 `SPATIAL_SPECTRUM_PREVIEW_BASE`로 바꿀 수 있다. 자산 검사에는 `SPATIAL_SPECTRUM_EXPORT_ROOT`로 실제 출력 디렉터리를 지정한다. 로컬 채택 주소는 `http://127.0.0.1:4175`다.

## 다음 순서

[25요건](spatial-spectra-acceptance-matrix.json) 중 단원 구현·통합24개를 확인했다. 마지막 원주제 감사가 남아 전체 완료로 선언하지 않는다. 가우스·비가우스·공간 스펙트럼 세 처리의 입문 범위를 대조한 뒤 경로적분의 정의·근사로 이어간다. 전체51영역509주제·영역 감사·연구 심화는 계속 미완료다. 검증은 로컬 출력에서 완료했으며, 홈페이지 반영 요청에 따라 검증본을 commit/push로 게시한다.
