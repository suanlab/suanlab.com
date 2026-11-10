# 안장점·요동·비가우스 근사: 실제 교재 통합

2026-11-10. `saddle-fluctuation-and-nongaussian-approximation`을 Interactive Book의600번째 단원으로 등록했다. **600단원·1857안내·한영1200원고**, 미등록 초안0개다. [20요건 범위 감사](saddle-fluctuation-scope-audit.md)를 완료했다. 단원은 완료했으며 원주제와 전체 목표는 미완료다.

## 등록과 실제 화면

한영 제목·소개·세 선수 단원·10개념·7안내·8해설 문제를 등록하고, 표지/개념/embedded 실험을 실제 reader에 연결했다. 별도 실험의 개념/퀴즈는 숨겨 중복하지 않는다. 이전 단원은 Euclidean 커널이며 목차·영역 검색과 학습 완료/문제 답·개념 단계·입력/표시 복원을 확인했다. Online Book 페이지나 메뉴에 새 콘텐츠 링크를 추가하지 않았다.

소유 입력/정상점/모멘트 수치, 독립 검산과 Worker, 기존 홈페이지 색상/서체를 따르는 시각화는 앞 단계의 범위를 그대로 유지한다. 전체 행렬을 줄이거나 어려운 입력을 제외하지 않았다. 일반 비영 정상점의 실제 적분비/고차 오차, 실제 사차 공분산, 전체 well 적분이나 양자 터널링을 계산한 것으로 표시하지 않는다.

## 검증

- 새reader6검사가 Node20/24에서 통과했다. 앞93수치/원고·154검산/Worker·10학습 검사를 합쳐 관련263검사다.
- 기존599단원·안내·51영역의 값과 순서, 이전1198원고·navigation/Online Book page의 byte hash를 보존했다. 새 단원까지1200원고 모두 strict 수식/영어 표시 검사를 통과했다.
- 별도 환경 파일 없는 작업 폴더에서 Node20.20.2의 실제 Next14 정적 빌드가 성공했다.1838HTML과 내부 링크·자산·fragment·sitemap·canonical을 검사했다. 등록/분기와 result 귀속의9개 소스가 이 빌드와 일치한다.
- 실제 생산 reader의 한영/명암/320·1440px8조합에서80개념·64문제와8접근성, 완료/퀴즈/개념/입력 저장·선수/이전/목차·영역 검색·3D 키보드/WebGL 복구를 확인했다.
- 실제 Next가 생성한 계산/검산 Worker URL을 읽어HTTP응답과 출력 파일의 hash를 비교했다.54입력 계산→독립 검산·수치 변조54·요청 귀속54거부·두 Worker 종료를 확인했다. 브라우저 응답은 Node 생산기와 같으며 독립 Python54기준의 물리 대조는 별도 수치 단계 증거다.
- 잘못된JSON/혼합 모형/잘못된 학습 상태/읽기 거부/쓰기 거부/정상 well 복원의6흐름,20기준 그림의 글자 경계와 최대32구간의 재생/실제 계산·검산 중 취소/타이핑/재시도/오입력/수정/새로고침을 검증했다. 추가7접근성까지 이 단계의 접근성은15회다. 최대 흐름 Worker9생성/9종료와 복원 후2생성/2종료·활성0이었다.

검사 harness의 alert 선택을 실험 안으로 한정해 Next의 전역 route announcer와 구별했다. 초기 테스트 기대/선택자 실패 로그를 보존한다. 제품의 오류 메시지나 입력 규칙을 느슨하게 바꾼 것이 아니다.

## 로컬 홈페이지와 원격 게시

기존4547출력 파일을 모두 hash 대조한 뒤 백업하고, 검증한600단원 출력을 로컬 `out/`에 반영했다.1838HTML/4612전체 파일이며 이전 Worker65개와 이전 static 자산을 보존했다. 기존4175 preview의 Interactive Book·교재 목록·새 readerHTTP200을 확인했다. 별도 후보 서버4203과 학습 화면34515도 유지한다.

앞06f4a33의 CI 검증/빌드는 성공했지만 Pages배포가 서비스 시간 초과로 실패했다. 이어진 최신 실행을 추적하고 있으며, 새600단원의 공개 도메인 게시 성공은 commit/push 이후 별도로 확인한다. 로컬 생산 검증을 원격 배포 성공으로 표시하지 않는다.

## 재현과 다음 순서

```bash
npx tsx --test scripts/learning-saddle-fluctuation-reader-integration.test.ts
npx -p node@20 node --import tsx --test scripts/learning-saddle-fluctuation-reader-integration.test.ts
SADDLE_FLUCTUATION_PREVIEW_BASE=http://127.0.0.1:4203 npx tsx scripts/check-saddle-fluctuation-reader.ts
SADDLE_FLUCTUATION_PREVIEW_BASE=http://127.0.0.1:4203 npx tsx scripts/check-saddle-fluctuation-production-assets.ts
SADDLE_FLUCTUATION_PREVIEW_BASE=http://127.0.0.1:4203 npx tsx scripts/check-saddle-fluctuation-reader-maximum.ts
npm run typecheck
npm run lint
npm run validate:content
npm run check:export
```

생산 자산 스크립트의 `SADDLE_FLUCTUATION_EXPORT_ROOT`는 서버의 실제 `out/` 경로를 지정할 수 있다. 기본값은 `/tmp/suanlab-saddle-fluctuation-candidate-20261110/out`이다. 검증 증거와 백업은 `.runtime/learning/saddle-fluctuation-reader-integration/`에 보존한다.

다음은 **실시간 위상·고전극한**의 일차 자료와 유한 모형 계약이다. 네 처리 중 마지막 단원과 원주제 감사가 남아 입문315/362검증·47미완료는 그대로다. 전체51영역509주제·영역 감사·연구 심화도 계속 진행한다.
