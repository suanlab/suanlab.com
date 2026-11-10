# Brownian bridge: 실제 Interactive Book 통합

2026-11-10. `brownian-bridge-time-slicing-measure`를598번째 단원으로 등록하고 검증한 정적 출력을 로컬 홈페이지에 반영했다. **598단원·1844안내·한영1196원고**, 미등록 초안0개다. 경로적분 네 처리 중 첫 처리이며 입문315/362검증·47미완료를 유지한다.

## 반영한 내용

한영20절·12표시수식·8해설 문제,8개념·6예측 안내·16입력 예제를 연결했다. 평균과 시각별 표준편차, 위치/증분/정밀도 열지도·Three.js, 정확분수 표와 면적/정련 비교를 제공한다. 표시 곡선을 표본 경로나 전체 경로 신뢰 구간으로 설명하지 않는다. 입력 한도는32구간이며 최대256구간 비교는 해석식이다.

도서 메뉴의 Interactive Book에서 진입하고 목록·영역/통합 검색·선수/이전 단원 링크로 찾는다. Online Book 목록은 별도로 유지한다. 공통 홈페이지 색과 스타일을 사용하며 실제 교재의 공통 개념/퀴즈와 실험 내부 UI를 중복 출력하지 않는다. 개념 단계·입력/표시·안내·퀴즈·완료를 저장하고 복원한다.

## 검증

- Node.js20/24 관련200검사를 통과했다. 기존597단원·안내 및51영역 내용과 순서의 해시가 유지된다. 등록1196원고의 strict 수식·영문 내 한글·중복 h1을 검사했다.
- TypeScript·ESLint·콘텐츠1833URL·정적 빌드와1836HTML의 링크/자산/fragment/sitemap/canonical 검사를 통과했다.
- 실제 Next reader의 한영×명암×320/1440px8조합에서64개념·64문제, 입력/학습/완료 복원,3D 키보드·컨텍스트 손실 복구·접근성을 확인했다. 기준 그림16개를 기록했다.
- 최대32구간 모바일에서1089개 공분산 셀·터치 회전·재생의 비재계산·취소/오류/재시도·새로고침 복원을 확인했다. N2 정밀도1셀,26글자 부호 분수,잘못된 입력의 무계산 거부와 한영 저장 오류를 포함한7흐름을 정상 종료0으로 재확인했다.
- 실제 생산 계산/검산 자산을HTTP/디스크 SHA로 대조했다. 독립47입력+16예제의63요청을 확인하고 수치 변조·요청 귀속 오류를 각각63건 거부했다.
- 이전 출력4535파일을 해시가 같은 백업으로 보존했다. 기존63Worker자산을 유지했고 새 후보/채택 출력4541파일이 동일하다. 한영 목록/검색과 기존 Gaussian·비Gaussian·공간 스펙트럼3reader를 확인했다.

## 재현과 다음 순서

실행 증거는 `.runtime/learning/brownian-bridge-reader-integration/verification.json`과 로그·사진·해시에 있다.

```bash
npx tsx --test --test-concurrency=2 scripts/learning-brownian-bridge-*.test.ts
npx tsx scripts/check-brownian-bridge-reader.ts
npx tsx scripts/check-brownian-bridge-reader-maximum.ts
npx tsx scripts/check-brownian-bridge-production-assets.ts
npx tsx scripts/check-brownian-bridge-catalog.ts
```

브라우저 검사는Playwright Chromium과 정적 미리보기가 필요하다. `BROWNIAN_BRIDGE_PREVIEW_BASE`로 주소를 지정하며 기본 후보는4201, 채택한 로컬 홈페이지는4175포트다. 자산 검사는 `BROWNIAN_BRIDGE_EXPORT_ROOT`로 실제 출력 디렉터리를 지정한다.

단원21요건을 완료했다. 다음은 Euclidean 양자 커널·시간 분할 오차, 이후 안장점/비가우스 근사와 실시간 위상/고전 극한이다. 네 처리 후 경로적분 원주제를 감사한다. 전체51영역509주제·영역 감사·연구 심화는 미완료다. 이번 통합의 공개 배포 성공은 별도로 확인한다.
