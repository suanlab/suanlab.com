# Euclidean 커널: 한영 개념·시각화·실습

2026-11-10. `euclidean-quantum-kernels-and-time-slicing`의 한영 학습 화면을 구현하고 로컬 검토 환경에서 확인했다. **8개념·6예측 안내·8해설 퀴즈·20입력 예제**다. 관련211검사가 Node.js20/24에서 통과했다. 실제 Interactive Book 등록과 생산 Next 자산 검증은 다음 단계이며598등록 단원·1844안내·등록1196원고·미등록 초안1개를 유지한다.

## 설명과 그림

이동/위치 비용→커널/조건부 정규화→고정끝 행렬→완전제곱/요동 행렬식→Mehler/유한 격자→끝점 규칙/오차→열적 trace→확률 해석의8단계다. N2 손계산Q=9/2,μ=4/9,C=2/9,P=4/9,최소 작용49/72와 trace제곱16/17을 연결한다.

공통 홈페이지 CSS·색 변수와 기존 실습 스타일을 사용했다. 한영16기준 그림·compact 표지·SVG title/desc를 제공한다. 기준 그림은 현재 입력과 별개라고 표시한다. 사다리꼴 potential의 양끝 절반 가중치,고정v의 주행렬식과 고정시간 정련의 차이,그림의 제한 범위와 전체 직선 적분의 차이를 명시했다. 긴 상자 설명은 줄을 나누었고 모든 라벨이 viewBox 안에 있는지 확인했다.

## 현재 입력의 실험

유한 평균·시각별 표준편차 띠·자유 평균·연속 Mehler 평균을 함께 그린다. 점 사이 연결은 표시 방법이고 표본 경로나 전체 경로 신뢰 구간이 아니다. 위치/정밀도 열지도와 event-driven Three.js 막대는 최대33×33공분산 항목을 보존한다. 두 가로 축은 시각 쌍이며 경로 곡면으로 해석하지 않는다.

정확한 커널 계수·최소/기대 작용·오른쪽 보정·끝점 적분·trace·정련을 표로 제공한다. 열적 위치 밀도는 공통 위치x의 대각을 trace로 정규화한 대상으로 고정끝 경로 분포와 구분한다. 그림은±3표준편차만 보여주며 꼬리가 남는다고 설명한다. 자유 trace가 발산하면 유한 열적 밀도를 그리지 않는다.

연속 Mehler 비교는 정확한 Worker 응답과 별도의 소수 근사다. 안정적인 작은ω 식을 사용하고6개 독립80자리 기준·24정련 행과 대조했다. 부동소수 해상도로 구별하지 못하는 차이는0이 아닌 미해결로 표시한다. 이 필터는 인증 구간이 아니다. 자유 커널의 오차0만 별도의 정확한 합성 항등식으로 표시한다.

## 입력·학습 상태와 모션

N/probe 문자열을 정수 문법으로 먼저 검사해`2.0000000000000001`의 반올림 수용과 조용한 clamp를 막았다. 부호를 포함한26글자 분수·기존 최대 입력을 유지한다. 새 입력은 선언한probe의 시각/대각을 선택하며 행렬 종류 전환은 가능한 같은 물리 첨자 쌍을 유지한다. 정밀도에서 표현할 수 없는 끝점은 가까운 내부 첨자로 옮기며 수학 입력은 바꾸지 않는다.

버전1저장 스키마는 입력·행렬/척도/시각/셀·개념 단계·안내·해설·퀴즈 응답을 소유 복사한다. 잘못된 입력/표시와 저장 불가 상태에서는 세션 실험을 계속한다. 시간 재생은 표식만 옮기며 새 경로나 계산을 만들지 않는다. 동작 줄이기 설정에서는 재생을 비활성화하고 숨김/unmount에서 타이머를 정리한다.

## 확인한 증거

- Node.js20/24 관련211검사: 기존 수치/검산/전송/수명201개와 학습 화면10개다. 한영 상세 원고·실행 Python·완전한 행렬 투영·소수 기준·SSR·클라이언트 산술 제외·저장 소유권·물리 첨자 보존을 포함한다. TypeScript·ESLint·콘텐츠 검증도 통과했다.
- 한영×명암×320/1440px8조합에서64개념·48안내·64퀴즈의 정답/오답 해설과41예제 적용을 확인했다.20예제를 한 모바일 흐름에서 모두 적용했다. 3D 키보드·context loss/retry·영문 한글 혼입·문서 가로 넘침·표시 조작의 비재계산을 검사했다. 계측114Worker가 모두 종료됐다.
- 최대32구간 모바일에서1089셀·N2 정밀도1셀3D·터치 회전·시간 재생·취소/오류/재시도·새로고침 입력/표시/학습 복원을 확인했다. 이 흐름의 계측14+2Worker가 모두 종료됐다. 오류/취소 재현에는 일부 postMessage 보류/오류 주입을 사용했다. 한영 입력 손상·표시 손상·저장 불가6흐름도 검사했다.
- 선택 보완 후 별도320px실제 화면에서 N2→4→32의 probe 선택,공분산/정밀도17,17과17,18쌍의 왕복,시각 표식 독립 조작·저장 복원을 확인했다.6+2Worker가 모두 종료됐고 표시 전환은 Worker를 추가하지 않았다.
- 최종 한영 기준16이미지와12안내를 재확인했다. 자유 trace의 발산과 미세 로그 차이의 미해결 표시를 실제 폼 입력으로 확인했다. 앞15접근성 검사에 최종 기준2회·선택1회를 더해 main 범위18접근성 검사를 통과했다.

## 재현과 다음 순서

```bash
python3 scripts/learning/generate-euclidean-kernel-display-reference.py
npx tsx --test --test-concurrency=2 scripts/learning-euclidean-kernel-{numerics,validation,manuscripts,workers,controller,teaching}.test.ts
npx tsx scripts/preview-euclidean-kernel.ts
EUCLIDEAN_LANGUAGE=en EUCLIDEAN_THEME=dark npx tsx scripts/check-euclidean-kernel-teaching-native.ts
npx tsx scripts/check-euclidean-kernel-storage-native.ts
npx tsx scripts/check-euclidean-kernel-selection-native.ts
npx tsx scripts/check-euclidean-kernel-reference-diagrams.ts
```

미리보기는 기존출력의 스타일과 이전 단계의 `.runtime/learning/euclidean-kernel-worker/{calculate,validate}.js`를 사용한다. `review-server.json`에 주소를 기록한다. 검토 중에는35239포트에서 실행했고 홈페이지 출력은 다시 빌드하거나 덮어쓰지 않았다. 실제 증거는 `.runtime/learning/euclidean-kernel-teaching/verification.json`과 관련 로그·사진에 있다.

앞 게시분d686c1b는 GitHub 배포 성공과 공개 교재598링크·Brownian bridge 페이지HTTP200을 확인했다. 신규 Euclidean 단원은 아직 그 목록에 없다. 다음은 실제 reader/개념/표지 dispatch·검색/선수 연결·생산 자산·기존598단원 보존을 검증하는17/18요건 이후의 마지막 단계다. 경로적분 원주제·전체51영역509주제·영역 감사·연구 심화는 미완료이며 입문315/362검증·47미완료를 유지한다.
