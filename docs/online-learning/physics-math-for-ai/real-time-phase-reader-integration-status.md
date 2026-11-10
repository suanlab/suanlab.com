# 실시간 경로적분의 실제 교재·생산 자산 통합

2026-11-11. `real-time-phase-and-classical-limit`을 Interactive Book의 마지막 단원으로 등록했다. **601단원·1863안내·1202한영 원고**, 이 단원은 **22/23요건 검증**이다. 최종 단원 범위 감사와 네 경로적분 단원의 원주제 감사는 남는다. 입문315/362검증·47미완료, 전체51영역509주제·영역 감사·연구 심화의 미완료 범위를 유지한다. 새 단원은 아직447개의 완전 검증된 신규 단원 수치에 포함하지 않는다.

## 실제 학습 화면

교재 목록·장이론 영역 검색·sitemap에 연결했다. 유클리드 경로적분, 안장점·요동, 파동함수 정규화 단원을 선수 학습으로 제공한다. 이전 단원은 안장점·요동이며 다음 링크는 교재 목차다. 기존 Book → Interactive Book 메뉴를 유지하고 Online Book 페이지는 변경하지 않았다.

compact 표지,13단계 개념도,33절 상세 원고, 본문8해설 문제와 완료 표시, 옆 실험의6안내·예측·세 모형 편집·2D/3D·저장을 연결했다. embedded 실험에서는 별도 퀴즈와 개념 단계를 숨겨 중복하지 않는다. 계산 실험은 lazy chunk로 불러온다. 수치 producer·exact core·독립 수학 checker는 reader 의존성에 들어가지 않는다.

## 실제 빌드와 브라우저

- 환경 파일을 복사하지 않은 별도 후보 폴더에서 Node20.20.2/Next14.2.35 정적 빌드 성공.1839HTML의 내부 링크·자산·fragment·sitemap·canonical,1836콘텐츠 URL을 검사했다. 최종 실행 소스1684개가 후보 빌드와 일치한다.
- 신규 reader6검사를 포함한 관련19검사가 Node20/24에서 통과했다. reader 의존성 허용 목록을 강화한 후 Node24의6검사,Node20의 해당 의존성 검사를 다시 확인했다. 전체1202원고의 strict 수식·영어 표시를 검사했다.
- 실제 생산 reader의 한영 × 명암 ×320/1440px8조합에서104개념·64문제의 정오답·48안내, quiz/완료/개념/안내별 예측/입력의 새로고침 복원, 선수·이전·목차·영역 검색을 확인했다. 사슬·packet·fold의 접근성24회와 데스크톱4상태의3D 키보드·컨텍스트 복구를 확인했다. 추적 reader port80개가 모두 종료되었다.
- 실제320px reader에서 최대31×31 Hessian,961행 전체 표·성분960·3D 선택의 복원, 키보드와 실제 CDP touch에 따른 그림 변화, WebGL 컨텍스트 손실 복구를 확인했다. 표시 조작은 재계산하지 않으며 추가 접근성1회·port6개 종료를 확인했다. 물리 모바일 기기 전체의 검증을 주장하지 않는다.

## 생산 Worker와 Node 대조

화면이 실제 생성한 계산/검산 Worker URL을 수집해 HTTP200 및 파일 byte hash를 확인했다. 이 두 생산 자산에서 **원래32예제 + 안내의 추가6입력 =38개 고유 입력**을 계산하고 독립 전체 응답 gate로 검산했다. 유효한 형식을 유지한 Hessian·packet 공분산·Airy 구간 변조38건, 요청 귀속 변조38건을 거부했다. 직접 사용한 두 Worker도 종료했다.

별도 Node producer의38전체 결과 hash가 브라우저 생산 결과 및 검산 receipt와 일치한다. Node 대조는 같은 producer의 플랫폼 일관성 검사이며 독립 수학 증명이 아니다. 독립 수학 검증은 별도 검산 Worker의 전체 응답 gate와 앞선 수치/검산 단계에서 다룬다.

처음 Node 기준 생성 도구의 source hash 목록에 잘못된 파일 경로가 있어 기준 파일 저장에 실패했다. 생산 Worker38회는 이미 완료되어 응답·변조 거부·종료 기록을 보존했다. 경로를 고치고 source 파일을 계산 전에 확인하도록 바꾸어 Node 기준을 재생성했다. 이미 완료된 native 실행 기록을 비교 모드로 읽어38개 hash를 대조했으며, Worker 자산이 해당 기록 이후 바뀌지 않았는지도 확인했다. 실패 로그는 보존했고 입력 범위나 검산 조건을 완화하지 않았다.

## 기존 홈페이지 보존과 반영

기존600단원·안내의 순서/값,1200원고 byte,51영역의 원래 주제·선수 관계, navigation과 Online Book 페이지 hash를 보존했다. 이전 정적 자산931개와 알려진65Worker 자산을 같은 byte로 유지했다.

이전4612출력 파일을 검증해 `.runtime/learning/real-time-phase-reader-integration/prior-homepage-out/`에 백업했다. 후보의4696파일을 모두 hash 대조한 뒤 로컬 `out/`에 설치했다. 기존4175서버의 Interactive Book·601카드 목차·새 단원 HTTP200을 확인했다. 첫 빌드가 잘못된 작업 폴더에서 시작되어 컴파일 중 중단한 기록도 보존했으며, 그 시점의 이전4612출력이 그대로였음을 별도로 대조했다.

이번에 시작한 후보 preview만 검증 후 종료하고 다른 서버는 보존한다. commit/push와 공개 도메인의 배포 성공은 구분한다. 실제 공개 배포는 해당 커밋의 GitHub Pages 결과로 별도 확인한다.

## 재현

```bash
node --import tsx --test scripts/learning-real-time-phase-reader-integration.test.ts
node --import tsx scripts/prepare-real-time-phase-production-reference.ts
REAL_TIME_PHASE_PREVIEW_BASE=http://127.0.0.1:4205 node --import tsx scripts/check-real-time-phase-reader.ts
REAL_TIME_PHASE_PREVIEW_BASE=http://127.0.0.1:4205 node --import tsx scripts/check-real-time-phase-production-assets.ts
REAL_TIME_PHASE_PREVIEW_BASE=http://127.0.0.1:4205 node --import tsx scripts/check-real-time-phase-reader-maximum.ts
```

`REAL_TIME_PHASE_EXPORT_ROOT`로 실제 서버의 출력 폴더를 지정할 수 있다. 기본 후보는 `/tmp/suanlab-real-time-phase-candidate-20261111/out`이다. 정상 재현에서는 Node 기준을 먼저 만든 뒤 생산 Worker를 실행한다. 완료된 native 기록만 재대조하려면 `REAL_TIME_PHASE_COMPARE_RECORDED=1`을 사용한다. 증거는 `.runtime/learning/real-time-phase-reader-integration/verification.json`이다.

다음은 최종23요건의 근거·범위·한계 감사와 Brownian/유클리드/안장점/실시간 네 단원의 원주제 대조다. 단원을 등록했다는 사실만으로 원주제나 전체 교육과정의 완료를 선언하지 않는다.
