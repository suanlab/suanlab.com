# 국소 질서변수: 시각화와 안내 실험 검증

2026-11-11. `microscopic-blocks-and-local-order-parameters`는 **등록 전 준비 중**이다. 수학 기반·한영 원고·독립 module Worker에 이어 격리된 React 화면을 검증했다. 실제 reader와 Next 생산 Worker 자산 검사가 남아 등록602단원·검증 원주제316/362·미완료46개를 유지한다.

## 새로 만든 화면

- `src/components/learning/block-order-lab.tsx`: 공간·활동도·프로필·검사 쌍 편집, 입력 적용, 취소·재시도, 예측/해설과 문제를 연결한다. 적용 전 초안과 적용 모형의 결과를 구별한다.
- `block-order-concept.tsx`, `src/data/block-order-teaching.ts`: **6개 개념 그림·6개 안내 실험·8개 해설 문제**다. 평균 사상, 가중합, 전체 합과 공간 프로필, 경계 평균, 국소 기준, 조건부 의존성을 단계별로 설명한다. 고정 설명 그림과 직접 계산 결과를 구별한다.
- `block-order-result.tsx`: 미시 배치, 조건부 스핀 평균, 반복한 블록 평균, 전체 앙상블 평균의 네 모드를 같은 격자에서 비교한다. 선택 배치의 확률·전체 프로필 확률·전역 합 분포와 자유에너지/엔트로피 포함 구간을 제시한다.
- `block-order-spatial.tsx`, `block-order-spatial-engine.ts`: 기존 테마 토큰을 쓰는2D 지도와 lazy Three.js 막대다. 실제 공간 두 축에 사이트를 배치하고 높이는 고정 스핀 단위−1…+1을 유지한다. 카메라는 화면 비율에 맞춰 배치하며 높이를 입력별로 재정규화하지 않는다.

블록 테두리, 사이트 번호, 부호/숫자와 정확한 표를 함께 사용한다. 파선은 cross edge, 주황 edge는 선택 미시 배치의 불일치다. 평균장 자체에 이산 불일치를 부여하지 않는다. 주기 wrap은 추가 사이트가 아니다.

예제 배치는 가장 작은 numeric mask 최대8개이며 시간 진화·MCMC·무작위 표본이 아님을 표시했다. 평균은 모든 상태를 합한 검산 결과다. 재생·모드·3D 시점 변경으로 계산 Worker를 재시작하지 않는다. 동작 줄이기와 숨김 상태에서 재생을 멈춘다.

## 분포, 정확한 값과 저장

프로필625개를25행씩 페이지로 제시하고 작은 확률도 유지했다. 마지막625번째 프로필까지 적용해 입력과 결과가 일치하는지 확인했다.6자리 유리수 활동도를 동시에 사용한 큰 응답도 모바일에서 열어 확인했다. 큰 분수는 줄바꿈하고 필요한 표·그림만 내부 가로 스크롤을 사용한다.

로그 구간은 [계산 응답 검증](block-order-producer-verification.md)의512bit 표시·2048bit 독립 검사 결과다. 근사 소수와 정확한 유리수 양끝점을 구별한다. 이 단계에서 새로운 수학 계산을 reader에 옮기지 않았다. SSR과 실제 browser bundle graph에서 생산/독립 검산 엔진이 제외됨을 확인했다.

`learning-block-order-state.ts`는 입력·미완성 초안·학습 선택만 저장한다. 결과·영수증·실행·재생 플래그는 저장하지 않는다. 복원이 끝나기 전에는 첫 저장과 계산을 기다린다. 주어진 적용 입력으로 결과를 다시 계산한다. 잘못된 저장은 기본값, 저장 차단은 방문 중 실험으로 처리한다.

## 실제 확인한 범위

`scripts/check-block-order-visuals.ts`가 자체 localhost 서버와 browser를 만들고 기존 사이트의 정적 CSS를 사용했다. 한영×명암×320/1440px **8상태**에서 다음을 확인했다.

| 검사 | 결과 |
|---|---|
| 개념·안내 | 개념48회, 안내 해설48회 조작 |
| 예제·문제 | 9예제를 각 상태에서 계산; 문제64개의 정답·오답과 해설 확인 |
| 계산과 표시 분리 | 배치 재생의 실제 변화와 평균 유지, 추가 Worker 없음 |
| 그림 | 3D 열기·회전/Home·닫기,2D/정확한 값 유지, 화면 캡처 직접 확인 |
| 입력 | 부적절한 지수 표기 거절, 기존 적용 결과 유지, x=1 적용 후 P(k)=1/4 |
| 조건부 평균 | 한 블록 예의1/3,2/3,2/3,1/3을 실제 표에서 확인 |
| 언어·접근성 | 영어 한글 혼입0, 문서 넘침0, axe 위반0, pageerror0 |
| 수명 | 이8상태에서 생성한 Worker230개가 모두 한 번 종료; 활성0·중복 종료0 |

별도 context에서 동작 줄이기·미완성 초안 reload 복원·언어 전환·계산 취소/재시도·손상/차단 저장·WebGL 없는 대체 화면을 확인했다. 별도 context의 Worker 수는 위230개에 합산하지 않았다. 사용한 browser와 자체 preview 서버는 종료했고 이전 서버는 유지했다.

최초 화면 확인에서 모바일3D 끝점 잘림을 발견해 카메라를 화면 종횡비에 맞추었다. 깨지는 결합 수학 기호도 읽을 수 있는 문구로 바꾸었다. 수정 뒤 전체8상태를 다시 검사했다. 재생 검사도 다음 mask로 실제 변경된 뒤에 멈추도록 확인한다.

## 재현과 한계

```bash
node --import tsx --test scripts/learning-block-order-view.test.ts scripts/learning-block-order-state.test.ts
node --import tsx scripts/check-block-order-visuals.ts
```

새6개 테스트는 Node20·24에서 통과했다. 전체 TypeScript와 신규 파일 lint도 통과했다. `.runtime/learning/block-order-visuals/verification.json`, `source-hashes.json`,8개의 axe 결과와 screenshot이 근거다. 최종 QA 중 해당24개 소스 hash가 유지되었음을 대조했다.

격리 preview는 esbuild의 Worker URL adapter를 사용한다. 이를 실제 Next 생산 자산, 기존 홈페이지 reader, 등록 또는 공개 배포 완료로 표시하지 않는다. 다음 단계는 reader/표지/안내/문제 등록 준비, 원고 연결, 저장·언어·반응형·접근성·실제 Worker URL 검증과 단원 범위 감사다. 통계장론 원주제와 전체 교육과정 목표는 계속 진행 중이다.

후속 관찰: [실제 reader 통합](block-order-reader-integration.md)에서 Next 생산 자산과 등록 단원을 별도로 확인했다. 위 결과는 등록 전 격리 preview 범위의 기록이다.
