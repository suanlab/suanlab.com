# 양자정보 아홉 가족의 Worker 연결과 브라우저 검증

2026-11-09. 계산 Worker·독립 검증 Worker·수명주기 controller·React hook을 준비했다. 아홉 가족과 두 주요화 모드를 실제 module Worker에서 확인했다. 학습 화면·시각화·reader 등록/생산 자산 검증은 아직 남아 있다.

## 계산과 검증의 분리

계산 Worker는 기존 정확 계산 진입점을 사용한다. 독립 검증 Worker는 원래 요청과 응답을 소유한 뒤 아홉 가족의 검증 진입점으로 대조한다. reader의 controller/hook에는 BigInt·수치 계산기·독립 수학 검증기가 들어가지 않는다. 번들 그래프로 분리를 확인했다.

모든 frame은 버전·양의 안전 정수 requestId·원래 입력/key·family를 확인한다. header 통과를 수학 검증으로 취급하지 않는다. 계산 결과는 별도 검증 Worker의 receipt를 받은 뒤에만 ready가 된다. 수학적으로 같은 분수라도 원래 raw 입력이 달라지면 다른 요청이다.

## 수명주기와 화면 귀속

각 실행은 계산/검증용 port를 소유한다. 입력 교체·취소·완료·오류·해제 시 handler를 제거하고 Worker를 종료한다. 이전 callback과 순서가 뒤바뀐 frame은 새 결과를 덮어쓰지 못한다. 계산/검증 전체에 기본30초 deadline을 두며 오류 뒤 재실행할 수 있다.

큰 응답의 두 번 복사는 기존 bounded descriptor-only 분할 복사를 사용한다.4ms/최대1024node 작업마다 양보하고 token 교체 시 중단한다. 이 예산 안에256표본의 큰 BB84 분수·64복사본 rank·256 KL 순서쌍이 들어감을 검사했다. getter·순환·구멍 배열·과도한 응답은 거부한다.

hook은 렌더링 중에도 현재 input key/enabled에 귀속되지 않는 이전 결과를 숨긴다. visibilitychange에서 비가시 상태가 되면 즉시 취소하고 다시 가시 상태가 되면 현재 입력을 재계산한다. 초기 hidden 상태의 실행도 막는다. SSR은 Worker를 만들지 않는다.

## 실제 검증 범위

새78검사는44예제의 envelope/독립 receipt와9가족 controller, 교체·복사/검증 중 취소·늦은 응답·실패 factory·message 오류·deadline·재시도·소유권·SSR·번들 분리를 포함한다. 관련 회귀8,968검사·타입·대상 lint를 통과했고 기존 정적 출력4,419파일과 이전 다체계 생성 기록11파일을 유지했다. 관련 회귀 결과와 타입/lint는 `.runtime/learning/quantum-information-workers/verification.json`에 기록한다.

실제 Chromium module Worker에서는44예제와 최대 입력3개, 총47입력의 정확 구간/분류를 CLI와 대조한다. 단순 extra field가 아니라 정상 모양의 잘못된 정확 계수 또는 전체 metadata를 맞춘 거짓 관측값으로 바꾼47응답도 거부한다. controller에서도 수학 변조를 거부하며 정상9가족·최대 입력·교체·양 단계 취소·두 deadline·시작 실패/재시도를 확인한다.

Node/Chromium의 libm 차이로 `value`/`logMagnitude` 표시 숫자의 bit 동일성을 요구하지 않는다. 이 두 값은 해당 브라우저의 formatter로 검증하고, 정확 bounds·sign·resolution과 나머지 필드는 CLI hash와 일치해야 한다.

별도 React harness는 실제 factory URL을 사용하는 Worker로 이전 결과 숨김·계산/검증 중 visibilitychange 취소·재개·retry·unmount를 확인한다. 가시성 속성과 이벤트를 제어한 테스트이며 실제 background 탭의 timer throttling 검증으로 확대하지 않는다. 두 harness에서 생성한46Worker가 모두 종료됐다. 입력 타이핑과 heartbeat를 함께 검사했으며 최대 간격은 native 기록에 남긴다. 이는 production reader의 성능 보증이 아니다.

## 다음 순서

아홉 교수 화면·대표 예제·자유 실험·시각화·개념/예측·안내·퀴즈·저장을 작성하고 실제 Interactive Book에 등록한다. 그 뒤 한영/명암/모바일·접근성·Three.js 대체/해제와 Next 생산 Worker·검색·정적 출력·실제 읽기를 검증한다.

575등록 단원·1726안내·입문310/362검증·52미완료, 전체51영역509주제·영역 감사·연구 깊이의 남은 범위를 유지한다. 기존 홈페이지 출력은 유지하며 commit/push·공개 배포는 하지 않았다.
