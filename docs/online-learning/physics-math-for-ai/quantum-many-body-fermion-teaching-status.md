# 다체계 두 번째 학습 화면: Fermion Fock·parity

2026-11-09. 두 번째 staged lesson에 상세 한영16절·17수식·4표·8해설 본문, 세 개념·여덟 고정 입력·세 안내·세 해설 퀴즈·입력/학습 저장과 계산/독립 검증 Worker를 연결했다. 실제 reader 등록·생산 Next 자산 통합은 남아 있다.

## 학습 내용과 그래픽

생성 순서의 상대 부호, 막힌 생성의 영벡터, hopping/pairing의 N·parity 보존 차이, parity 초선택과 형식적 순수 입력, twirl, 전체 Fock와 제한 sector의 바닥을 구별한다. 복소 hopping 위상과 네 모드의 긴 parity string도 실험한다.

2D/Three.js는 선택한 기저 벡터에 대한 두 연산 순서와 누적 부호를 보여준다. 원래 ψ의 복소 계수는 이 그림에 곱하지 않으며, 전체 중첩 작용·norm은 별도 표로 제공한다. 카메라와 안내 재생은 물리 시간을 바꾸지 않는다. [물리 가정·그림·입력 감사](quantum-many-body-fermion-teaching-audit.md)에 범위를 기록했다.

## 검증 결과

- 새11검사·기존107회귀·타입·lint 통과. 독립 bit-prefix oracle의2768경로와 여덟 손계산 기준 일치.
- Native 한영·명암·320/1440px·모션16상태와17접근성 상태 통과. 영어의 한글 잔류·문서 가로 넘침0.
- Same-mode CAR, 부호 반전, 영벡터/진공, 형식적 parity와 제한 바닥, sector 불일치 거부·입력/퀴즈/완료 reload·저장/Worker/WebGL 실패 복구 확인.
- 전체 run의 Worker74개 생성/종료 일치. 3D camera 확대 뒤 상호작용·저장·키보드·확장 접근성1상태를 다시 확인했고 추가42개 Worker도 일치.
- 어두운 select 대비 문제를 site token으로 고쳤고 같은 수정은 spin의 새 계 선택기에도 적용했다. Spin 관련9검사를 다시 통과했다.

근거: `.runtime/learning/many-body-fermion-teaching/`의 tests·regression·spin-theme-regression·native·native-interaction 로그, metrics·검증 JSON·dependency graph와 한영/3D 화면 캡처. Native는 실제 reader CSS와 별도 harness/Blob Worker이며, 실제 reader와 생산 Next Worker/Three 청크의 증거는 아니다.

## 다음 순서와 전체 범위

**Boson cutoff·coherent → MPS 수축/gauge → Schmidt 절단** 학습 화면을 준비하고 실제 reader·생산 자산·반응성·등록·정적 출력·한영·접근성을 통합 검증한다.

학습 화면 준비2/5·등록0/5다. 기존566한영 등록 단원·1698안내, 원래 입문308/362검증·54미완료와 전체51영역509주제·연구 깊이를 유지한다. 기존 홈페이지 출력·미리보기 서버와 **도서 → Interactive Book** 진입을 보존했다. Commit·push·공개 배포는 수행하지 않았다.
