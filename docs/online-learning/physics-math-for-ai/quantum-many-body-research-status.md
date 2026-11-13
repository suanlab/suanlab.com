# 유한 spin 응답·Majorana 공분산 연구 보강

2026-11-13. 기존 spin chain·fermion Fock 두 단원의 한영 4원고에 표시 수식 32개, Python·4행 표 각 4개, 연구 해설 문제 12개와 그림 4개를 추가했다. 기존 본문과 풀이 28개를 정확히 보존했다. [검증 근거](quantum-many-body-research-evidence.json)와 [자료·가정 대조](quantum-many-body-research-source-audit.md)를 따른다.

## 유한 gap과 정적 응답

`finite-spin-chain-local-hamiltonians`는 두-spin Bell 섹터를 완전히 풀고 ordered gap·다음 distinct level gap·국소 상태·전체 susceptibility를 구별한다. 종방향 장의 비가환 극한과1차 응답을 유도하고 전체/국소 인자·단위·고립 조건을 확인한다. 유한계 계산을 열역학적 임계점이나 무한계 대칭 깨짐의 증거로 확대하지 않는다.

## Gaussian 가정과 zero 연산자의 의미

`fermion-fock-parity-operator-algebra`는 Majorana·Γ의 부호와 quadratic 에너지/전개를 유도한다. Γ의 전파와 Gaussian 상태 복원을 분리하고, 같은 Γ를 갖는 cat·최대 혼합의 고차 모멘트 반례를 계산한다. 두 모드의 odd zero 연산자·짝 바깥 결합 splitting·parity superselection과 density interaction의 closure 실패를 구별한다. 긴 wire의 위상 보호를 이 작은 예제로 인증하지 않는다.

## 검증

- CI 선언 Python 3.12 격리 환경에서 관련 11파일·285검사가 통과했다. 한영 수식·코드·표가 같고 영어 한글 잔류·KaTeX 오류가 없다. 연구 절 제거 후 원본문 hash·풀이 28개를 대조했다.
- Spin spectrum·부분추적·전이 susceptibility 15준비와 종방향 ±장 45쌍을 독립 대각화했다. Gaussian thermal/ground 48준비·전체 quadratic unitary 96준비, 두 모드6준비·상호작용9준비를 CAR 행렬과 직접 비교했다. Γ가 같은 한 쌍의 비교 상태에서 고차 모멘트 차이를 확인했다.
- SymPy는 유리수 spin root 4경우·Majorana characteristic polynomial 4경우를 확인했다. 유한 차분이나 부동소수점 잔차는 방향성 구간 인증이 아니다. 독립 작은 예제의 ground 계산을 기존 일반 off-diagonal solver로 표시하지 않는다.
- 빌드·타입·콘텐츠·정적 출력 검사는 실제 종료 코드 0이다. 현재 한영 렌더링 4편과 제공된 2 HTML·4 SVG 바이트를 대조했다.
- 한영 320·768·1440px 밝은 화면과 1440px 어두운 화면 16상태, 모식도/해설·안내·퀴즈 각 6회와 저장·완료 복원 2그룹을 확인했다. 실제 계산 완료·현재 입력·에너지/분산을 함께 비교했다. 접근성 위반·페이지 오류가 없고 브라우저를 종료했다.
- 원본 그림 4개·정상 읽기 모바일 화면 4개를 직접 검토했다. SVG/캡션 경계·목차 앵커와 긴 코드/표/수식의 키보드 스크롤도 확인했다.

## 보존과 다음 순서

기존 일반1…4 spin/mode 조작과 비대각 ground 미계산 범위를 유지한다. 새 spectrum·response·Gaussian·Majorana 예제는 독립 유도이며 새 UI 기능이 아니다. 검토 원장 931항목을 보존하고 4편을 더했다. 기존 650단원·1,969안내·51영역·509원주제·입문 362/362 완료를 유지한다.

양자 순차 연구 대조는 **18/34단원**이다. 다음은 `boson-fock-cutoff-boundary-algebra`와 `matrix-product-state-contraction-gauge`다. 전체 **45영역·447주제 대조 / 6영역·62주제 후속**은 유지한다. 영역 종합 대조·교육과정 완료는 false이며 공개 배포는 별도로 확인한다.
