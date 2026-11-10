# 양자 상태·Hilbert 공간 집필·구현 계약

2026-11-08. 비평형 입문 점검 다음 주제다. 등록550단원·1650안내, 원래300/362입문 검증·62미완료를 유지한다. 전체51영역509주제와 분량·주차 제한 없는 연구 깊이를 계속 진행한다.

| 순서 | 계획 ID | 역할 |
| --- | --- | --- |
| 1 | `complex-state-rays-basis-amplitudes` | 복소 내적·정규화·ray·전역/상대 위상·정규직교 기저·Gram 기하 |
| 2 | `wavefunction-l2-projection-completeness` | L² 상태·거의 모든 곳의 동치·확률밀도와 구간 확률·완비성과 기저 절단 |

두 가족은 입문을 연결하는 경로다. C² 계산만으로 일반 Hilbert 공간·무한 차원·비분리 공간을 실험했다고 세지 않는다. 유한 다항식 절단으로 완비성이나 모든 점 수렴을 증명했다고 세지 않는다.

## 내용과 실험의 계약

첫 가족은 비영 정수 복소 벡터의 정규화된 대표와 ray를 다룬다. 내적은 첫 인자에 켤레를 취한다. 선택한 정규직교 기저로의 수동 좌표 변환과 물리적 상태 변화를 구분한다. 같은 한 기저 확률과 같은 상태는 다르다. 정규직교가 아닌 기저의 계수에는 Gram 행렬이 필요하다. 고정 기저 확률·fidelity·대수 잔차는 정확 유리수, 정규화된 진폭은 방향성512bit 제곱근 구간이다. 혼합 상태·일반 측정·시간 진화는 뒤 주제에서 다룬다.

둘째 가족은 u=x/L의 구간별 복소 affine g와 ψ(x)=g(u)/√(L∫|g|²du)를 다룬다. 확률은 샘플 합이 아니라 정확 다항식 적분으로 계산한다. shifted Legendre 기저의 유한 사영은 포착한 norm을 유지한다. 재정규화한 근사는 별도이며 영 사영에서는 미제공이다. L² 상태와 대표 함수의 점 값, Hamiltonian의 정의역·벽 경계조건을 합치지 않는다. 확률밀도와 진폭의 길이 단위도 명시한다.

## 독립 근거와 등록 조건

생산 복소 유리수 계산은 별도 SymPy 복소 행렬로, 생산 Legendre recurrence/Horner 적분은 독립 이항 계수 전개·단항식 적분으로 검산한다. 한영 설명은 가정·직관·수식·예제·풀이와 AI 연구 연결을 갖춘다.

이후 입력/Worker → 개념 비교·자유 조작·안내·퀴즈·모션·저장 복원 → **도서 → Interactive Book** 실제 읽기 순서로 연결한다. 한영·명암·모바일·키보드/터치·접근성·정적 출력까지 확인한 뒤에만 원래 입문 완료 수를 갱신한다. Online Book 목록에 새 교재 카드를 넣지 않는다.

다음 원래 주제는 관측·연산자이며 unitary evolution·Schrödinger, spin·angular momentum, density matrix, uncertainty·measurement, tensor product·entanglement, variational principle, many-body·tensor network, quantum circuit·channel, quantum information으로 이어진다. 연산자 정의역·스펙트럼·rigged Hilbert space·혼합 상태와 연구 심화 범위도 유지한다.

2026-11-08 교수 화면 갱신: [두 가족의 Worker·개념·자유 실험 검증](quantum-state-interface-status.md)을 마쳤다. 준비21개·개념 단계6개·Bloch 3D/2D·진폭/밀도·정확 질량·모션·저장과 오류 복구를 연결했다. 다음은 Interactive Book 읽기 등록·목차·표지·퀴즈·검색과 실제 Next.js 읽기·정적 출력 검증이며 입문 완료 수는 유지한다.

## 등록 검증 결과

2026-11-08 최신: [상태·Hilbert 공간 두 단원의 실제 교재 검증](quantum-state-integration-status.md)을 마쳐552단원·1656안내, 원래301/362입문 검증·61미완료다. 전체9879검사·타입·lint·콘텐츠·빌드·1790 HTML, 실제 읽기16조합·접근성16조합·3D8회·WebGL 복구2회와 키보드/터치·Worker 복구를 확인했다. 다음은 관측·연산자다. 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다. 아래는 이전 기록이다.
