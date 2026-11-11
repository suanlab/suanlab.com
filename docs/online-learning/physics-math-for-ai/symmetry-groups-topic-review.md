# 군·대칭·표현론 원주제 대조

2026-11-11 기준. `symmetry-groups`의 10개 원주제를 기존 한영 원고·계산·실습과 대조했다. 기본 설명·연습·제어 범위는 보존하면서 유한 군부터 Lie 군·gauge·tensor 표현까지 증명과 독립 예제를 보강했다. 누적 대조는 22영역·215개 원주제이며 전체 51영역·509개 이론과 연구 심화 완료를 뜻하지 않는다.

## 원주제와 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`, 계산은 `src/lib/learning-symmetry.ts`다.

| 원주제 | ID | 보강한 근거 |
|---|---|---|
| 군·군 작용 | dihedral-group-actions | Burnside 이중 계수 증명, 16개 꼭짓점 색칠의 6개 궤도와 불완전한 색 개수 불변량 |
| Invariance·equivariance | invariance-equivariance-maps | SO2 방사형 vector 법칙·원점 연속성·반사 조건, 작은 비등방의 해석적 commutator |
| 표현·불가약 표현 | cyclic-real-irreducibles | 평균 내적의 불변 보완공간, 정확 사영 E0·E2·Ecs와 실수 Schur 조건 |
| Lie group·Lie algebra | lie-brackets-commutators | Jacobi 전개의 취소, group commutator 삼차항과 중심 차분의 오차 차수 |
| SO3·SE3 | rigid-rotation-translation | 반직접곱·twist adjoint와 lever arm, spatial/body frame의 관계 |
| 생성자·지수사상 | generators-twist-exponentials | 최소다항식과 적분 Jacobian 유도, singular value·full-turn rank·log branch 구별 |
| Gauge 도입 | local-gauge-transport | 비가환 SO3 loop의 정확 켤레 변환, spanning tree gauge와 잔여 loop 정보 |
| 대칭과 제약 | reynolds-symmetry-constraints | 다른 input/output의 intertwiner 평균과 affine bias 고정 조건 |
| Invariant feature | gram-invariants-chirality | 퇴화 rank에서 SO3/O3 궤도의 합치, 최소 특이값과 orientation 잡음 margin |
| Tensor representation | symmetric-tensor-representations | 3⊗3=1+3+5 분해·직교성·5차원 불가약성 논증과 axial vector의 반사 parity |

## 독립 예제와 그림

표준 라이브러리 Python 10종을 한영에 동일하게 제공한다. 색칠은 모든 16개 구성과 여덟 군 작용을 열거하고 Burnside 평균과 직접 궤도 수를 비교한다. 표현 사영과 affine 제약은 Fraction으로 정확 계산한다. Lie commutator는 Rodrigues를 재사용하지 않고 행렬 지수 급수를 합산해 한쪽·중심 quotient를 비교한다. Adjoint와 비가환 loop는 정수 행렬 곱으로 부호·순서를 확인한다. Gram 예제는 정확 유리수로 conditioning을 분리하며 tensor는 Fraction 분해·직교성·반사 법칙을 확인한다.

SE3 예제는 100자리 Decimal로 정확 binary64 입력의 sine/cosine을 합산한다. 2π−2⁻⁴⁸의 횡병진은 약 (−6.0441×10⁻¹⁶,1.1477×10⁻³⁰,0)이다. 원고의 명목상 정확 quarter/full-turn과 실습의 cardinal-angle 관례를 별도로 명시한다. 유한 계산을 전칭 정리의 증명 대신 사용하지 않는다.

한영 SVG 네 개는 여섯 색칠 궤도와 고정 구성 수, full-turn 원점 궤적과 J의 특이값을 보여준다. 폭 440, 접근성 title·desc·본문 대체 텍스트를 제공하고 인접 표와 코드로 값을 재현할 수 있다. 새 확장 예제는 기존 실습에 추가된 제어 기능을 뜻하지 않는다.

## 수치 수정

기존 행렬곱 차이로 계산한 equivariance error는 a=1,b=0,c=10⁻²⁰,θ=π/2에서 c를 잃고 0을 보였다. 현재는 해당 모형의 정확한 commutator 크기 √5|c sinθ|를 안정적으로 계산한다. 표시한 행렬과 두 곱은 별도의 부동소수점 평가이며 작은 c를 잃을 수 있다는 설명을 실습과 한영 원고에 추가했다. 모든 각도 등변 조건 c=0은 바꾸지 않는다.

기존 θ=10⁻⁴ 경계의 계수는 B≈0.49999999696,C≈0.16666666148로 소거 오차가 있었다. θ<0.1의 급수를 더 높은 항까지 적용하고, 나머지 범위의 B에는 반각 공식을 쓴다. 병진은 작은 각도에서 기존 J 다항식을, 그 밖에서는 회전축 분해를 사용한다. 한 바퀴 부근에서 항등과 상쇄해 작은 횡성분이 사라지는 문제를 줄인다. 정확 full-turn의 평행 성분과 기존 임의 3차원 지수·미분·군 합성 검산을 유지한다.

브라우저 경계 검사에서는 계수 변경 후 영 생성자에서도 K=R R⁻¹의 roundoff가 h⁻²로 확대되는 회귀를 잡았다. 생성자 하나가 0이면 K를 정확한 항등으로 계산하도록 수정하고, 여러 h에서 K·quotient 오차가 정확히 0인 테스트를 추가했다. 수치 허용 오차를 늘려 검사를 우회하지 않는다.

## 근거·보존·후속 범위

유한 표현의 분해·체 조건은 [Etingof 등의 저자 강의 노트](https://math.mit.edu/~etingof/replect.pdf), rigid motion의 frame·지수는 [Lynch·Park의 Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-3-exponential-coordinates-of-rigid-body-motion/)를 기준으로 대조했다. [Cohen·Welling의 원 논문](https://proceedings.mlr.press/v48/cohenc16.html)과 [Cohen 등의 gauge 논문](https://proceedings.mlr.press/v97/cohen19d.html)은 전역·국소 대칭 설계의 연구 근거이며 이 교재의 작은 행렬 예제가 그 전체 네트워크를 구현했다는 뜻은 아니다.

새 색칠·급수·정수 행렬·rank margin 계산과 명시적 논증은 교재의 독립 예제다. 변경 원고 20개의 과거 fixture SHA와 현재 SHA를 보존 ledger에 기록하고 네 immutable 기준 파일을 바꾸지 않는다. 허용한 현재 내용 이외의 변경은 계속 거부한다.

일반 compact/noncompact 표현, Peter–Weyl 정리의 전체 증명, 고차 Clebsch–Gordan 계수·비가환 연속 gauge field·대규모 steerable kernel과 이산화 오차의 연구 검증은 후속 범위다. 전체 교육과정 완료 표시는 유지한다. 다음 영역은 볼록해석과 제약 최적화의 10개 원주제다.

## 검증

- 선택 계산·한영 렌더링·안내 예제·원고 보존 테스트 65개 통과. 전체 CI 테스트의 통과를 뜻하지 않는다.
- Python 20개 블록의 실행 출력·SHA를 확인하고 한영 수식·제목 수·언어 순도를 대조했다. 기존 20개 원고는 추가 절 앞뒤로 원문 그대로 보존했다.
- 100자리 Decimal 4×4 지수 급수와 54개 임의 축·각도·시간 사례를 대조했다. 관측 최대 절대 오차는 회전 4.44×10⁻¹⁶, 병진 1.11×10⁻¹⁶이었다. 작은 비등방·이전 계수 경계·full-turn 횡성분·영 생성자의 별도 회귀 테스트를 통과했다.
- `npm run typecheck`, `npm run lint`, `npm run validate:content` 통과. 영 생성자 수정 후 다시 확인했다.
- Node 20 격리 후보의 `npm run build` 통과: 1,893개 경로 생성. `npm run check:export`로 HTML 1,888개의 내부 링크·자산·앵커·사이트맵·canonical 검사 통과.
- `npm run check:symmetry` 최종 실행 통과: 한영 320/768/1440px, 영어 다크 모드, 경계값·접근성 등 96개 상태와 동작 12개 묶음. 안내 예제 30개·퀴즈 20개, 설정 저장·언어 전환·정수 초안/Escape 복구, 기존 16개 계산 회귀를 포함한다. Axe WCAG 위반과 수집된 페이지 실행 오류는 0개다. 최초 영 생성자 실패 기록과 수정 후 성공 기록을 구별한다.
- 최종 정적 출력의 한영 그림 네 개를 320px에서 캡처했다. 소스 1,882개·학습 원고 1,300개·반영 파일 31개·설정 8개의 후보 byte 일치와 네 SVG 출력 byte 일치를 확인했다.
- commit/push와 공개 배포는 구분한다. 공개 홈페이지 완료 여부는 해당 GitHub Pages 작업 성공으로 별도 확인한다.
