# 실시간 producer의 정밀도·전송 계약

2026-11-10. 물리 모형의 입력 범위 N2…32·12자리 유리수·크기16을 유지한다. 아래는 **producer 응답**의 수학적 높이/문자열 상계다. 독립 응답 검산·실제 Worker 기한/메모리와 reader의 완료 증거는 아니다.

## 정밀도와 수치 대상

chain/packet은8192bit 작업, Airy는4096bit 작업과1862bit Gamma 꼬리 경계를 쓴다. 관찰은512bit 격자다. 위상은1024bit 구간과 크기에 적응하는 π reduction을 사용한다. 입력을 Number로 바꿔 위상을 줄이지 않는다.

[정상 작용의 보수적 상계](real-time-phase-rational-size-bounds.md)는 |S*/ℏ|<2⁴⁵⁶⁸을 준다. raw rational phase primitive의65536bit 분자/분모 한도와 별개다. 실제 중심/국소 위상은8192bit에서 감싼 뒤 midpoint/radius로 전달하며 Lipschitz 반경을 유지한다.

s=m/δ에서2⁻⁴³<s<2⁴⁹이다. 직접 계수 형태 `H_ii=2mN/τ−mω²τ/N`를 쓰면 |H_ii|<2⁵¹, |H_off|<2⁴⁹이다. 조절자 ε>2⁻⁴⁰에서는 |det Cε|≥εᵈ>2⁻¹²⁴⁰, row norm bound로 |det Cε|<2¹⁶¹²다. 해당 norm²를8192bit에서 반올림해도 양의 하계를 잃지 않는다. 가장 작은 prefactor 제곱도 충분히 보수적으로2⁻³²⁴⁸보다 크다. 양의 pivot의 half-angle은 cos≥1/√2인 식을 사용해 뺄셈 상쇄를 피한다.

packet 공분산은 determinant ℏ²/4를 정확히 유지한다. 보수적인 map/초기 covariance 상계에서 최종 위치 분산은2⁻⁹⁹⁰보다 크고2⁹⁰⁸보다 작다. 따라서 정규화와 폭의 근호가 작업 격자에서0이 되지 않는다. 작은 위치 분산만으로 단일 고전 궤적을 판정하지 않는다.

Airy 중앙 경로의 |z|≤64와 별도 겹침 |z|≤72에서는 원점 상수 오차와 두 계열의 상쇄를 함께 감싼다. 큰 실수 인수 경로는 두 부호 각각의 첫 생략항을2⁻⁶⁴⁰이하로 줄인다. |z|가 커질수록 각 생략항이 감소하므로 경계 z=64가 항 수의 최악 경우다. 값·scaled 값·지수·log를 분리하며0을 포함한 진동 구간에는 log 보증을 붙이지 않는다. 스크립트는 |z|≤72의 절대 항비를373249/[(3j)(3j−1)]로 감싸 성장 곱<2⁶⁴⁰·2048번째 항<2⁻⁵⁰⁰⁰을 정확분수로 확인한다. 반올림 상계 U_next≤ratio·U+2를 더해도 종료 quantum보다 작다. 점근식은 z=64의256/257번째 생략항<2⁻⁶⁴⁰을 정확 검사해 모든 더 큰 실수 인수를 덮는다.

## 손실 없는 exact 자료와 관찰 격자

전체 H/역행렬은 continuant 인수로 지정한다. 조절자 pivot은 `p_j=Θ_(j+1)/Θ_j`로 모두 복원되므로 큰 분수를 중복 전송하지 않는다. 각 ratio sqrt의 양의 실수부 branch를 명시하며 determinant의 주 sqrt로 바꾸지 않는다. 전체 complex mean과 exponent도 보존한다.

실수 inverse와 복소 연결 응답의 모든 성분은 추가로 대칭 upper rows에 감싼다. 이는 complex conjugate 대칭이 아니다. 각 정수 끝점 c의 물리적 의미는 `c·2^(scalePowerOfTwo−512)`다. 공통 척도는 행렬 전체에 대한 절대 격자이며 작은 각 성분에512bit 상대 오차를 약속하지 않는다. exact 인수는 계속 보존하고 작은 비영 성분을 exact0으로 표시하지 않는다. kernel/packet의 복소 값도 공통 이진 척도를 가진다. 큰 Abel 감쇠는 별도 지수와 log로 보존한다.

## exact 높이 상계의 유도

각 입력 분자/분모는2⁴⁰보다 작다. Cε의 공통 분모를 `L=m_d τ_n ω_d τ_d N ε_d`로 잡으면 L<2²⁰⁵이며 모든 계수가 Gaussian integer/L이다. row norm은2⁵²보다 작다. Hadamard 부등식으로 k차 minor numerator의 복소 크기는2^(257k)보다 작고 denominator는 Lᵏ을 나눈다.

`Cε⁻¹=L adj(P)/det(P)`를 conjugate로 유리화한다. forcing의 공통 분모는 `F_D=m_d τ_n a_d b_d<2¹⁶⁰`, 각 forcing numerator는2²¹⁴보다 작다. 두 끝점의 항만 합친 complex mean의 각 분자/분모는2¹⁶¹²⁸으로 감쌀 수 있다. 완전제곱 exponent는 끝점 상수와 ℏ를 포함해2¹⁶⁸⁹⁶으로 감싼다.

실수 continuant는 k차에서 numerator2^(141k), denominator2^(130k)로 감싼다. `q_i=(θ_(d−1−i)a+θ_i b)/θ_d`의 분자/분모는2⁴⁶⁰⁸보다 작다. S*는2⁵¹²⁰으로 감싼다. 정확한 초점에서는 inverse 대신 family/null vector를 보존한다.

고정 시간 정련은 최대256구간이다. Chebyshev/continuant 표현에서 각 map entry의 numerator/denominator를 각각2^(148n+120),2^(137n+130)으로 감쌀 수 있다. 반복 행렬 곱의 미약분 중간 분모를 최종 serialization 높이로 오인하지 않는다. 기본 map이 refinement 목록에 다시 나타나는 비용도 계산에 넣는다.

## 실행 가능한 필드 집계

```bash
python3 scripts/learning/verify-real-time-producer-bounds.py
```

기약 분수의 문자열 상계는 `digits(2^numeratorBits−1)+digits(2^denominatorBits−1)+2`로 세며 slash/음수 부호를 항상 포함해 과대 계상한다. complex mean62성분, continuant/거리 power/기본·정련 map, 두 packed 행렬의496개 대칭 성분, stationary/family 자료와 진폭·위상 필드를 합친다. matrix 끝점은512bit 계수와 여분 padding을 포함해 부호당156자 이하이다.

스크립트는 chain 전체 문자열을180만자 미만, 단일문자열24000자 미만으로 감싼다. packet은 공통 map denominator2⁴⁶⁸⁶과 covariance denominator2⁹⁷⁷²·numerator2¹⁰⁶⁸⁰,65표본의 좌표/위상/값/밀도를 포함해75만자 이하로 감싼다. fold는 |z|<2³¹,ζ<2⁴⁶과 고정 필드 수로24000자 이하이다. 구조 상계는 node10000·depth12·최대 배열65다.

기존 한도인 총문자열2000000·단일65536·node100000·depth32·배열256을 변경하지 않았다. source hash와 실제 최대/경계32입력의 snapshot 측정을 별도로 보존한다. 이 상계는 코드 shape/인수 표현이 바뀌면 다시 감사해야 한다. 실행 시간은 입력별 실제 관찰과 구별하며 아직 Worker deadline을 증명한 것은 아니다.
