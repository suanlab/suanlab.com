# 다음 연구 보강: 응력 관측의 식별성과 변형장의 적합성

2026-11-13 설계 기록. 집필 전에 기존 두 단원의 한영 원본문과 통합 보고서를 대조했다. 기존 응력 회전·체적력·Green 변형률·작은 회전·정확한 affine 지도를 보존했다. 아래 설계를 한영 4원고·4그림에 반영했고 [최종 검증](continuum-stress-research-status.md)을 마쳤다. 이미 존재한 입문 내용을 미집필로 다시 세지 않는다.

## 첫 단원: 방향별 스칼라 관측에서 응력 복원하기

같은 점의 대칭인 평면 응력을 미지수로 두며 선택한 모든 관측은 같은 상태를 읽는다고 선언한다. 새 관측은 전체 traction이 아닌 법선 응력만이다. s=(p,d,tau), sigma=[[p+d,tau],[tau,p−d]]로 두면 m(theta)=p+d cos(2theta)+tau sin(2theta)이다. 각 값은 Pa, 잡음 공분산은 Pa² 단위다. 이 p는 압축 양의 유체 압력이 아니라 두 평면 응력의 평균이다.

측정 행렬 A의 행은 (1,cos2theta,sin2theta)이며 세 다른 축이 충분한 경우와 같은 축을 반복하는 경우를 구분한다. 잡음이 없다면 역행렬로 복원하지만, 유일성과 안정성은 다르다. 독립·동일 분산 잡음과 이 매개변수의 Euclidean 비교 척도를 선언하고 Cov(s_hat)=sigma_noise²(AᵀA)^−1을 유도한다. Frobenius 응력 노름은 sqrt(2) 곱하기 s의 Euclidean 노름이어서 이 비교가 특정 성분만 과도하게 축소하지 않음을 설명한다.

3개 독립 관측의 최선 조건수 하한은 sqrt(2)다. 첫 열 노름 제곱이 3이고 남은 두 열의 제곱합이 3이므로 최대 고윳값은 3 이상, 최소 고윳값은 3/2 이하이다. 축 (0,pi/3,2pi/3)는 Gram=diag(3,3/2,3/2)로 하한을 이룬다. 이 주장은 위 잡음·변수 척도와 관측 수에 대한 것으로 모든 센서 비용·상관 잡음·장치 최적성으로 확장하지 않는다.

군집한 축 (0,delta,2delta), 0<delta<pi/2에서는 det A=4 sin²(delta) sin(2delta)이며 작은 delta에서 영이 아니어도 불안정해진다. 중앙 축을 기준으로 회전한 변수에서는 2x2 Gram block [[3,1+2c],[1+2c,1+2c²]], c=cos2delta, 나머지 고윳값 2sin²2delta다. block determinant=2(1−c)²=8sin⁴delta를 이용해 작은 고윳값을 큰 값의 차로 빼지 않고 계산할 수 있다. 정확한 delta=0은 순위 상실이며 작은 양의 delta와 다르다.

한 방향의 전체 traction만 있으면 lambda m mᵀ 방향의 응력은 관측되지 않는다. 두 비평행 법선의 전체 traction과 세 스칼라 법선 관측은 다른 정보다. 균일 응력의 주축 비유일성과 센서 행렬의 순위 상실도 다르다. 복원한 응력이 변위·물성·경계 준비까지 결정한다고 주장하지 않는다.

## 둘째 단원: 양의 국소 척도가 실제 평면 변형장이 되는 조건

기존 affine 가족은 F가 상수라 적합성이 자동으로 성립한다. 보충 예에서는 기준 좌표의 양의 등방 척도 C=exp(2phi) I와 Green 목표 E=(C−I)/2를 공간 함수로 지정한다. orientation-preserving 평면 C2 지도 chi가 있다면 F=grad chi=exp(phi) R(theta)여야 한다. 각 점의 양의 C가 실제 적분 가능한 F를 보장하는 것은 아니다. 3차원 곡면 임베딩이나 성장·결함을 포함한 다른 물리 모형으로 바꾸지 않는다.

F 두 열의 혼합 미분을 같게 두면 theta_X=−phi_Y, theta_Y=phi_X가 나온다. 다시 혼합 미분을 교환하면 Delta phi=0이 필요하다. 또는 닫힌 작은 루프의 dtheta 적분은 0이어야 하는데 Green 정리로 면적에 대한 Delta phi 적분을 얻는다. 한 각도의 2pi 모호성으로 국소 결함을 숨기지 않으며 단순 연결 작은 영역의 연속 각도 lift를 선언한다.

phi=alpha(X²+Y²)/(2L²)는 양의 C를 주지만 Delta phi=2alpha/L²다. alpha가 0이 아니면 평면의 어떤 국소 회전장으로도 이 척도를 적분할 수 없다. [-L,L]²의 각도 결함은 8alpha이며 임의의 작은 부분 영역에서도 결함이 남는다. 중앙의 E=0 한 점은 이 공간 장애를 보지 못한다. 작은 변형의 목표 epsilon=phi I에서도 선형 Saint–Venant 조합이 Delta phi이지만 이것을 일반 유한 Green 변형률의 선형 적합성 공식으로 오용하지 않는다.

대조로 phi=alpha X/L는 harmonic이다. theta=alpha Y/L, chi=(L/alpha)(exp(alpha X/L)cos(alpha Y/L)−1, exp(alpha X/L)sin(alpha Y/L))는 정확한 국소 지도이며 FᵀF=exp(2alpha X/L)I, det F>0이다. alpha=0은 별도 항등 지도 극한이다. 충분히 넓은 Y 영역에서는 2pi L/|alpha| 주기로 같은 현재 위치가 반복되므로 국소 적합성과 양의 Jacobian도 전역 단사성을 보장하지 않는다. 이것을 국소 지도 자체의 실패로 오인하지 않는다.

7개의 symbolic 항등식으로 점프 없는 혼합 미분·사각 루프·conformal gradient·양의 determinant·측정 행렬 determinant·최선 Gram·안정한 작은 고윳값 인자를 확인했다. 추가 독립 수치/유리수 참조, 한영 본문·그림·실제 reader와 원문 보존 검증을 준비한다. 이 설계만으로 새 단원 완료를 집계하지 않는다.
