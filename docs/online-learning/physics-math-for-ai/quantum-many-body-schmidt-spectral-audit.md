# Schmidt 응답용 독립 스펙트럼·로그 인증 감사

2026-11-09. `learning-many-body-schmidt-spectral-wire.ts`는 이후 전체 Schmidt 응답의 수학 귀속을 위한 인증 계산이다. 현재 여섯 번째 응답 검증기 자체와 실제 학습 화면 통합은 미완료다. 완료한 다섯 응답 종류와 구별한다.

## 고유값과 정확한 퇴화

입력은 차수1..4의 유리수 density characteristic polynomial이다. 정확한 유클리드 polynomial gcd와 square-free factorization으로 중복도를 먼저 분리한다. 각 factor에 대해

`(1+t)^d P((left+right*t)/(denominator*(1+t)))`

의 정수 계수 부호 변화를 센다. 양의 t는 열린 절단 구간과 일대일 대응한다. Descartes 부호 변화가0이면 근이 없고,1이면 정확히 하나다. 둘 이상이면 구간을 이분한다. 근 하나를 인증한 구간은 정확한 polynomial 부호로4096비트 폭까지 좁힌다. 이분점과 끝점0·1의 정확한 근은 선형 factor로 제거한 뒤 나머지를 다시 인증한다. 생산 코드의 Sturm 열·root isolator를 호출하지 않는다.

최종 근들은 [0,1] 안에서 엄격히 분리되며 중복도 합이 다항식 차수와 같아야 한다. 따라서 복소근·범위 밖 근·미분리 근을 정상 spectrum으로 반환하지 않는다. Polynomial coefficient는16384비트, 차수는4, 탐색 방문은100000, 깊이는4352로 제한한다. 일반 수치 고유값 tolerance로 가까운 두 근을 합치지 않는다. 10⁻¹⁸⁰ 간격의 두 근과10⁻²²⁰ 양의 근도 각각 분리해 검증했다.

## 로그의 독립 구간

양의 유리수 x를 x=2^k y,1≤y<2로 바꾸고 z=(y−1)/(y+1)을 사용한다. 다음 양의 급수600항과 나머지 상계를 직접 계산한다.

`ln(y) = 2 sum[j=0..m-1] z^(2j+1)/(2j+1) + tail`

`0 <= tail <= 2 z^(2m+1)/((2m+1)(1-z^2))`

1536비트 고정 소수점에서 각 항·곱·합·나머지를 바깥 방향으로 반올림한다. 같은 직접 급수로 ln2를 인증하고 k의 부호에 맞춰 범위를 더한다. 생산 로그 엔진의 argument reduction·상수·출력값을 가져오지 않는다. 합법 입력은 양의 분자와 분모 각각16384비트 이하다. 0·음수는 거부한다.

## 기준과 범위

기존 독립 SymPy/mpmath Schmidt26준비의65절단·129고유값·130엔트로피를 대조한다. mpmath1800자리 계산으로 별도 생성한34개 로그의520유효 숫자 기준을 사용한다. 이 기준은 새 generator로 byte-identical 재현한다. 정확 중복도·비이진 유리근·0/1 끝점·quartic 무리근·매우 가까운 근·작은 양의 근·역수·dyadic scaling·부호·근1의 로그도 검사한다.

근 cache64개·로그 cache256개는 정규화한 유리수 key를 사용하며 반환값은 중첩 동결한다. Eviction 후에도 같은 수학 결과를 반환한다. 번들 의존성에서 생산 Schmidt·root·Ritz·로그·행렬·Hilbert 엔진이 제외됐는지 확인한다.

이 인증만으로 projector·절단 벡터·오차·fidelity의 전체 응답을 검증했다고 볼 수 없다. 다음에는 독립 interval matrix 계산과 전체 schema·입력 귀속·변조 거부 검사를 연결해야 한다. 실제 reader·취소 controller·브라우저 반응성·생산 Worker 자산은 이후 별도 통합 대상이다.
