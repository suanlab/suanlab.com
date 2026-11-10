# 유한 장의 측도와 열 자유에너지: 수학 모형 계약

2026-11-11. `finite-field-measure-and-thermal-free-energy` 준비 계약. 계산·독립 검산 Worker는 격리된 Chromium에서 확인했다. 실제 reader·시각화와 Next 생산 자산 연결은 남아 있다.

## 공간, 계수와 측도

1차원 길이 n 또는 2차원 직사각형 nx×ny, 각 축 2 이상, 전체 32점 이하를 사용한다. 공간 차원 d, 고정 간격 a>0, cell 부피 v=a^d, 전체 부피 V=Nv다. open은 내부 edge만, periodic은 각 점에서 각 양의 축으로 가는 edge, dirichlet은 open에 각 경계면의 외부 영점 연결을 더한다. 길이2 주기 축에는 평행 edge 두 개가 있다.

`K=v[r I + κ L/a²]`, `J=v j`, `E0=V f0`, `E=E0+φᵀKφ/2−Jᵀφ`. κ≥0이며 r은 음수도 허용한다. 외부 source j는 에너지 밀도 계수다. r, κ, f0, j는 지정한 T>0에서 값·1차·2차 T 미분을 유리수로 받는다. 이는 국소 Taylor 경로를 지정하며, T 이외 a·경계·제약·기준 측도는 고정한다. 전 온도 구간의 안정성을 주장하지 않는다.

전체 R^N 또는 평균 영 초평면을 적분한다. 기준 측도는 T와 무관한 유도 Euclidean 측도를 φ_*^q로 나눈 것이다. q=N 또는 N−1, φ_*>0. 평균 영 basis B=[e1−eN,…,e(N−1)−eN]에서 A=BᵀKB, Gram determinant=N이다. 따라서 restricted determinant는 det A/N이지 det A가 아니다.

정확 LDLᵀ의 모든 pivot이 양수일 때만 정규화 가능하다. 실패 시 전체 R^q Gaussian은 정규화 불가능하다고 반환한다. 영 pivot만으로 전체 행렬이 singular라고 분류하지 않는다. 평균 영은 확률법의 변경이며 발산하는 원래 적분을 정규화한 결과가 아니다.

## 절대 정규화와 열 미분

`G=B A⁻¹ Bᵀ`, `μ=GJ`, `Cφ=T G`, `m=E0−JᵀGJ/2`.

`log Z=(q/2)log(2πT)−(1/2)log det K_restricted−q log φ_*−m/T`.

로그와 π를 유리수처럼 저장하지 않는다. 정확 출력은 이 기호식의 유리수 피연산자다. 숫자 표시용 구간은 아래 별도 계약으로 검산한다. Fourier 공액 쌍을 독립 complex 두 개로 적분하면 차원과 determinant 거듭제곱이 잘못된다.

고정 측도에서 `Athermal=E−T E_T`, `U=T²∂T log Z=〈Athermal〉`, `S=log Z+U/T`, `F=−T log Z`다. 혼동을 막기 위해 basis 정밀도 A와 열 관측량 Athermal을 구별한다.

`Cv=Var(Athermal)/T²−T〈E_TT〉`.

`Q=K−T K_T`, `b=J−T J_T`, `c=E0−T E0_T`로 놓으면

- U=c+tr(Q Cφ)/2+μᵀQμ/2−bᵀμ.
- Var(Athermal)=tr[(Q Cφ)²]/2+(Qμ−b)ᵀCφ(Qμ−b).
- 〈E_TT〉=E0_TT+tr(K_TT Cφ)/2+μᵀK_TTμ/2−J_TTᵀμ.

고정 미시 Hamiltonian일 때만 U=〈E〉, Cv=Var(E)/T²가 된다. 임의 온도 의존 유효 family의 Cv는 음수도 가능하며 이를 고정 Hamiltonian의 음의 정준 열용량으로 해석하지 않는다.

또한 기준 측도에 대한 Gibbs entropy는 `SG=log Z+〈E〉/T`다. 열역학 entropy `S=−∂T F`와는 `S=SG−〈E_T〉` 관계다. 온도 의존 유효 모형에서 두 entropy를 같은 값으로 표시하지 않는다. 로그가 들어간 entropy·F 전체를 정확한 유리수 값이라고 표시하지 않는다.

## 독립 검산과 필수 반례

생산 준비 core는 rational LDLᵀ·Gaussian 모멘트를, 독립 Python은 Fraction Gauss–Jordan·log determinant 미분을 사용한다. 행렬, determinant, μ, Cφ, m, 〈E〉, U, Cv를 전 항목 대조한다. 두 구현이 같은 열 관측량 모멘트 식을 복사하지 않게 한다.

- 고정 에너지: q 자유도, source에 의한 최소 에너지를 포함해 U=m+qT/2, Cv=q/2.
- K(T)=T I: normalized law가 일정하고 U=Cv=0이지만 〈E〉=qT/2.
- K(T)=T² I: U=−qT/2, Cv=−q/2.
- φ_*→cφ_*: log Z는 −q log c만큼 변하고 U,Cv는 그대로다.
- φ=c z: Kz=c²K, Jz=cJ, φ_*z=φ_*/c이면 Z가 같다.
- E0→E0+a+bT+cT²: 법은 같고 U는 a−cT², Cv는 −2cT만큼 이동한다.
- 평균 영 Gram 인자, 음의 r이 제약으로 안정화되는 예, periodic 영모드, 부정부호와 영 pivot, 2점 축의 edge 중복, 2차원 고정 경계, 비균일 온도 의존 source를 포함한다.

유리수 일치 검사는 연속체 극한·임계 지수·상호작용 적분 오차를 증명하지 않는다. 수학 기반 검증만으로 단원 완료나 홈페이지 반영을 선언하지 않는다.

## 입력·응답과 숫자 구간

입력은 명시한 열 개 필드와 각 jet의 세 필드만 받는다. 문자열은 부호 있는 18자리 이하 정수, 또는 분자·양의 분모 각각 18자리 이하 분수다. plain data descriptor와 밀집 배열을 읽고 별도 동결 복사본을 만든다. 분수는 기약형으로 정규화한다. 원천·배경의 1/2차 미분도 입력 key에 포함한다. 수학적으로 정규화 불가능한 입력은 잘못된 문법과 구별해 정상 응답의 `not-normalizable` 분기로 반환한다.

정상화 가능한 응답은 `exact`와 `observations`를 가진다. 후자의 log Z, F, 열역학 S, Gibbs SG는 각각 **2^-512 격자 위 유리수 양끝점**이다. 생산 산술은 1024bit 작업 정밀도에서 방향 반올림과 해석적 나머지를 포함한다. determinant는 고정소수점으로 바꾼 뒤 로그를 취하지 않고 정확한 분수에서 범위 축소한다. 따라서 2^-1024보다 작은 양수도 영으로 오인하지 않는다.

독립 검산은 별도 2048bit 산술과 다른 π 항등식으로 더 좁은 포함 구간을 만든다. 응답은 그 구간을 포함하고 폭이 6×2^-512 이하일 때만 통과한다. 양끝점은 canonical 분수여야 한다. 동일한 참값을 감싸는 다른 유효 구간도 허용하며, 모든 유효 구간의 byte 일치를 요구하지 않는다. Decimal 진단은 이 해석적 구간 인증과 별도다. 정규화 불가능 분기에는 숫자 관측을 넣지 않는다.

독립 exact 검산은 incidence·Gauss–Jordan·log determinant 미분을 사용하며 생산 LDL·열 관측량 모멘트를 호출하지 않는다. 모든 exact 필드, 요청 ID, 정규화한 입력 key와 응답 귀속을 확인한다. reader 측에는 큰 행렬·로그 계산을 넣지 않는다. 입력의 짧은 분수 정규화에 필요한 작은 BigInt gcd는 reader에서도 실행된다.

## Worker와 공개 연결의 경계

각 요청은 계산 Worker와 별도 검산 Worker를 소유한다. 검산 수락 전에는 ready를 공개하지 않는다. 교체·취소·오류·시간 초과는 관련 Worker와 messageerror listener를 해제한다. 큰 응답 복사는 작업 단위로 양보하며 취소 token을 확인한다. 기본 전체 기한은 30초다. 구조·문자 수 등 자원 한계를 넘으면 `resource-limit`으로 구별하며 물리적 발산과 혼동하지 않는다.

실제 Chromium의 esbuild module Workers 검사는 Next가 출력한 생산 chunk 검사와 다르다. hook의 SSR·제어기 검사는 실제 reader의 hydration·visibility·상태 저장·접근성 검사와도 다르다. 후자는 reader를 만든 뒤 별도로 검증해야 한다.
