# 국소 spin 사슬 원고 감사

2026-11-09. 원고는 `finite-spin-chain-local-hamiltonians-chapter.{ko,en}.md` 두 파일이다. 등록 전 집필 원본이며, 이번에는 `content/`·단원 registry·홈페이지 출력을 바꾸지 않았다.

## 독자가 확인할 주장과 가정

| 원고 위치 | 필요한 가정 / 핵심 주장 | 확인한 근거 |
| --- | --- | --- |
| 2…3절 | 순서 있는 spin-1/2 곱 공간, n₀가 최상위, Pauli convention | 국소 Kronecker X₀·Y₁·Z₀Z₁의 모든 열을 독립 행렬과 대조 |
| 3…4절 | S=ℏσ/2, 실수 두-body 계수, 중복 없는 변 | 계수 변환 J_σ=J_Sℏ²/4, 다른 site의 교환성, 기존61수치 검사의 아홉 축·ring 검사 |
| 5…6절 | J>0 Ising ground span·다음 서로 다른 준위, XY의 위상 민감성 | 직접4×4 spectrum, XX+YY의±2·0, Heisenberg의−3 singlet/+1 triplet |
| 7절 | 비영 원시 상태, Hermitian H | N=8/E=0/Var=1의 독립 H² 계정, 들뜬 영분산 반례·gap2 |
| 8절 | 네 spin의 순수 GHZ, 열린 ZZ와 횡장 | symbolic h에서 E=−3·Var=4h², XᵢGHZ의 Gram=I₄, shift/scale 항등식 |
| 9절 | 전체 순수성 아래 국소 mixedness의 얽힘 해석 | 실제 partial trace의 I/2, mixed-state 비교의 Z 통계와 전체 X 기대1/0·purity1/1/2 |
| 10절 | 시간 비의존 관측량, 전체 operator commutator | [M,XX]·[M,YY]의 상쇄, sharp state와 보존 구별, 횡장 Ising Q=∏X 보존·P=∏Z와 구별 |
| 11절 | 세 중복 없는 변의 대각 반강자성 Ising | 여덟 기저 에너지의 직접 열거:−1 여섯·3 둘·gap4 |
| 12…14절 | 유한 수치 범위와 새 연구 주장 구별 | 비대각 ground=null, 일반3-body/시간/무한 계/최적화의 부재, 여섯 해설의 한영 수식·번호 일치 |

## 독립 유도와 제품 비교의 구분

`generate-spin-chain-authoring-references.py`는 제품 모듈을 가져오지 않는다. SymPy의 Kronecker/H²·symbolic field·직접 density partial trace·실제 commutator를 사용한다. 본문 사례를 직접 만들었으며 외부 해답을 복제하지 않았다. 제품의 정확 수치 결과와 비교한 부분, 혼합 상태·전체 X 곱·symbolic field처럼 원고에서 해석적으로 확장한 부분을 구별한다. 현재 pure-state·two-body 계산기가 혼합 상태 입력이나 four-body Hamiltonian을 지원한다고 주장하지 않는다.

새10검사는 원고 두 언어의 strict KaTeX·GFM 렌더링, ket가 들어간 표의 열 수, 영어의 한글 부재와 수식/문제 번호 일치, 위 물리 사례를 확인한다. 각 언어14절·13표시 수식·3표·6해설이다. 한국어 Hangul5363자·영어3172단어다. 분량·구조 검사는 편집 관문이며 학습 효과를 자동 증명하는 검사는 아니다. 일반 spin 수치61개 회귀도 통과했다.

## 참고 자료를 읽은 범위

[MIT 8.05 강의 목록](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/resources/lecture-videos/)에서18…20강의 tensor product와24…26강의 angular momentum addition 주제를 확인했다. 전 영상을 시청하거나 문제 해답을 검증한 것으로 세지 않는다.

[Sandvik, Computational Studies of Quantum Spin Systems](https://physics.bu.edu/py502/lectures6/qspins.pdf)의 서지(AIP Conf. Proc.1297,135,2010)·서론에서 spin 모형과 exact diagonalization/Monte Carlo·finite-size scaling의 학습 배경을 확인했다.205쪽 전체 알고리즘을 검증했다고 주장하지 않는다. 본문의 작은 행렬·부호·반례는 위 독립 symbolic 계산으로 확인했다. 큰 계의 수렴·복잡도·임계 주장에는 추가 자료와 조건이 필요하다.

## 남은 검증

이 감사는 Markdown 원고와 hand calculation의 검증이다. 등록할 때 원고를 실제 content loader에 연결하고 내용·그림·controls·guide·quiz·storage의 의미가 일치하는지 다시 확인한다. native Worker·실제 reader·키보드/터치·모션·접근성·한영/명암/작은 화면·생산 자산 관문은 아직 남아 있다. 전체 교육과정·다체 원주제·연구 깊이의 완료로 세지 않는다.
