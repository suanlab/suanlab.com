# Unitary evolution: 시간 독립 스펙트럼 계산 기반

2026-11-08. [두 가족 설계](quantum-unitary-authoring-plan.md)의 첫 모형과 방향성 sin/cos 기반을 구현했다. **pulse 순서 비교 모형·한영 원고·Worker·교수 화면·실제 읽기는 아직 남아 있다.** 등록554단원·1662안내, 원래302/362입문 검증·60미완료와 전체51영역509주제·연구 깊이·공용 그림 후속 범위를 유지한다.

## 첫 모형

`two-level-spectral-schrodinger-evolution`은 닫힌 C² 순수 상태의 실제 물리 시간을 계산한다. 비영 원시 상태·Hamiltonian 방향, 중심·간격·끝점 시간·양의 ℏ와 관측/Euler 격자를18개 소유 정수로 준비한다. 크기는10⁶이하이며 분모는 양수다. H=aI+b(2P−I)의 스펙트럼으로 전파한다.

- E0·T0·E0T0 단위를 복원하고 실제 시간·무차원 위상·전파 행렬·정규화 진폭·표준/에너지 측정 확률을 반환한다.
- 초기 에너지 스펙트럼·평균/분산은 정확 유리수다. 각 시간의 직접 norm·에너지 모멘트와 unitarity·반 시간 합성·역전파 잔차를 구간으로 확인한다. 잔차가0을 포함하는 것과 구조적으로0인 것을 구별한다.
- 에너지 공통 이동의 전역 위상과 상대 위상에 의한 ray 변화를 구분한다. 거의 돌아온 상태를 반올림된 fidelity만으로 같은 ray라고 분류하지 않는다.
- 하나의 고정 Δt 격자에서 전진 Euler를 정확 복소 유리수로 계산한다. 출력을 재정규화하지 않고 norm 증가·성분 진폭·정확 스펙트럼 norm 항등식·실제 전파와의 오차를 보여준다. Number 범위를 넘는 증가를 infinity나 확률로 바꾸지 않는다.

방향성768bit 작업 연산과512bit 공개 구간을 사용한다. 삼각함수는 작은 각의 급수·명시적 나머지·반복 배각으로 계산하며 Math.sin/cos나 부동소수 π 축소를 인증 근거로 삼지 않는다. 수학적 범위[−1,1]·[0,1]과의 교차와 norm 재정규화를 구분한다. 자세한 논증은 [원문·구간 계약](quantum-unitary-source-audit.md)에 기록했다.

## 검증

- 새 스펙트럼/구간71검사와 기존 관측량 수치62검사, 합계133검사 통과. 타입·전체 lint 통과.
- 독립 준비31개·실제 시간 표본152개. mpmath250자리 직접 행렬 exponential, SymPy 고유공간, Python Fraction 직접 Euler 행렬 반복으로 대조했다.
- 정확 유리수 인증1649회와 방향성 점 관측8990회를 따로 계수했다. 점 구간 폭은 max(1,|참조|)에 대한1e−125이하로 확인한다. 정확한 수는 `.runtime/learning/quantum-unitary-foundation/reference-checks.json`에 기록한다.
- 별도 삼각함수 준비15개와 넓은 위상 구간5개·표본45개·sin/cos90항목을 확인했다.2⁶⁴까지의 부호 있는 위상, binary64에서 같은 값으로 보이는10¹⁸과10¹⁸+1을 구별한다.
- 별도 복소 DOP8539궤적·180개 실수 성분을3e−11이내로 대조했다. 이 수치 대조 허용 오차를 엄밀한 ODE 전역 오차 인증으로 부르지 않는다.
- 독립 참조 재생성이 바이트 단위로 일치했다. 입력 소유 필드·범위·격자 호환성과 불변 복사, 큰 Euler 성장·반 시간 group 성질·역전파·보존량·전역 이동·가까운 revival을 확인했다.

앞선 전체10026검사·정적1792 HTML은 관측·연산자 통합의 근거다. 이번 첫 모형의 Worker/화면/읽기 검증으로 세지 않는다. 기존 교재·수치 모형·참조와 정적 출력은 유지한다.

## 다음 순서

구간별 Hamiltonian pulse의 실제 시간 전파·역순·평균 Hamiltonian 근사 가족을 구현한다. 비직교 방향의 평균 생성자를 처리하고 경계의 연속성·영 지속 시간·가환/비가환 경우를 독립 검산한다. 이어 두 가족의 상세 한영 원고 → 입력/Worker → 개념 비교·자유 실험·안내·예측·퀴즈·저장·모션 → 도서 → Interactive Book 실제 읽기로 연결한다.

한영·명암·모바일·접근성·정적 출력까지 확인한 뒤에만 원래 입문 완료 수를 갱신한다. Online Book 목록에는 새 교재 링크를 추가하지 않는다. 무한 차원 정의역·일반 시간 의존 H·열린계·연구 심화와 뒤의 모든 영역은 계속 남는다.

새 빌드·공개 배포·commit·push는 수행하지 않았다. 근거는 `.runtime/learning/quantum-unitary-foundation/verification.json`, `reference-checks.json`, `tests.log`, `reference-reproduction.log`와 보존 기록이다.

```bash
npx tsx --test scripts/learning-spectral-evolution.test.ts scripts/learning-quantum-observable.test.ts
python3 scripts/learning/generate-spectral-evolution-references.py --output .runtime/learning/quantum-unitary-foundation/reference-regenerated.json
npm run typecheck
npm run lint
```
