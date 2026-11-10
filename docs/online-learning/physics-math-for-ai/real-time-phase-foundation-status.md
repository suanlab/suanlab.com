# 실시간 위상·고전극한: 자료·유한 수학 기반

2026-11-10. `real-time-phase-and-classical-limit`의 [일차 자료](real-time-phase-primary-source-audit.md)와 [수학 계약](real-time-phase-model-contract.md)을 준비했다. 경로적분 네 처리 중 마지막 단원이며 아직 등록·원고·생산 계산을 시작하지 않았다.

**62유한 모형·41191정확분수 검사·13소수 진단**을 통과했다. 새 단원의23요건 중7개 수학/범위 기반만 확인했다. 생산 상수/정밀도·입력/전송 비용·독립 기준/검산·Worker·한영 원고·그림/실험·reader와 단원/원주제 감사는 남는다.

## 모형과 확인한 구별

실시간 이차 사슬, 그 유한 symplectic map에 의한 chirped Gaussian 파동묶음, 별도1좌표 canonical fold/Airy를 사용한다. N2…32와 전체 행렬을 유지할 계획이다. 음의 진동수 제곱·곡률, 정확한 초점과 near-caustic, 영 pivot을 허용한다.

- 양끝 작용과 전체 Hessian/forcing·determinant·고전 map/생성함수를 독립 경로로 대조했다. 원점의 첫 pivot0이 전체 행렬의 특이성을 뜻하지 않는다.
- signature는 정확 Sturm 부호 변화와 별도 밀집1×1/2×2 congruence로 확인했다. determinant 크기만으로는 Maslov 위상을 얻지 못한다.
- 정확한 초점에서 호환한 양끝의 affine 정상 경로족과 비호환한 양끝의 무해를 구별했다. kernel은 parity를 포함한 delta distribution이며 가짜 유한 점별 값이나 역행렬을 주지 않는다.
- 양의 Abel 조절자의 복소 inverse·mean·완전제곱·양의 실수부 pivots를 검사했다. 각 pivot의 sqrt 곱과 determinant의 주 sqrt가 부호상 다른 N4 자유 예를 확인했다. 소수 branch 진단은 인증된 오차가 아니다.
- Gaussian의 전체2×2공분산과 Möbius complex width가 일치하고 determinantℏ²/4가 보존됨을 검사했다. 고정 초기 폭의 고전 ensemble와 σ²=ℏr의 국소화 족을 구별했다. 초점에서도 packet 분산은 유한하다.
- fold의 두 정상점·작용·uniform rescaling, Airy ODE 계열과 Gaussian 모멘트로 얻은 양의 인수 계수를 정확 검사했다. x64에서160번째 생략항은2^−540보다 작다. 전체 Airy 함수의 인증된 생산 구현을 완료한 것은 아니다.

62모형에는 N2/3/4/6/8/16/31/32, 자유·양의/음의곡률·정확 초점·영 선행 pivot·거친 hyperbolic 분할·양쪽 near-caustic·최대 서로소 분수와 두 packet족이 포함된다. 조절한 복소 dense inverse는 작은 모형과 N32의 정확 초점에서 확인했다. 모든 연속 입력의 생산 비용을 증명한 표본이라고 세지 않는다.

## 자료 확인과 진단의 한계

Evans–Zworski Lemma3.13/Theorem3.15·Wipf§3.1·Horváthy§§2–3·DLMF9.2/9.5/9.6/9.7 및10.32.9의 직접 읽은 범위를 기록했다. PDF의 실제 이미지는 텍스트 추출이 놓친 절댓값 위치를 확인하는 데 사용했다. Horváthy의 작용 분모는 부호 있는sin이며 각 곱의 절댓값과 구별한다. 읽지 못한 Tong 자료는 검토 근거에서 제외했다.

65자리 Decimal contour의512/1024/2048격자 진단6행과 양성1검사,75자리 branch의6검사, 합계13진단은 **적분 오차 인증이 아니다**. 두 유효한 직접 계산을 수렴 관찰로 대조했으며 구간 산술·무한 꼬리/정밀도 증명을 대신하지 않는다. 실제 생산판에서는 Airy 중앙/점근 경로와 상수·argument reduction·scaled/log 표현을 별도로 인증해야 한다.

## 재현과 남은 작업

```bash
python3 scripts/learning/verify-real-time-phase-derivations.py
```

생산 TypeScript 모듈이나 응답을 읽지 않는 표준라이브러리 검산이다. 증거는 `.runtime/learning/real-time-phase-foundation/derivation-verification.json`과 `verification.json`, 진행/실행 로그·PDF/이미지다. 첫 실행과 branch를 추가한 실행을 모두 보존했다.

다음은 일반 대학원생을 위한 상세 한영 원고다. 위상과 확률·정상/최소·signature·Abel/초점·파동묶음/고전극한·정련·fold/비실수 정상점·AI 연결을 중간 유도·손계산·같은 Python 예제로 설명한다. 생산 입력/정밀도/비용 확정과 독립 수치 기반도 이어서 진행한다.

등록600단원·1857안내·1200원고, 미등록 초안0개와 입문315/362검증·47미완료를 유지한다. 전체51영역509주제·영역 감사·연구 심화는 미완료다. 600등록 commit1e11ee1의 GitHub Pages 배포 성공을 확인했다.
