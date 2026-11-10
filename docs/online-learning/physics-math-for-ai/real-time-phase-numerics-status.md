# 실시간 위상·고전극한: 세 모드의 수치 producer

2026-11-10. `calculateRealTimePhase`에 chain/packet/fold의 계산을 연결했다. **새72검사(파동/행렬28·Airy44)**가 Node.js20.20.2/24.13.0에서 통과했다. [23요건 중13개](real-time-phase-acceptance-matrix.json)를 확인했으며 독립 전체 응답 검산·Worker·교수 화면·실제 reader와 단원/원주제 감사는 남는다.

## 계산하는 대상

- **chain:** 전체 exact 작용/Hessian/forcing·정상 경로·signature·초점/affine family·고정 시간 정련을 유지한다. 비특이 kernel의 진폭·위상·복소 값·log를 감싼다. 정확한 초점에는 점별 수 대신 parity/delta와 Maslov 위상을 준다. 양의 Abel 조절자에서는 각 pivot sqrt의 branch를 곱하고 complex mean·연결 응답·scaled kernel·별도 damping/log를 보존한다. 확률 공분산이나 유니터리 감쇠 전파라고 부르지 않는다.
- **packet:** chirp와 두 폭 족의 전체 공분산, 중심 작용과 metaplectic phase를 유지한다. 정규화 계수·중심 파동함수와 mean±4σ의65개 위치/복소 값/위상/밀도를 감싼다. 표시는 유한 창이지만 전체 Gaussian을 유한 창에 다시 정규화하지 않는다. B=0에서도 정확한 parity 위상을 적용한다.
- **fold:** 독립 물리 u/ℏ로 `2πℏ^(1/3)Ai(−u/ℏ^(2/3))`를 계산한다. 중앙 ODE 급수, 양의 인수 scaled 점근식, 음의 인수 두 진동 계열에 명시한 tail을 적용한다. 두 실수 정상점의 위치/작용/곡률과 첫 간섭 근사도 제공한다. u<0에서는 두 복소 정상점도 감싸며, 실수 contour에 대응하는 양의 허수 위치/작용의 감쇠 가지를 구별한다. 합류점·비실수 정상점 영역, 지수적으로 작은 진폭의 scaled/exponent/log와 미해결 부호를 구별한다. 조화 delta 초점을 Airy 곡선으로 대체하지 않는다.

## 인증과 독립 기준

[Gamma 적분 경계](real-time-airy-origin-certificate.md)를 TypeScript에서도 계산한다. 소수 상수를 그대로 믿지 않는다. origin fixture의1536bit certified 구간, 별도 exact finite-core fixture와 독립 mpmath의25wave/40Airy 사례를 대조했다.

wave 기준은 조절자의 analytic eigenvalue-root product를 사용해 producer의 pivot/half-angle 방식과 대조한다. packet65표본 전체, 총520표본의 좌표·복소 값·위상·밀도를 포함한다. packed real inverse와 complex 응답의 대칭 성분4031개는 손실 없는 exact 인수에서 별도로 복원한 분수와 직접 교차 곱으로 비교한다. 앞 단계의 모든 exact cell과 Python 밀집 기준 hash 대조도 유지한다.

큰 실수 Airy의 두 계산 경로는 양/음 |z|=64/68/72에서 겹침 검사를 했다. 첫 영점 근처의 양쪽, 물리 control±16/ℏ 최소, 극소 control/큰 ℏ도 포함한다. 네 Fresnel quadrant는 정상 작용0인 모형에서 정확한 ±√(1/2)와 대조한다.

독립 소수 기준은 **인증된 참조 구간이 아니다**. 생산 경계의 근거는 정수 방향 반올림·정확한 유리수·해석적 tail이다. 처음 대조에서 발견한 두 기준 문제도 수정했다. v=2의 정확한 대칭 spectrum은 짝지어 원래0인 성분에 수치 잔차를 넣지 않는다. 작은 raw regulated 값에는9216bit 기준 격자와10240/10496bit 계산을 쓴다. 생산 경계를 넓혀 기준 오차를 숨기지 않았다. 이전 실패 로그와 최종 성공 로그를 모두 보존한다.

Airy 공식/실수 remainder 조건은 [DLMF9.7.9 및 §9.7(iii)](https://dlmf.nist.gov/9.7#E9), 원점값은 [DLMF9.2(ii)](https://dlmf.nist.gov/9.2#ii)를 직접 확인했다. 유한 시간 양끝 규약과 위상 branch는 앞 [일차 자료 감사](real-time-phase-primary-source-audit.md)를 따른다.

## 정밀도와 전송

[producer 계약과 상계](real-time-phase-producer-budget.md)는 chain/packet8192bit·Airy4096bit·관찰512bit, 큰 위상1024bit와 적응 π reduction을 명시한다. exact 행렬 인수와512bit 관찰은 별도 자료다. packed 행렬의 공통 binary scale은 모든 성분에 같은 절대 quantum을 주며 각 작은 성분의512bit 상대 정확도를 약속하지 않는다.

각 조절자 pivot은 successive continuant ratio로 복원하므로 중복된 큰 분수를 전송하지 않는다. 전체 행렬을 제거하거나 N32/12자리/세 물리 모드의 범위를 줄이지 않았다. [실행 가능한 상계](../../../scripts/learning/verify-real-time-producer-bounds.py)는 chain 총문자열1619514자·단일22041자, packet750000자·fold24000자로 감싼다. node10000·depth12·배열65의 구조 상계도 기존 wire 한도 안이다.

최대/경계32입력의 실제 Node20/24 snapshot 최대는 **1431270bytes·총문자열1379949자·단일20979자·node5588**이다. 최대 응답은 혼합 서로소 N32모형이다. 이 기계에서 최대 producer 계산은 Node24 약42초·Node20 약45초였으며 Worker deadline의 증명과 구별한다. 독립 검산/브라우저 Worker의 시간·복사·취소/복구 비용은 다음 단계에서 검증한다.

## 재현

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-airy.test.ts scripts/learning-real-time-wave-numerics.test.ts
npx tsx scripts/learning/profile-real-time-phase-numerics.ts
python3 scripts/learning/verify-real-time-producer-bounds.py
npm run typecheck
```

체크인 독립 기준은 `scripts/fixtures/real-time-{airy,wave}-references.json`, 생성기는 `scripts/learning/generate-real-time-{airy,wave}-references.py`다. fixture 재생성에만 mpmath가 필요하고 Node 검사는 체크인 JSON을 읽는다. 타입 검사·신규 파일 lint도 통과했다. 생산 build/브라우저/접근성 검증을 이번 미등록 수치 단계의 증거로 주장하지 않는다.

증거는 `.runtime/learning/real-time-phase-numerics/verification.json`, 양쪽 Node의 응답 profile·검사 측정/로그·source hash와 producer 상계다. 수학·유도식·fixture 사례 수와 프로젝트 검사 수를 합산하지 않는다.

## 수량과 다음 순서

등록600단원·1857안내·1200원고, 미등록1단원/2원고와 입문315/362검증·47미완료를 유지한다. 기존 홈페이지 출력과 Online Book 메뉴는 변경하지 않았다. 전체51영역509주제·영역 감사·연구 심화도 미완료다.

다음은 request 귀속·전체 수치/인수·phase branch·꼬리/scale/log를 확인하는 독립 응답 검산기와 Worker 계약이다. 이후 실제 Worker 수명/기한·변조 거부 → 기존 홈페이지 스타일의 한영 그림/실험/저장 → Interactive Book reader → 단원/원래 경로적분 주제 감사를 진행한다.
