# 실시간 응답 검산: packet의 전체 관찰값

2026-11-10. [Abel 조절 파동 검산](real-time-phase-abel-validator-status.md)에 이어 `verifyRealTimePacketWave`를 추가했다. 정규화·metaplectic/center phase·분산과 **65개 표본의 모든 위치·위상·복소 값·밀도**를 검산한다. Airy·세 mode 전체 응답 gate·Worker·화면은 아직 미완료다. 완료 요건 13/23과 등록 수치는 유지한다.

## 별도 계산 경로

소유 물리 입력의 exact certificate를 먼저 다시 검산한다. 공분산은 독립 canonical flow로 이미 확인한 전체 Σ에서 읽는다. 물리 밀도 `ρ(x)=exp(−z²/2)/sqrt(2πΣqq)`를 직접 계산하고 파동 진폭은 그 제곱근으로 얻는다. 생산자의 파동 envelope를 제곱해서 밀도를 재구성하지 않는다. 로그 정규화는 log2·logπ·logΣqq를 각각 합성한다.

z는 −4…4의 1/8 간격이다. 이 유한 창 안에서 밀도를 다시 정규화하지 않는다. 65개 순서와 각 메타데이터를 확인한다. 같은 표준화 좌표의 exp(−z²/2)는 대칭인 33개 상수만 frozen cache에 저장한다. 물리 입력에 따른 참조값이나 후보 응답은 저장하지 않으며, 매번 exact 자료·전체 후보 관찰값을 검사한다.

## Branch와 표본 위상

초기 complex width를 Z₀=χ+iℏ/(2σ₀²)라 두고 `α=A+B Z₀`의 principal inverse-root phase를 구한다. α의 실수부가 음수일 때는 안정적인 허수 Cartesian root를 먼저 구해 상쇄를 피한다. α=iB·cg 관계를 사용하므로, signature의 음수 모드 수 ν에 따라 다음 winding을 적용한다.

- B>0: `exp(−iπν/2)`
- B<0: `exp(−iπ(ν+1)/2)`
- B=0: 검증된 caustic index k의 parity phase `exp(−iπk/2)`

생산자가 쓰는 양의 실수부 cg의 half-angle 식을 호출하지 않는다. 두 B 부호·α 실수부의 양/음/0·정확한 parity·초점의 양쪽을 별도로 확인한다.

표본의 dx=z√Σqq에서 `Σqp/Σqq·dx²=Σqp·z²`를 **정확한 유리수 계산으로 먼저 소거**한다. 위상은 `(Sc+pt·z√Σqq+Σqp·z²/2)/ℏ`를 합친 뒤 독립 halving/doubling으로 한 번 회전한다. 생산자의 center phase와 local phase를 따로 곱하는 계산 경로와 다르다. irrational sqrt의 구간 반경은 sin/cos의 Lipschitz bound로 포함한다.

## 전체 물리 입력의 정밀도 예산

N2…32, 12자리 유리수와 원래 두 폭 족을 유지한다. 아래는 경험 격자가 아니라 모든 허용 물리 입력에 대한 보수적인 크기 상계다. 양수 매개변수는 2⁻⁴⁰보다 크고 16 이하이며, δ=τ/N은 2⁻⁴⁵보다 크고 8 이하다.

| 양 | 보수적 경계 |
|---|---|
| 전체 physical flow 행렬 성분 | 절댓값 <2⁶⁸⁹ |
| 최종 공분산 성분 | 절댓값 <2¹⁴⁷⁰ |
| Σqq | >2⁻¹⁵⁵² |
| α의 절댓값 | >2⁻⁷⁴⁵ |
| 표본 위상 크기·sqrt 민감도 | <2¹⁵¹⁶ |

좌표를 `(q,δp/m)`으로 바꾸면 step의 row-sum norm은 2²⁰ 미만이다. 최대 32회 합성과 physical rescaling으로 flow 경계를 얻는다. 초기 공분산 성분은 2⁸⁸ 미만이고 detΣ=ℏ²/4>2⁻⁸²이므로 최종 공분산 및 Σqq의 하계를 얻는다. `1=AD−BC`와 η=ℏ/(2σ₀²)>2⁻⁴⁹를 사용하면 `|A|+|B|≤|α|(1+17/η)`에서 α의 하계가 따른다.

작업 정밀도는 3072비트다. 최대 conditioning 1552비트와 여유 64비트를 배정하면 1456비트가 남아 512비트 관찰값에 여유가 있다. 큰 위상의 독립 회전은 별도로 1536비트를 사용한다. [Fraction 상계 검사 스크립트](../../../scripts/learning/verify-real-time-packet-checker-bounds.py)는 크기·조건수 부등식을 확인하며, directed 산술/급수 잔여항의 근거나 전체 응답 검산을 대체하지 않는다. 증거는 `.runtime/learning/real-time-packet-wave-check/physical-bounds.json`이다.

## 검증과 남은 범위

기존 8개 packet 모형에 음의 B 두 branch, 양의 B/음의 α, 근접 초점 양쪽과 최대 혼합 서로소 입력을 더했다. 별도의 음의 B/정확히 0인 α 실수부 경계도 확인한다. 합계 15개 모형·975개 표본이다. 상위 scalar와 처음/중앙/마지막 표본의 모든 primitive field, 모든 65개 위치 변조, 후반 복소 성분·coordinated phase sign 반전·유한 창 재정규화를 검사한다.

반환 scope는 `packet-wave-only`, `fullResponseValidated=false`다. header·전체 응답·fold를 이 부분 검산으로 인증하지 않는다. canonical dyadic endpoint·구간 포함/폭·공유 척도·소유권·getter·귀속·표본 순서도 확인한다.

```bash
python3 scripts/learning/verify-real-time-packet-checker-bounds.py
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-packet-wave-check.test.ts
npm run typecheck
```

Node.js20.20.2/24.13.0에서 새22검사가 통과했다. Node24는 기본21검사와 추가 초점 경계1검사로, Node20은 최종22검사 묶음으로 확인했다. primitive 변조70개와 모든 표본 위치 변조65개를 거부했다. 타입 검사와 변경 파일 lint도 통과했다. 실행 결과·source hash·변조 수와 시간은 `.runtime/learning/real-time-packet-wave-check/verification.json`에 기록했다. Worker의 복사·취소·기한은 아직 검증하지 않았다.

다음은 Airy 전체 관찰값과 세 mode 전체 응답 gate이며, 실제 Worker·시각화·학습 화면·접근성·reader·홈페이지 회귀 점검이 이어진다. 등록600단원·1857안내·1200원고, 미등록1단원/2원고, 입문315/362 검증·47미완료 및 전체51영역509주제·영역 감사·연구 심화는 유지한다.
