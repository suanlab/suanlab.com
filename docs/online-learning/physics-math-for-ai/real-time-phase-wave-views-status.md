# 실시간 위상·파동묶음·fold 결과 시각화

2026-11-11. [행렬 열지도/Three.js](real-time-phase-matrix-status.md)에 이어 나머지 계산 결과 그림을 실제 계산·독립 검산 Worker에 연결했다. **시각화 요건까지17/23검증**이다. 안내 실험·해설 문제·전체 입력/학습 저장과 접근성·실제 reader/생산 자산·이전 콘텐츠 보존·단원/원주제 감사는 남는다. 새 단원을 아직 등록하지 않았다.

## 현재 입력의 결과를 읽는 화면

- **사슬**: 검산된 unregulated/Abel 단위 위상을 직접 투영한다. 큰 S/ℏ를 reader에서 다시 삼각함수에 넣지 않는다. 정확 초점은 delta 분포의 계수와 parity를 설명하며 점별 높이를 그리지 않는다. 정상점의 유일성·affine family·비호환을 구별하고, 대표 경로와 영벡터 방향을 비교한다. 모든 내부 모드의 정현 모양을 보여주되 부호·영 판정은 검산된 관성 지수로 한다. 소수 고유값의 부호가 판정과 맞지 않으면 숫자를 제시하지 않는다.
- **고정시간 정련**: 기본 N의1·2·4·8배 map을 성분별로 비교한다. 연속 진동자의 sin/cos/sinh/cosh map은 별도 binary64 진단으로 명시한다. 이 선을 오차 인증이나 유한 초점 판정에 사용하지 않는다. 성분별 단위와 서로 다른 극한을 구별한다.
- **파동묶음**: 검산된65표본 전체를 표준화 좌표 z=(x−μ)/σ에서 보여준다. 밀도 축은 σρ(x), 복소 파동 축은 ψ/normalization이다. 전체 이진 배율을 조합하므로 극소 폭에서도 표본을 잃지 않는다. 복소 진폭은 빠른 위상을 보간하지 않도록 점으로 그린다. 선택 표본의 물리 위치·진폭·밀도·metaplectic 위상·원래 구간을 유지한다. 창 밖 꼬리도 명시한다.
- **두 고전극한 족**: 같은 map·chirp·폭 매개변수에서 fixed σ²와 σ²=ℏr를 각각 새로 계산하고 독립 검산한다. 비교 ℏ를 현재 값/1⁄2/1⁄8/1⁄32로 바꾸어 위치·운동량 분산을 별도 패널에서 비교한다. log10 표시와 전체 공분산을 제공하며 위치 초점만으로 위상공간 한 점을 결론내리지 않는다. width의 두 단위를 명시한다.
- **fold/Airy**: 현재 cubic 위상의 소수 진단, 검산된 실수/복소 정상점과 실수 적분 경로의 기여를 구별한다. J=scaledIntegral·exp(−negativeExponential), log 크기·부호·원래 구간을 보존한다. raw의 underflow는 정확한0이 아니다. 현재 u가 어느 허용값이든 유지하면서, 같은 ℏ에서 물리 u∈[−2,2]의33표본과 첫 정상 위상 근사를 비교한다. 이 창은 입력 영역의 축소가 아니다. 점만 그려 고주파 진동·영점을 보간하지 않으며, u=0에는 첫 근사를 넣지 않는다.

공통 곡선은 unavailable 표본 사이를 연결하지 않고, 양·음 값의0 기준선을 명시한다. 실제 검산 응답과 이전13단계의 고정 설명용 그림을 혼동하지 않는다. 기존 홈페이지의 CSS/HSL/버튼·가로 스크롤 스타일을 유지했다.

## 실행과 취소

새 `useRealTimePhaseSeries`는 한 번에 계산/검산 Worker 한 쌍만 실행한다. 표본마다 소유 입력을 보존하고, 구성 변경 시 이전 표본을 render 단계에서 숨긴다. 취소 후 완료 표본을 보존해 이어서 진행하며 오류/기한 종료 후에는 명시적으로 재시도한다. 부모 입력 변경·unmount에서는 기존 controller의 소유권과 종료 규칙을 따른다. 전체33개를 한꺼번에 병렬 Worker로 만들지 않는다. packet 비교와 Airy sweep은 사용자가 요청할 때 시작한다.

## 검증의 범위

- Node20.20.2/24.13.0 각33검사: 원래32예제 모두 전체 응답 독립 gate를 통과한 뒤 화면에 투영했다. 모든 사슬 모드의 부호/끝점, 단위 위상, packet8개·520표본의 표준화 밀도·복소 진폭의 크기·위상 norm, 전체 한영 SSR과 reader 의존성 분리를 확인했다. producer·exact/검산 엔진·Three.js를 wave reader에 import하지 않는다.
- 양쪽 Node에서 별도 각4곡선 검사로 signed 축의0 위치, 양/음 경계, unavailable gap·점만 표시하는 모드, 빈 투영을 확인했다. 합계는 각37검사다. 추가0 기준선은 실제 Worker 경로 그림과 접근성 검사로도 확인했다.
- Chromium KO/EN × light/dark ×320/1440px8조합에서 focus 두 경우·영방향·모드 부호·65밀도/130복소 점·두 ℏ의 두 폭 족·fold 합류/실수 정상점 부재를 확인했다. 세 모드의 접근성24회, 영어 한글 혼입·문서 넘침 부재를 확인했다. Airy33표본은 취소 후 이어서 완주했다.
- 별도 native extreme 검사에서 지수적으로 작은 shadow의 scaled/log 보존, 복소 정상점, 극소 폭65표본, packet 비교 중 부모 입력 변경의 취소를 확인했다. 실제 sweep의 scaled/raw49점(33Airy+16첫 근사), log33점을 확인하고 전체 그림을 저장했다. 정규8조합243개와 extreme75개, 추적 native port318개가 모두 종료되었다. 추가0 기준선 probe의 시작 port는 이 수치에 합치지 않는다.
- packet·scaled Airy·복소 정상점 그림을 직접 검토했다. 소수 그림은 원래 구간의 표시용 투영이며 새로운 수학 오차 인증이 아니다. 연속 진단이나 성긴 Airy 표본을 일반적인 수렴 정리로 확대하지 않는다. 실제 OS background-tab·모든 물리 모바일 기기 검증을 주장하지 않는다.

대상 lint/typecheck가 통과했다. root out과 기존 preview를 보존하고 이번 review 서버만 검사 후 종료했다. Next 생산 자산과 실제 reader 검증은 뒤 단계다. 기존 홈페이지의 배포 성공과 공개 교육과정의 HTTP200·600단원 카드를 별도 확인했으며, 이번 미등록 단원이 공개되었다는 뜻은 아니다.

## 재현

```bash
node --import tsx --test --test-concurrency=1 scripts/learning-real-time-phase-wave-view.test.ts scripts/learning-real-time-phase-curve.test.ts
node --import tsx scripts/preview-real-time-phase-waves.ts
node --import tsx scripts/check-real-time-phase-waves-native.ts
node --import tsx scripts/check-real-time-phase-wave-extremes-native.ts
node --import tsx scripts/check-real-time-phase-wave-extremes-native.ts --zero-guide-only
```

증거와 source/log/그림 hash는 `.runtime/learning/real-time-phase-waves/verification.json`에 있다. 다음은 안내 실험·해설 문제다. **600등록 단원·1857안내·1200등록 원고**, 미등록1단원/2원고, **입문315/362검증·47미완료**, 전체51영역509주제 및 영역 감사·연구 심화는 유지한다.
