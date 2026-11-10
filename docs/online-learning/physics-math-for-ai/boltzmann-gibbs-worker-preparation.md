# Boltzmann·Gibbs worker와 유한 비트 경로 준비 검증

2026-11-07. 계산 전송·취소와 표본 경로를 구현했다. 전용 실제 worker를 격리 브라우저에서 확인했으며, 실제 교재 화면과 Next 정적 출력 통합은 아직이다.518단원·1554안내·원래77미완료 수를 유지한다. 최신 증거는 `.runtime/learning/boltzmann-gibbs-worker/verification.json`이다.

## 구현

- 입력/결과·저장값: `src/lib/learning-boltzmann-gibbs-protocol.ts`.
- controller/evaluator/factory/entry: `src/lib/learning-boltzmann-gibbs-{controller,evaluator,worker-factory}.ts`, `learning-boltzmann-gibbs.worker.ts`.
- hook: `src/components/learning/use-boltzmann-gibbs-{calculation,series}.ts`.
- 경로: `src/lib/learning-finite-gibbs-path.ts`.

세 종류의 입력과 기본값·배열·지원·온도·관측 단계·경로 옵션을 검증하고 불변 스냅샷을 사용한다. 숫자 parser는 비영 underflow를0으로 바꾸지 않는다. 결과는 ID/key, 수치 상태·부호·유리 경계, 지원별 구조와 정상화 분기, `fullyResolved` 범위를 대조한다.147,672자리의 실제 수축 계수도 수용하고 무한 payload는 제한한다. 전송 검증을 임의 송신자의 정답 증명으로 간주하지 않는다.

새 요청은 이전 worker를 종료한다. 오래된 callback·오류·부분 series 결과는 노출하지 않는다. 오류·취소·해제·재마운트에서 포트를 정리하고 SSR/실패 시 main-thread 수치 대체 계산을 하지 않는다. 유효하지 않은 series도 생성 전에 실패한다.

## 양자화 법칙과 결정적 경로

옵션은 seed0..2³²−1,1..32비트,1..2048단계, 첫/둘째 자리·random·systematic·block이다. 이론적 행렬의 선형 표시값 대신 유리 구간에서 CDF ceiling을 판정한다. 정확한 에너지 대칭의 동률은 증명된 경계로 처리하고, 확정하지 못한 그리드는 보류한다.

Q1/Q2는 좌표 조건부를 각각 양자화한다. random은 혼합, systematic은 Q1 Q2, block은 목표의 범주 추출이다. 한 단계 TV 상한은 좌표/random≤1/M, systematic≤2/M, block≤(n−1)/M이며 M=2^bits다. 이 상한은 독립 균등 격자 추출을 가정한다. 실제 seed 경로는 LCG32-high-bits의 결정적 예시이며 독립 난수·정확한 실수 Gibbs 표본·물리 시간을 뜻하지 않는다.

Q와 원래 kernel은 별도로 보존한다. 작은 양의 이론적 전이가 Q에서0이 될 수 있으며 Q의 연결성을 따로 계산한다. random은 자리 선택과 추출에2draw, systematic은 두 조건부에2draw를 사용하지만 좌표 갱신 수는 각각1과2다. 블록의 좌표 수를 실제 실행 비용과 동일시하지 않는다.

## 검증 근거

새 `learning-boltzmann-gibbs-worker.test.ts`와`learning-finite-gibbs-path.test.ts`의62검증을 포함해 회귀289개를 통과했다. lint·타입 검사도 통과했다.27예제와 실제 입력 독립 참조47사례, 불량 응답·지원 그래프·stale·재진입·SSR·큰 dyadic payload를 포함한다.

독립 `generate-finite-gibbs-path-oracle.py`는 기존 고정밀 참조의 Decimal CDF와 정수 LCG로12사례를 계산한다. kernel·경로·빈도·seed·draw 수·오차 상한이 일치했고 fixture 재생성은 바이트가 일치했다.

격리 브라우저10검사는 실제 worker27예제, 최대 경로, 무거운 계산 중 이벤트루프, 취소/교체/재시도, 언어/테마 렌더, 생성/로드 오류, 불량 입력/series, 해제/재마운트를 확인했다.40생성·40종료·0잔존, stale 계산/series0, page error0이다. main probe에는 수치 구현 표식이 없고 worker에만 있었다. 빌드 재현 해시도 브라우저 검증 출력과 같았다.

## 격리 검사의 재현

프로젝트 루트에서 실행한다.4186은 임시 검사 포트이며4175 사용자 미리보기와 별도다.

```bash
node scripts/learning/build-boltzmann-gibbs-worker-probe.mjs
python3 -m http.server 4186 --bind 127.0.0.1 --directory .runtime/learning/boltzmann-gibbs-worker/browser
# 다른 터미널에서 실행하고 완료 후 위 임시 서버를 중지한다.
node scripts/check-boltzmann-gibbs-worker-browser.mjs
```

이 harness는 Next worker URL을 별도 esbuild 출력으로 연결한다. 따라서 Next asset 경계, 실제 읽기24조합·접근성·모션·반응형·저장 복원은 아직 검증하지 않았다. 이번 소유 브라우저와 임시 서버는 종료했고4175는 유지했다.

다음은 개념12개·패널·퀴즈6개와 경로 화면이다. 그 뒤 기존518단원 스냅샷을 보존하여 등록하고 전체 빌드와 실제 읽기 검증을 진행한다. UI·주제 완료 수는 그 전에 늘리지 않는다.
