# 실시간 위상·고전극한: 한영 개념 그림

2026-11-11. 실제 Worker와 공통 오류 호환성 점검 다음으로 **13단계 한영 개념 설명·기준 SVG·compact 표지**를 구현했다. 단원 요건은 **16/23 검증**이다. 계산 실험 UI, 안내 실험·문제 UI, 전체 학습 저장·접근성, 실제 reader·생산 자산, 이전 콘텐츠 보존 및 범위 감사는 미완료다. 등록은 아직 하지 않았다.

## 학습 순서와 해석

1. 네 복소 단위 화살표의 합과 확률을 구별한다.
2. 비정상 위상과 정상점 주변의 회전을 비교하고 부분적분 조건을 설명한다.
3. 음의 곡률 정상점을 직접 미분해 최소점과 구별한다.
4. 첫 pivot이0인 가역 행렬의 양·음 고유 방향을 유지한다.
5. determinant 크기와 Maslov 위상, 각 제곱근의 곱과 곱의 주 제곱근을 구별한다.
6. 정확 초점의 호환·비호환 양끝과 delta 분포, 유한 파동묶음의 작용을 구별한다.
7. Abel 감쇠에도 복소 위상이 남으며 유니터리/확률 해석을 주지 않음을 설명한다.
8. 실제 정규화된 파동묶음의 평균·대칭화 공분산과 자유 전파를 비교한다.
9. 고정 폭과 ℏ에 비례하는 분산 족을 같은 눈금으로 비교한다. 별도의 chirp 위치 초점은 위상공간 한 점을 보장하지 않는다.
10. 작은 ℏ·시간 정련·물리 역진동자·거친 격자의 불안정성을 구별한다.
11. fold의 두 정상 기여와 Fresnel 위상을 함께 더한다.
12. 합류의 Airy 척도와 실수 정상점이 없는 쪽의 비영 진폭을 설명한다.
13. AI의 Laplace 근사·선형 응답과 복소 간섭의 해석 경계를 연결한다.

모든 그림은 명시적인 **고정 설명용 기준**이다. 현재 입력의 생산 결과나 검산 증거로 사용하지 않는다. 유한 화살표 합을 경로적분 추정으로, 위상 곡선을 Airy 값으로, 복소 이차 응답을 확률 공분산으로 표시하지 않는다. 작은 표지는 큰 위상 화살표와 닫힌 합만 남겨 좁은 카드에서도 읽을 수 있게 했다. 본문의 상세 그래프에는 축·범례·같은 분산 눈금을 표시했다.

[기존 수학 계약](real-time-phase-model-contract.md)과 한영 원고를 따르며, 이번 단계에서 [Evans–Zworski](https://math.mit.edu/~vwg/semiclassicalEvansZworski.pdf), [Horváthy](https://arxiv.org/pdf/quant-ph/0702236), [DLMF9.5](https://dlmf.nist.gov/9.5), [9.2](https://dlmf.nist.gov/9.2), [9.7](https://dlmf.nist.gov/9.7)를 다시 열어 정리 조건·위상 및 Airy 표현의 경계를 확인했다. Airy 수치 곡선이나 일반 정상 위상 오차를 새로 인증한 단계는 아니다.

## 검증과 기존 스타일

- Node20.20.2/24.13.0에서 각4검사: 예제 연결·한영 구분, 독립 Python Fraction 손계산과 전체 그림 표본, 26개 상세 SVG·2개 compact 기준과 concept/cover SSR, reader 의존성 분리를 확인했다.
- Python은 정상 작용·음의 고유 방향·행렬 역수·초점 map·자유 Hessian principal minors·branch 차이·공분산 determinant·두 폭 족을 분수로 확인한다. 곡선 좌표는 별도 Python binary64 투영과 비교한다. 이 표본 대조는 임의 입력의 고정밀 인증이 아니다.
- Chromium153에서 KO/EN × light/dark ×320/1440px의8조합, 각13단계 **104상태·접근성104회**를 확인했다. 모든 상세 SVG의 글자가 viewBox 안에 있고 영어 본문에 한글이 없었다. 상세26그림·표지8상태의 screenshot을 남기고 작용·fold·분산·모바일 표지를 직접 확인했다.
- 키보드12회, 새로고침8회·잘못된 저장6경우·저장 차단1경우, 언어 전환 시 단계 보존과 부모 제어 상태를 확인했다. uncontrolled 단계는 초기 저장 복원이 끝난 뒤 활성화한다. 이는 개념 화면의 검사이며 전체 실험 저장·물리 모바일 기기 검증은 아니다.
- 기존 homepage CSS와 HSL/learning 색상, Button·개념 단계·가로 스크롤 패턴을 재사용했다. 새 색상 체계를 만들지 않았다. 타입·대상 lint가 통과했고 root out과 기존 preview는 변경하지 않았다. 아직 실제 reader의 Next 자산 build를 주장하지 않는다.

## 파일과 재현

`src/data/real-time-phase-teaching.ts`에 한영 설명, `learning-real-time-phase-reference.ts`에 작은 고정 표본, `real-time-phase-concept/graphic/cover.tsx`에 화면을 둔다. reader에는 producer·exact/검산 엔진·Worker·Three.js를 import하지 않는다. 실제 계산과 Three.js 행렬 화면은 다음 단계에서 별도로 연결한다.

```bash
node --import tsx --test scripts/learning-real-time-phase-concepts.test.ts
node --import tsx scripts/preview-real-time-phase-concepts.ts
node --import tsx scripts/check-real-time-phase-concepts-native.ts
```

로컬 review 서버가 출력한 주소로 검토한다. browser 검사 후 이번에 시작한 review 서버는 종료한다. `.runtime/learning/real-time-phase-concepts/verification.json`에 source/log/그림 hash와 검사 결과를 보관한다. 기존 **600등록 단원·1857안내·1200등록 원고**, 미등록1단원/2원고, **입문315/362 검증·47미완료**, 전체51영역509주제 및 영역 감사·연구 심화는 유지한다.
