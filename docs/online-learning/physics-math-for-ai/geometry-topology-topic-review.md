# 미분기하·위상·데이터 형상 원주제 대조

2026-11-11 기준. 기존 `geometry-topology` 10개 단원의 한영 이론·계산·실습을 대조하고 보강했다. 기존 설명과 연습은 보존했다. 누적 범위는 21영역·205개 원주제이며 전체 51영역·509개의 일반 이론과 연구 심화 완료를 뜻하지 않는다.

## 주제와 확인 위치

원고는 `content/learning/physics-math-for-ai/{ID}.{ko,en}.md`, 계산은 `src/lib/learning-geometry.ts`다.

| 원주제 | ID | 보강한 논증·예제 |
|---|---|---|
| 위상·연속성·연결성 | finite-topology-continuity | 최소 열린 이웃 graph와 유한 연결 성분의 동치, quasi-component 일반화의 한계 |
| 다양체·chart | circle-manifold-charts | compactness로 전역 chart 부재 증명, 역매개화의 metric과 겹침 변환 |
| 접·여접공간 | tangent-cotangent-pairing | 일반 Jacobian의 vector·covector·metric·gradient 변환과 비선형 국소 예제 |
| Metric·geodesic | circle-metrics-geodesics | lift별 삼각부등식·Cauchy–Schwarz 최소성, 같은 자취의 서로 다른 에너지 |
| Connection·curvature | sphere-connection-holonomy | Christoffel 계수·반대칭 운반 생성자, Gauss–Bonnet의 경계항 |
| 미분형식 | differential-forms-stokes | 별 모양 영역의 방사 potential 구성과 punctured domain에서의 실패 |
| Embedding·immersion | embeddings-immersions-cusps | compact/proper 가정, 단사 immersion인 무리수 torus winding의 비embedding 반례 |
| Homology·persistence | rips-homology-persistence | rank–nullity 몫공간 유도, GF(2) chain 전수 조사와 고차 coface 필요성 |
| 위상적 안정성 | persistence-stability-conditions | 공통 labeling의 interleaving, 대수적 안정성 정리의 적용 조건, feature별 생존 보증 |
| 좌표 독립 표현 | coordinate-invariant-metrics | 비선형 평면 좌표의 metric·gradient·비영 connection과 영 곡률 구별 |

## 독립 계산·시각화

10종의 Python 예제를 한영에 같은 코드로 제공하고 20개 블록을 실행했다. 정확분수는 chart·쌍대·비선형 metric 항등식을 검사한다. Simpson 구적은 같은 사분원의 길이와 에너지를 비교한다. 구면은 두 성분 운반 ODE를 RK4로 풀어 최종 각도 해와 대조한다. Angular form은 방향과 중심을 바꿔 직접 중점 적분한다. Pell 근사는 60자리 Decimal의 작은 위상으로 재방문을 계산한다.

정사각형 homology는 열 축약을 재사용하지 않고 모든 GF(2) chain의 경계 상 원소 수를 센다. 변형 사각형의 H1은 변 길이의 최대와 먼저 들어오는 대각선으로 생성·소멸을 직접 구하고, 직접 대응/대각선 삭제 두 경우를 비교한다. 이 독립 코드는 실습의 새로운 제어 기능이 아니다.

한영 SVG 네 개는 실제 원과 chart metric, persistence 구간과 threshold Betti를 나란히 보여준다. 440단위 폭과 접근성 title·desc·본문 대체 텍스트를 제공하며 인접 표와 실행 코드로 숫자를 확인할 수 있다. 마지막 그림의 death 끝점은 포함하지 않는 [birth,death) 관례다.

## 수치 수정과 보존

Separation 2π는 같은 끝점이지만 기존 현 공식은 약 2.45×10⁻¹⁶을 남겼다. 주각변위로 현을 계산해 정확한 0을 유지한다. 구면 holonomy는 이미 [-2π,2π] 범위이므로 조건부 한 바퀴 감기를 사용한다. π를 더했다 빼는 연산으로 적도 근처의 작은 회전을 반올림하는 문제를 줄인다. 두 수정은 독립 경계 회귀 테스트를 추가했다.

한영 figure-eight 수식의 마침표 차이 하나를 정리했다. 변경한 20개 원고의 과거 기준 SHA·현재 SHA·이 검토 기록을 보존 ledger에 명시하며 네 과거 fixture를 바꾸지 않는다. 추가 미검토 변경은 계속 거부한다.

## 근거와 남은 심화

정의와 정리는 저자·개발기관의 자료를 기준으로 확인했다. [Hatcher의 위상 노트](https://pi.math.cornell.edu/~hatcher/Top/TopNotes.pdf)는 위상·연결성과 compactness의 기초를 제공한다. [Lee의 Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/) 및 [Riemannian Manifolds](https://sites.math.washington.edu/~lee/Books/RM/)의 저자 안내와 해당 주제에 맞춰 tensor 종류·connection·내재 기하를 구별했다. [Sadun의 미분형식 강의](https://arxiv.org/abs/1604.07862)는 Stokes와 좌표 호환성을 다룬다. [GUDHI 공식 Rips 문서](https://gudhi.inria.fr/doc/latest/group__rips__complex.html)는 지름 filtration과 skeleton 비용을 명시한다. [Chazal·de Silva·Glisse·Oudot의 논문](https://arxiv.org/abs/1207.3674)은 module 안정성 단계의 근거다.

본문의 명시적 행렬·구적·Pell·네 꼭짓점 계산은 이 교재의 유도와 독립 예제다. 이 유한 모델이 미지 데이터의 다양체를 복원한다는 정리는 주장하지 않는다. 일반 de Rham 이론, 특성류, 비교 기하, 대규모 persistence·표본 복원 정리와 학습 metric의 연구 검증은 후속 범위다. 따라서 전체 교육과정 완료 표시는 바꾸지 않는다.

## 검증

- Python 20개 블록 실행, 한영 코드 SHA·수식·제목 수·언어 순도 대조 완료.
- `learning-geometry.test.ts`와 원고 보존 테스트: 선택 테스트 60개 통과. 전체 CI 테스트의 통과를 의미하지 않는다.
- `npm run typecheck`, `npm run lint`, `npm run validate:content` 통과.
- Node 20의 격리 후보에서 `npm run build` 통과: 생성 경로 1,893개. `npm run check:export`로 HTML 1,888개의 내부 링크·자산·앵커·사이트맵·canonical 검사 통과.
- `npm run check:geometry`: 한영·320/768/1440px·다크 모드·경계값·접근성 95개 상태, 동작 12개 묶음, 안내 예제 30개·퀴즈 20개 통과. 정수 입력의 잘못된 초안 보존과 Escape 복구도 확인했다. 수집된 페이지 실행 오류와 Axe WCAG 위반은 0개다.
- 최종 정적 출력에서 네 그림의 320px 캡처를 검토했다. 소스 1,882개·한영 원고 1,300개·이번 반영 파일 31개·설정 8개의 후보 일치와 네 SVG 출력 byte 일치를 확인했다.
- commit/push와 공개 배포는 구분한다. GitHub Pages의 최종 반영은 해당 배포 작업 성공으로 별도 확인한다.
