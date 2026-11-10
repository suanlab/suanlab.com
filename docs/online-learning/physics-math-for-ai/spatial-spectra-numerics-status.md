# 공간 스펙트럼·감쇠·상관 길이: 한영 원고와 수치 기반

2026-11-10. `spatial-spectra-decay-and-correlation-length`의 한영 초안과 계산 기반을 작성했다. 실제 Interactive Book에는 아직 등록하지 않았다. 등록596단원·1832안내·한영1192원고, 입문314/362검증·48미완료를 유지한다. 새 원고2파일은 미등록1단원이며 전체 원고 파일은1194개다.

## 작성한 학습 내용

언어별22절·13표시수식·8해설 문제, 실행 가능한 동일 Python 예제와6예측 비교 실험을 작성했다. 위치/파수의 직관, 물리 간격이 포함된 에너지·Green 정규화, 세 경계와 총합0공간, 전체 복소 Fourier 변환, 네 위치의 정확한 손계산과 trace를 설명한다.

연속체 비교용 a배 파워, 무한 사슬의 점화식과 감쇠 길이, 유한 고리의 주기 복제, 질량0의 부호 변화, 결합0 조건화 반례, 국소 비율의 한계, 부호 있는 파수를 고정한 정련을 다룬다.21입력 예제의 의미와 연구 적용 조건도 한영으로 설명했다. 원고에 예측 실험을 썼다는 사실은 안내 실험의 실제 화면 등록을 뜻하지 않는다.

## 구현과 독립 대조

- `src/lib/learning-spatial-spectrum-input.ts`: 3–16위치·12자리 정수/분수·세 경계·두 공간의 입력 검증과 소유권 복사,21예제. 무거운 수치 엔진 없이 묶인다.
- `src/lib/learning-spatial-spectrum.ts`: 정확한 유리수 공분산·LDLᵀ·투영 잔차, 전체 FCF*·FKF*, trace/Hermitian/역변환 검사, 주기 고유값·무한 감쇠·주기 복제·질량0식·국소 비율·네 수준 symbol 정련.
- `scripts/reference-spatial-spectrum.py`: 별도의 발생 행렬·확장 Gauss–Jordan 역산·Berkowitz 주부분행렬식과300자리 직접 복소 DFT/acosh.30사례의14466구간값과2963행렬원소를 대조했다. 점 참조의 비교 여유1e−250을 인증된 참조 오차 상계로 주장하지 않는다.

유한 행렬은 정확한 유리수다. 비유리 계산은768비트 작업 스케일·512비트 근호·512비트 출력 양자화를 사용한다. 이는 임의의 작은 값에 대한 균일 상대 정밀도 보증이 아니다. 최대 서로소 입력의977592bytes 결과는 기존 wire 한도를 통과했고 입력 범위를 축소하지 않았다.

## 검증 결과와 재현

관련44검사(수치42·원고2), TypeScript, ESLint를 통과했다. 실제 콘텐츠 로더의 엄격한 KaTeX 처리, 한영각22목차와 수식/코드 대응, 영어의 한글 부재,21예제 대응과 NumPy 실습의 실행 결과를 확인했다.

```bash
npx tsx --test scripts/learning-spatial-spectrum-numerics.test.ts scripts/learning-spatial-spectrum-manuscripts.test.ts
python3 scripts/reference-spatial-spectrum.py
npx tsc --noEmit
npm run lint
```

Python 참조에는 SymPy/mpmath, 본문 실습에는 NumPy가 필요하다. 별도 환경을 쓰면 `SPATIAL_SPECTRUM_PYTHON`에 본문 실습용 Python 경로를 지정할 수 있다. 결과는 `.runtime/learning/spatial-spectrum-numerics/`에 기록했다.

## 다음 완료 조건

생산 계산기와 다른 계산 경로의 전체 응답 검산, 변조·귀속·자원 검사, 실제 Worker와 취소/오류복구를 구현한다. 이어 기존 홈페이지 색/스타일의 개념·기준 그림·2D/Three.js·입력/저장·안내/퀴즈를 만들고 실제 reader와 생산 자산을 검증한다. 최종25요건은 아직 완료로 표시하지 않았다. 세 상관함수 단원의 원주제 감사도 별도로 남아 있다.

기존 홈페이지 출력과 이전 소스는 지정된 진행 문서 외에 보존한다. 이 단계에서 정적 빌드 교체, commit/push와 공개 배포는 하지 않았다. 전체51영역509주제의 집필·영역 감사·연구 심화를 계속한다.
