# Brownian bridge: 정확한 수치 기반·독립 검산·실제 Worker

2026-11-10. `brownian-bridge-time-slicing-measure`의 계산과 전체 수치/통신 검산을 연결했다. 아직 교수 화면과 실제 reader는 미구현이다. 미등록 초안1개를 제외한597등록 단원·1838안내·등록1194한영 원고·입문315/362검증·47미완료를 유지한다.

## 계산과 독립 대조

[모형 계약](brownian-bridge-model-contract.md)에 일곱 입력·단위·점 원천·고정 양끝·관측량·정련과 보증 한계를 명시했다. N=2…32의 내부 정밀도, 양끝0행/열을 포함한 전체 공분산, 기울어진 평균, LDL/행렬식, 자유 커널의 정확한 유리수 계수, 면적·증분·기대 작용과4단계 정련을 계산한다. 양끝 원천은 정규화하지 않은 무게만 바꾸며 조건부 분포는 유지한다.

Python 참조47입력은 별도 시간 발생 행렬·Gauss–Jordan 역행렬과 pivot 행렬식·끝점 힘 풀이, 완전제곱 차이와 증분 선형 관측량을 사용한다.16명시 예제와 N=2…32전 범위31입력이다. 생산의 닫힌 Green 식과 모든 수치 필드를 대조했다. 앞서 검산한 유한 항등식695개도 유지한다.

독립 브라우저 검산기도 별도 유리수 산술·발생 행렬·소거·증분 가중합으로 전체 수치 결과를 재구성한다. 불신 응답의 분수를 산술에 넣지 않으며, schema·범위 주장·정규화·원천·면적·정련 필드를 빠짐없이 비교한다. 작은4모형의 개별 leaf798건과 최대2입력의 선택 항목54건 변조를 각각 거부했다. 최대 입력 전체 leaf를 모두 변조했다고 주장하지 않는다.

최대32구간 출력은 서로소 입력138008bytes와 작은 양의 계수 입력88123bytes이며, 기존 wire 한도·33×33 공분산·31×31 정밀도를 유지했다. 최고256구간 정련은 해석적 행이며 그 크기의 전체 행렬을 풀었다고 표시하지 않는다. π·제곱근·지수의 소수 구간, 연속 측도의 존재, 임의 potential의 극한은 보증하지 않는다.

## 실제 실행과 검증

- 수치60·원고2·독립 수치 검산52·통신/Worker54·controller18, 관련186검사를 통과했다. CI와 같은 Node.js20에서도186검사가 모두 통과했다. TypeScript와ESLint도 통과했다.
- 실제 module Worker에 독립 참조47입력과16예제를 전달했다. 반복을 포함한63요청의 계산/검산 해시가 일치하고 수치 변조63·requestId 불일치63을 거부했다. 계산 중 타이핑이 가능했고 두 Worker를 종료했다. 이것은 생산 Next 자산 검증과 구분한다.
- 실제 React hook에서 입력 비활성화·계산/검산 중 숨김·복귀·검산 중 새 입력·오류 후 재시도·pending상태 unmount를 확인했다.18Worker생성/18종료·활성0이다. 숨김은 visibility 이벤트 시뮬레이션이며 실제 다른 탭 이동을 자동화했다고 주장하지 않는다.
- main thread hook은 계산기·독립 검산기·정확 산술을 포함하지 않는다. 별도 검산 Worker에도 생산 계산기와 생산 산술의 runtime 의존성이 없다. 이전 입력의 결과는 새 입력이나 숨김 상태에서 노출하지 않는다. 필드 삽입 순서가 달라도 같은 입력을 인식하고, 잘못된 hook 입력은 Worker를 만들지 않고 invalid-input 오류를 반환한다. 원래 분수 표기가 다른 입력은 계속 구분한다.

## 재현과 다음 순서

```bash
python3 scripts/learning/generate-brownian-bridge-references.py
npx tsx --test --test-concurrency=2 scripts/learning-brownian-bridge-{numerics,manuscripts,validation,workers,controller}.test.ts
npx tsx scripts/check-brownian-bridge-workers-native.ts
npx tsx scripts/check-brownian-bridge-hook-native.ts
```

브라우저 검사는Playwright Chromium과 `http://127.0.0.1:4175`의 로컬 정적 미리보기를 사용한다. native Worker 검사가 만든 `.runtime/learning/brownian-bridge-worker/{calculate,validate}.js`를 hook 검사에서 사용한다. 실행 로그·해시·독립 의존 그래프·변조 기록은 두 `.runtime/learning/brownian-bridge-*` 디렉터리에 남겼다.

[21요건](brownian-bridge-acceptance-matrix.json) 중15개를 검증했다. 다음은 기존 색과 스타일의8개념/설명 그림·2D/Three.js·정확 표·6안내·8퀴즈·입력/저장→실제reader등록·생산 자산 검증이다. 현재 홈페이지 출력은 보존한다. 경로적분 원주제에는 Euclidean potential 커널·안장/요동 근사·실시간 위상과 최종 감사가 계속 남아 있다. 전체51영역509주제·영역 감사·연구 심화도 미완료다.
