# 무차원 수 자료 검토

2026-11-06. 구성방정식 다음 주제의 자료 검토다. 이후 [두 단원 홈페이지 통합 검증](continuum-dimensionless-authoring-status.md)을 마쳤다.

## 확인한 일차 자료

- [Langtangen·Linge, Finite Difference Computing with PDEs](https://hplgit.github.io/fdm-book/doc/pub/book/html/._fdm-book013.html)의 정지 이류·확산 경계값 문제와 명시적 지수 해를 확인했다. 실제 길이·속도·확산계수와 두 경계 농도를 복원해 별도 유도한다. 물리적 Peclet 수와 격자 크기를 사용하는 수치 Peclet 수를 구분하며, 수치 진동 조건은 본문 문구를 그대로 복사하지 않고 실제 차분식에서 다시 유도할 대상으로 남긴다.
- [MIT 3.185, Chapter2 PDF](https://ocw.mit.edu/courses/3-185-transport-phenomena-in-materials-engineering-fall-2003/a556573bb7fc3f3dd159f62cbd040359_chap2.pdf)의4쪽(식2.29–2.32)에서 일차 소모 반응의 정지 확산과 대칭 판의 cosh 해를 확인했다.6쪽의 대칭점 무유속,19쪽의 무차원 설계 논의도 확인했다. 원문의 L은 양면 판 전체 두께이며 계획의 L은 표면에서 대칭면까지의 반두께다. 이 기하 차이의 계수2를 혼동하지 않는다.
- [B. Finlayson의 저자 제공 반응확산 무차원화](https://faculty.washington.edu/finlayso/ebook/reaction/rxndiff/nondim.htm)는 Thiele 계수 제곱을 확산 시간/반응 시간 비로 설명한다. 본문의 구·원통·평면 차이와 외부 경계 전달을 우리 일차 평판의 정의와 구분한다. 온도 결합·외부 전달 저항을 지금 모델에 포함했다고 주장하지 않는다.
- [NASA Glenn의 Similarity Parameters](https://www.grc.nasa.gov/www/k-12/airplane/airsim.html)의 Re=ρVL/μ와 Ma=V/cs 정의 및 점성·압축성의 닮음 구분을 확인했다. 특정 속도 이하이면 무조건 비압축성이라는 일반화는 채택하지 않는다.
- [MIT 13.021 Lecture6](https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec06/lecture6.html)의 지배식 무차원화와 Fr=V/√(gL)를 확인했다. 관성으로 나눈 중력 계수는 1/Fr²이며,Fr 자체를 관성력/중력의 비와 동일하게 쓰지 않는다. 원문의 큰 Re에서 점성을 생략하는 표현도 경계층을 포함한 전역 허용으로 확대하지 않는다.

## 채택하지 않은 연결

MIT 20.430J의 “Diffusion and convection; The Peclet number…” 자료 페이지는 실제로 내려받은15쪽 PDF가 세포 역학·점탄성·낮은 Reynolds 흐름을 다룬다. 페이지 제목만을 근거로 그 PDF에서 Pe 식이나 경계층 해를 검토했다고 기록하지 않았다. 정확한 수송 예제는 위 저자 제공 교재와3.185 자료를 사용한다.

## 자체 유도와 검증 범위

두 특수 경계값 가족에서 Pe=UL/D와 Da=kL²/D를 직접 얻고 실제 경계·단위·플럭스/소모 적분·차원 복원과 닮음 변환을 검사한다. 일반 유체의 Re·Ma·Fr,열/물질의 Pr·Sc·Pe,반응의 Da와 점탄성 Wi의 정의·지배 항·0기준속도 예외는 상세 원고에서 연결하되, 두 scalar 해가 모든 무차원 수의 지배 현상을 푼다는 주장을 하지 않는다.

[집필 설계](continuum-dimensionless-authoring-plan.md)
