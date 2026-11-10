# 연속체 종합 점검 3단계 준비

이후 [3단계4단원의 종합 점검](continuum-domain-audit-stage3.md)을 마쳤다. 아래 부분 진행 수치·미완료 설명은 이 문서 작성 시점의 기록이다. 현재 다음은 [4단계 구성식·무차원 수](continuum-domain-audit-stage4-preparation.md)다.

2026-11-06. 다음 대상은 확산·반응수송4단원·한영8원고·입력45개·예제72개다. 실제 등록의 다음 순서를 따른다.

1. `fourier-heat-diffusion-mode-budget`: 열용량·전도율·절대 온도, 기준 온도의 sensible energy와 이차 노름, 실제 유한 창의 경계 열출력.
2. `gaussian-fick-diffusion-spreading`: 실제 질량·단면적, 확산 분산·이동 평균, 유한 창의 작은 꼬리와 경계 수송.
3. `reversible-reaction-advection-diffusion-budget`: 순반응과 전·역 turnover, 종별 저장·수송·원천, 전체 질량과 조성.
4. `fisher-kpp-reaction-diffusion-front`: 반응 원천과 전선 속도, 이류·확산·성장 분리, 수송만 남긴 대조.

## 먼저 재현한 전체 온도의 가용성

열모드 계산 일부를 읽고, 표시할 수 없는 작은 온도 섭동 때문에 정상적인 전체 온도까지 미정이 되는 사례를 재현했다. 체적 열용량0.25, 전도율4, 기준 절대 온도300K, 대비를 binary64 최소 양수, 파수4, 위상0, x=0, 시간4초로 두었다. 수정 전 `heatModeAt`은 `temperature=null`, `roundedTemperature=300`을 반환했다.

독립1600자리 식에서 섭동은 양의 비영이지만 표시 범위 밖이고, 전체 온도는300K로 반올림된다. `.runtime/learning/continuum-domain-stage3/heat-temperature-before.json`에 실제 입력·출력과 독립 참조를 보존했다. 전체 온도와 창의 현열 가용성을 수정하고, 한영 본문에 전체량과 섭동의 차이를 설명했다. [부분 수정 기록](continuum-domain-audit-stage3-heat-totals.md)을 참조한다. 이는4단원 전체 점검 완료를 뜻하지 않는다.

## 이어서 확인할 범위

한영8원고와 모든 계산·입력·안내를 실제로 읽기 전에는 점검 행렬을 완료로 갱신하지 않는다. 작은 상대 변화·가중 절대 변화·총량, 정확한0·비영/null, 셀과 유한 창의 단위·저장·경계·원천을 대조한다. 표시할 수 없는 관측이 있어도 선언한 가로 범위를 읽을 수 있는지 그림도 확인한다. 새로 바꾸는 계산은 독립 참조와 현재 소스의 화면에서 검증한다.

이 준비 문서는4단원의 본문·독립 계산·화면 검증 완료를 뜻하지 않는다. 전체 연속체 및 전체 교육과정·연구 심화도 미완료다.
