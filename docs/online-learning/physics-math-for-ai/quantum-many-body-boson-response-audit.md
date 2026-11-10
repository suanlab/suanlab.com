# Boson 원시 응답의 정확 field·cutoff 귀속 감사

2026-11-09. `boson-fock`의 전체 응답을 검증한다. Coherent builder의 tail·무한 상태 비교는 별도의 다음 가족이며 이번 완료에 포함하지 않는다.

## 정확 field와 구간의 의미

`learning-many-body-field-wire.ts`는 Q(√2)[√3]의 이차쌍 연산으로 네 유리수 계수를 계산한다. 제품의 XOR 좌표 곱셈·field 관측·root/interval engine을 가져오지 않는다. 정확 부호는 이차체의 부호와 부호 있는 제곱 비교로 결정한다. 일반 나눗셈도 이차쌍의 공액과 유리수 norm으로 계산한다.

응답의 네 계수·exactSign을 정확히 대조한 뒤 observation의 양쪽 유리수 endpoint와 실제 대수적 값의 차이를 부호로 검사한다. 정규화 좌표는 endpoint와 `x/√N`의 비교를 부호 있는 제곱으로 환원한다. 소수 근삿값을 정답으로 쓰지 않으며 참조와 endpoint의 단순 overlap만 확인하지 않는다. 0은 정확한 점이어야 하고, 모든 구간에는 폭 관문을 적용한다.

독립 SymPy52준비의 여섯 연산에서 계수1248개·부호312개를 대조했다. Norm1·3/2·10⁻⁶에 대한156정규화 구간도 검사했다. Pell 및 nested cancellation, mixed radicals, zero numerator를 포함한다.

## 입력에서 도출하는 유한 계산

국소 ladder와 Kronecker 곱으로 전체 연산자를 만들고 H·drive/hopping의 복소 adjoint·에너지 모멘트·잔차·density·γ·number 분포/보존을 계산한다. 제품의 전체 basis step 생성과 구별되는 경로다. 국소 density는 직접 partial trace하고 covariance는 실제 symmetrized XP/PX로 대조한다.

Cutoff C의 교환자 `I−(C+1)P_top`, trace0, 무한 CCR과의 차이, 상단 확률·경계 보정, 실제 commutator를 쓰는 불확정성 bound와 probe의 두 순서를 모두 검사한다. C=0의 영 quadrature와 상단에서 음수가 되는 commutator 평균도 유지한다. 대각 spectrum은 유한 투영 Fock 공간에 한정하며 비대각 spectrum·무한 공간 바닥 주장은 허용하지 않는다.

모든 소유 field/배열·입력·ID/key·기저/차원·관측 metadata·부호·전체 수학 결과를 의미 트리와 대조한 뒤 동결 복사본을 반환한다. 공급된0잔차는 수학의 oracle가 아니다.

## 공통 경계의 보완

최대 raw 준비의 실제 descriptor 방문은396196회, 깊이13, JSON 약8.58MB였다. 기존250000회 한도는 정상 최대 결과를 거부하므로500000회로 조정했다. 깊이64·배열65536항·record512항·문자열8192자의 다른 제한은 유지하고, 방문 상한을 넘는 배열 검사를524288개 primitive로 갱신했다.

공통 matcher는 구간의0교차·상대 폭2⁻⁴⁸ 기준으로 해결/미해결 metadata를 대조한다. 좁은 정상 구간을 미해결로 바꿔 value/log 검사를 숨길 수 없다. Spin·fermion에서도 이 변조를 거부하고 이전87정상 준비를 새 브라우저 번들로 재검증했다.

## 검사와 한계

새94검사(raw 응답41·field53),593변조 거부와 기존319회귀(Worker214·spin51·fermion54), 타입·lint를 확인했다. Raw33독립 준비의 정확201702·방향성2079관측을 대조한다. 생성/종료한 Native Worker120개에서 raw33·이전87의 브라우저 검증기 수락, wrong ID120개·오염 metadata120개 거부와 전체 기준 재대조를 수행했다.

회귀 실행이 생성한 최신 dependency graph는 이번 runtime에 보존하고, 이전 단계의 graph는 baseline hash와 일치하는 원래 bytes로 복원했다. 과거 증거를 현재 결과로 바꿔 쓰지 않는다.

격리 preview-origin/Blob Worker이며 실제 reader·생산 자산·controller 검증은 아니다. 최대 raw 검증 시간은 약1.12초였다. 실제 학습 화면에 연결하기 전에 main thread 반응성 관문을 해결해야 하며, 계산이 맞다는 이유로 성능까지 통과 처리하지 않는다. 다음은 coherent·MPS·Schmidt 응답, 이어 controller/hooks·학습 화면·reader다.
