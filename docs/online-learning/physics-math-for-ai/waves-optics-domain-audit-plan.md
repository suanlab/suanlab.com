# 파동·진동·음향·광학 영역 종합 점검 계획

2026-11-05에 아래 계획을 수행했다. [실제 종합 점검 기록](waves-optics-domain-audit.md)을 참조한다. 계획 자체와 통과 근거를 구분하며 연구 심화·전체 교육과정은 미완료다.

| 원래 주제 | 실제 단원 | 개별 검증 기록 |
| --- | --- | --- |
| 조화진동·감쇠·공진 | `sinusoidal-forcing-transient-response`, `oscillator-frequency-response-power-resonance` | `docs/online-learning/physics-math-for-ai/waves-oscillations-authoring-status.md` |
| 결합 진동·정상모드 | `mass-weighted-coupled-normal-modes`, `coupled-mode-beating-energy-transfer` | `docs/online-learning/physics-math-for-ai/coupled-modes-authoring-status.md` |
| 파동 방정식 | `coupled-chain-continuum-wave-limit`, `wave-initial-data-reflection-energy-flux` | `docs/online-learning/physics-math-for-ai/wave-equation-authoring-status.md` |
| 중첩·간섭·회절 | `coherent-superposition-detector-averages`, `finite-aperture-fresnel-fraunhofer` | `.runtime/learning/interference-integrated-verification.json` |
| 분산·군속도 | `dispersive-wave-packets-phase-group-velocity`, `lattice-wave-packet-transport-dispersion` | `.runtime/learning/dispersion-integrated-verification.json` |
| 편광 | `jones-polarization-waveplates-analyzers`, `stokes-coherency-polarization-tomography` | `.runtime/learning/polarization-integrated-verification.json` |
| 기하·파동 광학 | `paraxial-ray-matrices-thin-lens-imaging`, `gaussian-beam-focusing-ray-wave-limit` | `.runtime/learning/ray-wave-optics-integrated-verification.json` |
| Fourier optics | `coherent-fourier-pupil-spatial-filtering`, `incoherent-psf-otf-image-formation` | `.runtime/learning/fourier-optics-integrated-verification.json` |
| 관측·샘플링·센서 | `pixel-aperture-sampling-aliasing`, `photon-counting-exposure-sensor-noise` | `.runtime/learning/sampling-sensors-integrated-verification.json` |

## 실제 실행할 점검

1. 원래9개 주제와18개 한영 원고·선수 관계·실험·수치 가정을 직접 대조한다. 기존 통과 manifest만으로 새로운 영역 검증을 대신하지 않는다.
2. 파동 전체 전용 테스트·독립 기준과 최신 소스의 관계를 확인한다. 물리적 특이점·영입력·위상 분기·양의 강도·전체 적분·단위·지원 근사의 의미를 비교한다.
3. 같은 최종 빌드에서18개 단원 모두 한영·모바일/데스크톱·명암 화면을 확인한다. 각 단원의 개념·안내·예제·입력 저장·퀴즈·표/그림 키보드 이동과 원본 그림을 점검한다. 필요한 범위의 실제 종료·이미지 근거를 남긴다.
4. 단원 간 이전/다음·목차·선수 링크와 용어 연결을 확인하고 영어 잔존·접근성·화면 넘침을 검사한다.
5. 모든 요구의 실제 근거를 확인한 뒤에만 `wholeDomainAuditComplete`를 갱신한다. 연구 심화는 미완료로 유지한다. 그 다음 전하·전류·장의 모델·원고를 작성한다.
