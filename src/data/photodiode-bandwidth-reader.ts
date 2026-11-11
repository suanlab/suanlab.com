import type { LearningGuide, LearningUnit } from './interactive-book';

export const photodiodeBandwidthId = 'photodiode-circuits-and-sensor-bandwidth' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const photodiodeBandwidthUnit: LearningUnit = {
  id: photodiodeBandwidthId,
  title: t('광다이오드 회로와 센서 대역폭', 'Photodiode Circuits and Sensor Bandwidth'),
  subtitle: t('입사광이 전압으로 바뀌는 과정에서 이득과 응답 속도를 계산합니다.', 'Calculate gain and response speed as incident light becomes a voltage.'),
  prerequisites: t('선형 미분방정식과 주파수 응답을 복습하세요.', 'Review linear differential equations and frequency response.'),
  prerequisiteIds: ['electronic-bands-in-a-dimerized-chain'],
  eyebrow: 'SEMICONDUCTORS AND SENSORS',
  formula: 'f_c = 1/(2πRC)',
  accent: 'block-order',
  questions: [
    {
      id: 'photodiode-load-tradeoff',
      question: t('수동 읽기 회로의 저항을 두 배로 하면 차단 주파수는 어떻게 되나요?', 'What happens to the passive readout cutoff when the resistance doubles?'),
      options: [t('절반이 됩니다.', 'It halves.'), t('두 배가 됩니다.', 'It doubles.'), t('같습니다.', 'It stays the same.')],
      answer: 0,
      explanation: t('RC 시간 상수가 두 배가 되므로 차단 주파수는 절반이 됩니다.', 'The RC time constant doubles, so the cutoff halves.'),
    },
    {
      id: 'photodiode-calibration',
      question: t('센서의 시간 응답을 추정하려면 출력 전압과 함께 무엇을 기록할까요?', 'What should be recorded along with the output voltage when estimating the sensor response?'),
      options: [t('교정한 광 입력의 시간 파형', 'The calibrated optical input waveform'), t('센서 이름만', 'The sensor name alone'), t('출력의 표본 수만', 'Only the output sample count')],
      answer: 0,
      explanation: t('입력 파형을 알면 광원의 응답과 읽기 회로의 응답을 구분할 수 있습니다.', 'Knowing the input waveform helps separate the light-source response from the readout response.'),
    },
  ],
};

export const photodiodeBandwidthGuide: LearningGuide = {
  intuition: t('축전기에 전하가 쌓이는 동안 출력 전압은 광 입력의 변화를 뒤따릅니다.', 'As charge accumulates on the capacitor, output voltage follows changes in optical input.'),
  steps: [{
    title: t('한 개의 시간 상수', 'One time constant'),
    explanation: t('RC가 클수록 시간 응답은 느려지고, 더 낮은 주파수부터 진폭이 줄어듭니다.', 'A larger RC slows the step response and causes attenuation to begin at lower frequencies.'),
    formula: 'τ = RC,  f_c = 1/(2πτ)',
    diagram: t('계단 응답과 주파수 응답이 같은 시간 상수를 공유합니다.', 'The step and frequency responses share the same time constant.'),
  }],
  experiments: [
    {
      title: t('부하 저항 두 배', 'Double the load resistance'),
      predict: t('저항을 100 kΩ에서 200 kΩ으로 늘리면 최종 전압과 차단 주파수는 어떻게 변할까요?', 'What happens to final voltage and cutoff when resistance increases from 100 kΩ to 200 kΩ?'),
      action: t('정전용량을 50 pF로 두고 저항을 200 kΩ으로 바꾸세요.', 'Keep capacitance at 50 pF and set resistance to 200 kΩ.'),
      explain: t('최종 전압은 두 배, 차단 주파수는 절반이 됩니다.', 'The final voltage doubles and the cutoff halves.'),
      preset: 0,
    },
    {
      title: t('정전용량 절반', 'Halve the capacitance'),
      predict: t('정전용량만 줄이면 최종 전압이 달라질까요?', 'Does reducing capacitance alone change the final voltage?'),
      action: t('저항을 100 kΩ으로 되돌리고 정전용량을 25 pF로 낮추세요.', 'Restore 100 kΩ and reduce capacitance to 25 pF.'),
      explain: t('최종 전압은 0.5 V로 같고 차단 주파수는 약 63.7 kHz로 두 배가 됩니다.', 'The final voltage stays at 0.5 V and the cutoff doubles to about 63.7 kHz.'),
      preset: 1,
    },
  ],
};
