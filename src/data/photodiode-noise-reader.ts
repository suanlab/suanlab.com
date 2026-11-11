import type { LearningGuide, LearningUnit } from './interactive-book';

export const photodiodeNoiseId = 'photodiode-noise-and-equivalent-bandwidth' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const photodiodeNoiseUnit: LearningUnit = {
  id: photodiodeNoiseId,
  title: t('광다이오드 잡음과 등가 대역폭', 'Photodiode Noise and Equivalent Bandwidth'),
  subtitle: t('산탄 잡음과 열잡음을 필터에 통과시켜 측정 분산을 계산합니다.', 'Filter shot and thermal noise to calculate measurement variance.'),
  prerequisites: t('RC 필터의 주파수 응답과 확률변수의 분산을 복습하세요.', 'Review RC frequency response and variance.'),
  prerequisiteIds: ['photodiode-circuits-and-sensor-bandwidth'],
  eyebrow: 'MEASUREMENT NOISE',
  formula: 'B_eq = 1/(4RC)',
  accent: 'block-order',
  questions: [
    {
      id: 'photodiode-noise-enbw',
      question: t('RC 필터의 등가 잡음 대역폭이 차단 주파수보다 큰 이유는?', 'Why does an RC filter have a noise bandwidth larger than its cutoff?'),
      options: [t('차단점보다 높은 주파수의 잡음도 일부 통과합니다.', 'Some noise above the cutoff still passes.'), t('차단점에서 모든 잡음이 사라집니다.', 'All noise disappears at cutoff.'), t('저항이 열잡음을 없앱니다.', 'The resistor removes thermal noise.')],
      answer: 0,
      explanation: t('정규화한 이득의 제곱을 양의 주파수 전체에서 적분합니다.', 'The squared normalized gain is integrated over all positive frequencies.'),
    },
    {
      id: 'photodiode-noise-thermal',
      question: t('정전용량과 온도를 고정하고 저항을 늘리면 출력 열잡음 분산은 어떻게 되나요?', 'What happens to output thermal variance when resistance increases at fixed capacitance and temperature?'),
      options: [t('같습니다.', 'It stays the same.'), t('저항에 비례해 증가합니다.', 'It increases in proportion to resistance.'), t('항상 0이 됩니다.', 'It always drops to zero.')],
      answer: 0,
      explanation: t('출력 열잡음 분산은 kᵦT/C입니다.', 'Output thermal variance is kᵦT/C.'),
    },
  ],
};

export const photodiodeNoiseGuide: LearningGuide = {
  intuition: t('잡음 밀도는 주파수 한 구간당 분산을 나타냅니다. 필터를 통과한 값을 합하면 측정 분산을 얻습니다.', 'Noise density represents variance per unit frequency. Integrating after the filter gives measurement variance.'),
  steps: [{
    title: t('같은 넓이의 직사각형', 'A rectangle of equal area'),
    explanation: t('정규화한 전달함수 크기 제곱의 넓이와 같은 직사각형의 폭을 구합니다.', 'Find the width of a unit-height rectangle with the same area as the squared normalized transfer function.'),
    formula: 'B_eq = (π/2) f_c = 1/(4RC)',
    diagram: t('차단 주파수와 등가 잡음 대역폭이 서로 다른 위치에 놓입니다.', 'The cutoff and equivalent noise bandwidth occur at different locations.'),
  }],
  experiments: [
    {
      title: t('저항과 출력 분산', 'Resistance and output variance'),
      predict: t('저항을 두 배로 하면 어느 출력 잡음 분산이 변할까요?', 'Which output noise variance changes when resistance doubles?'),
      action: t('전체 전류를 1 µA로 유지하고 저항을 100 kΩ에서 200 kΩ으로 바꾸세요.', 'Keep the total current at 1 µA and change resistance from 100 kΩ to 200 kΩ.'),
      explain: t('산탄 잡음 분산은 두 배, 열잡음 분산은 같습니다.', 'Shot variance doubles while thermal variance stays fixed.'),
      preset: 0,
    },
    {
      title: t('열잡음 바닥', 'Thermal floor'),
      predict: t('전체 전류를 0으로 해도 어떤 잡음이 남을까요?', 'Which noise remains when the total current goes to zero?'),
      action: t('전체 전류를 0으로 설정하고 온도를 바꾸세요.', 'Set total current to zero and vary temperature.'),
      explain: t('산탄 잡음은 사라지고 열잡음 분산은 절대온도에 비례합니다.', 'Shot noise disappears and thermal variance scales with absolute temperature.'),
      preset: 1,
    },
  ],
};
