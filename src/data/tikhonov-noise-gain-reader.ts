import type { LearningGuide, LearningUnit } from './interactive-book';

export const tikhonovNoiseGainId = 'tikhonov-regularization-and-noise-gain' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const tikhonovNoiseGainUnit: LearningUnit = {
  id: tikhonovNoiseGainId,
  title: t('티호노프 정규화와 잡음 증폭', 'Tikhonov Regularization and Noise Gain'),
  subtitle: t('벌점이 약한 관측 모드의 편향과 분산에 미치는 영향을 계산합니다.', 'Calculate how a penalty changes bias and variance in a weak observation mode.'),
  prerequisites: t('특잇값, 최소제곱과 편향·분산을 복습하세요.', 'Review singular values, least squares, bias, and variance.'),
  prerequisiteIds: ['parameter-identifiability-and-sensor-design'],
  eyebrow: 'REGULARIZATION',
  formula: 'x̂ = sy/(s² + λ)',
  accent: 'block-order',
  questions: [
    {
      id: 'tikhonov-signal-retention',
      question: t('λ=s²이면 참 신호가 추정량 평균에 얼마나 남나요?', 'With λ=s², what fraction of the true signal survives in the mean estimate?'),
      options: [t('절반입니다.', 'One half'), t('전부입니다.', 'All of it'), t('0입니다.', 'Zero')],
      answer: 0,
      explanation: t('보존 계수 s²/(s²+λ)에 λ=s²를 대입합니다.', 'Substitute λ=s² into the retention factor s²/(s²+λ).'),
    },
    {
      id: 'tikhonov-larger-penalty',
      question: t('참 신호가 0이 아닐 때 벌점을 늘리면 어떤 효과가 생기나요?', 'What does increasing the penalty do when the true signal is nonzero?'),
      options: [t('잡음 분산이 줄고 편향 크기가 커집니다.', 'Noise variance shrinks and bias magnitude increases.'), t('두 값이 항상 동시에 0이 됩니다.', 'Both quantities always become zero.'), t('신호 보존율이 더 커집니다.', 'Signal retention increases.')],
      answer: 0,
      explanation: t('분모가 커지면서 잡음 증폭률과 신호 보존율이 모두 줄어듭니다.', 'A larger denominator reduces both noise gain and signal retention.'),
    },
  ],
};

export const tikhonovNoiseGainGuide: LearningGuide = {
  intuition: t('약한 방향을 그대로 뒤집으면 잡음도 커집니다. 벌점은 그 방향의 신호와 잡음을 함께 줄입니다.', 'Direct inversion amplifies noise in weak directions. A penalty shrinks both signal and noise there.'),
  steps: [{
    title: t('하나의 모드에 적용한 벌점', 'A penalty on one mode'),
    explanation: t('특잇값의 역수 대신 s/(s²+λ)를 관측에 곱합니다.', 'Multiply the observation by s/(s²+λ) to filter a single mode.'),
    formula: 'g = s/(s² + λ)',
    diagram: t('편향 제곱과 분산의 합이 평균제곱오차입니다.', 'Squared bias and variance add to mean squared error.'),
  }],
  experiments: [
    {
      title: t('절반 보존 지점', 'The half-retention setting'),
      predict: t('s=0.05에서 λ=0.0025를 주면 추정량의 잡음 표준편차는 어떻게 될까요?', 'What happens to the noise standard deviation at s=0.05 and λ=0.0025?'),
      action: t('λ를 0과 0.0025로 바꾸고 수치를 비교하세요.', 'Compare the readout at λ=0 and λ=0.0025.'),
      explain: t('잡음 표준편차가 1에서 0.5로 줄고 신호도 절반만 남습니다.', 'Noise standard deviation falls from one to 0.5, and half of the signal remains.'),
      preset: 0,
    },
    {
      title: t('벌점이 큰 구간', 'A large penalty'),
      predict: t('벌점을 계속 늘리면 평균제곱오차가 계속 줄어들까요?', 'Does mean squared error keep decreasing as the penalty grows?'),
      action: t('같은 s와 σ에서 λ를 최대로 올리세요.', 'Keep s and σ fixed, and raise λ to its maximum.'),
      explain: t('분산은 계속 줄지만 편향 제곱이 커져 합이 다시 늘어날 수 있습니다.', 'Variance keeps decreasing, while squared bias can raise the sum again.'),
      preset: 1,
    },
  ],
};
