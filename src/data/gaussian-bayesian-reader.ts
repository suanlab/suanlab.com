import type { LearningGuide, LearningUnit } from './interactive-book';

export const gaussianBayesianId = 'gaussian-bayesian-inversion' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const gaussianBayesianUnit: LearningUnit = {
  id: gaussianBayesianId,
  title: t('가우스 베이즈 역문제', 'Gaussian Bayesian Inversion'),
  subtitle: t('사전 분포와 관측을 결합해 추정값과 예측 불확실성을 계산합니다.', 'Combine a prior and observation to estimate a parameter and predictive uncertainty.'),
  prerequisites: t('가우스 분포, 베이즈 규칙과 이차 벌점을 복습하세요.', 'Review Gaussian distributions, Bayes’ rule, and quadratic penalties.'),
  prerequisiteIds: ['tikhonov-regularization-and-noise-gain'],
  eyebrow: 'BAYESIAN INVERSE PROBLEMS',
  formula: 'v⁻¹ = v₀⁻¹ + s²/r',
  accent: 'block-order',
  questions: [
    {
      id: 'bayes-inverse-no-signal',
      question: t('관측 강도 s=0이면 사후 분포는 어떻게 되나요?', 'What is the posterior if the observation strength is zero?'),
      options: [t('사전 분포와 같습니다.', 'It equals the prior.'), t('항상 분산이 0입니다.', 'Its variance is always zero.'), t('항상 관측값에 집중됩니다.', 'It always concentrates at the observation.')],
      answer: 0,
      explanation: t('우도가 x에 대해 상수이므로 사전 밀도의 모양을 바꾸지 않습니다.', 'The likelihood is constant in x, preserving the prior density.'),
    },
    {
      id: 'bayes-inverse-predictive-noise',
      question: t('새 관측의 예측 분산에는 어떤 항이 더해지나요?', 'What extra contribution enters the predictive variance of a new reading?'),
      options: [t('새 관측의 잡음 분산', 'Noise variance of the new reading'), t('사후 평균', 'The posterior mean'), t('관측 개수', 'The number of observations')],
      answer: 0,
      explanation: t('새 관측은 추정한 매개변수의 불확실성과 새 잡음을 모두 포함합니다.', 'A new reading includes both parameter uncertainty and fresh observation noise.'),
    },
  ],
};

export const gaussianBayesianGuide: LearningGuide = {
  intuition: t('관측은 모형과 일치하는 매개변수에 더 큰 확률을 부여합니다.', 'The observation gives more posterior weight to parameters that explain the reading.'),
  steps: [{
    title: t('정밀도 더하기', 'Add the precisions'),
    explanation: t('사전 분포의 정밀도와 센서가 제공하는 정밀도를 더합니다.', 'Add prior precision and the precision provided by the sensor.'),
    formula: 'v⁻¹ = v₀⁻¹ + s²/r',
    diagram: t('관측 잡음이 작아지면 사후 분포가 좁아집니다.', 'Smaller observation noise narrows the posterior for nonzero sensor strength.'),
  }],
  experiments: [
    {
      title: t('정보가 없는 센서', 'An uninformative sensor'),
      predict: t('센서 강도가 0이면 사후 곡선은 어떻게 될까요?', 'How does setting sensor strength to zero affect the posterior curve?'),
      action: t('s를 0으로 낮추고 두 밀도를 비교하세요.', 'Lower s to zero and compare the densities.'),
      explain: t('관측이 x와 무관해 사전 곡선과 사후 곡선이 겹칩니다.', 'The observation is independent of x, so the prior and posterior overlap.'),
      preset: 0,
    },
    {
      title: t('관측 잡음 줄이기', 'Reduce observation noise'),
      predict: t('s가 양수일 때 잡음을 줄이면 사후 분산은 어떻게 될까요?', 'For positive s, what happens to posterior variance as noise decreases?'),
      action: t('s=0.5에서 관측 잡음 표준편차를 줄이세요.', 'At s=0.5, decrease observation noise standard deviation.'),
      explain: t('센서 정밀도가 커지면서 사후 분산이 줄어듭니다.', 'Higher sensor precision reduces posterior variance.'),
      preset: 1,
    },
  ],
};
