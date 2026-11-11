import type { LearningGuide, LearningUnit } from './interactive-book';

export const kalmanAssimilationId = 'scalar-kalman-data-assimilation' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const kalmanAssimilationUnit: LearningUnit = {
  id: kalmanAssimilationId,
  title: t('스칼라 칼만 자료동화', 'Scalar Kalman Data Assimilation'),
  subtitle: t('예측과 새 관측을 반복해서 결합하고 관측 공백의 영향을 살펴봅니다.', 'Combine forecasts and new observations in sequence, and inspect the effect of missing readings.'),
  prerequisites: t('가우스 조건부 분포와 이산 상태 갱신을 복습하세요.', 'Review Gaussian conditioning and discrete state updates.'),
  prerequisiteIds: ['discrete-adjoint-for-terminal-control', 'gaussian-bayesian-inversion'],
  eyebrow: 'DATA ASSIMILATION',
  formula: 'm⁺ = m⁻ + K(y − m⁻)',
  accent: 'block-order',
  questions: [
    {
      id: 'assimilation-missing-observation',
      question: t('이번 시각에 관측이 없으면 어떤 값을 다음 단계로 전달하나요?', 'Which estimate is carried forward when there is no observation?'),
      options: [t('이번 단계의 예측 평균과 분산', 'The current forecast mean and variance'), t('0인 평균과 분산', 'Zero mean and variance'), t('가장 최근 관측값 그대로', 'The last reading unchanged')],
      answer: 0,
      explanation: t('상태 갱신은 실행하고 관측을 이용한 분석만 건너뜁니다.', 'Run the state forecast and skip the observation analysis.'),
    },
    {
      id: 'assimilation-noisy-gain',
      question: t('예측 분산을 고정하고 관측 잡음 분산을 늘리면 이득은?', 'At fixed forecast variance, what happens to the gain as observation variance increases?'),
      options: [t('작아집니다.', 'It decreases.'), t('커집니다.', 'It increases.'), t('항상 1입니다.', 'It is always one.')],
      answer: 0,
      explanation: t('K=P⁻/(P⁻+r)에서 r이 커지면 분모가 커집니다.', 'A larger r increases the denominator of K=P⁻/(P⁻+r).'),
    },
  ],
};

export const kalmanAssimilationGuide: LearningGuide = {
  intuition: t('상태 모형이 다음 위치를 예측하고 관측이 예측을 교정합니다.', 'The state model forecasts the next value, and a reading corrects that forecast.'),
  steps: [{
    title: t('예측, 혁신, 분석', 'Forecast, innovation, analysis'),
    explanation: t('예측과 관측의 차이를 칼만 이득만큼 반영합니다.', 'Apply the Kalman gain to the difference between the observation and forecast.'),
    formula: 'K = P⁻/(P⁻ + r)',
    diagram: t('관측이 들어온 단계에서 분석 분산이 예측 분산보다 작습니다.', 'When an informative observation arrives, analysis variance is smaller than forecast variance.'),
  }],
  experiments: [
    {
      title: t('빠진 관측을 다시 넣기', 'Restore a missing observation'),
      predict: t('두 번째 관측을 포함하면 그 단계의 분석 분산은 어떻게 되나요?', 'What happens to the second-step analysis variance when its reading is included?'),
      action: t('두 번째 관측 포함을 켜고 단계 2의 두 분산을 비교하세요.', 'Include the second reading and compare the two variances at step two.'),
      explain: t('두 번째 분석 분산은 같은 단계의 예측 분산보다 작아집니다.', 'The second analysis variance becomes smaller than its forecast variance.'),
      preset: 0,
    },
    {
      title: t('관측을 덜 신뢰하기', 'Give the observation less weight'),
      predict: t('r을 높이면 첫 단계 분석 평균이 1에 가까워질까요, 0에 가까워질까요?', 'With higher r, does the first analysis move closer to one or to zero?'),
      action: t('단계 1을 선택하고 r을 높이세요.', 'Select step one and increase r.'),
      explain: t('관측 가중치가 작아져 분석이 예측 평균 0에 가까워집니다.', 'The observation gets less weight, so the analysis moves toward the zero forecast.'),
      preset: 1,
    },
  ],
};
