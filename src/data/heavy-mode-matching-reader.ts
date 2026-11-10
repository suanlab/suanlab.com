import type { LearningGuide, LearningUnit } from './interactive-book';

export const heavyModeMatchingId = 'effective-theory-heavy-mode-matching' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const heavyModeMatchingUnit: LearningUnit = {
  id: heavyModeMatchingId,
  title: t('유효 이론과 무거운 모드 정합', 'Effective Theory and Heavy-Mode Matching'),
  subtitle: t('무거운 모드를 적분하고 저주파 전개의 오차를 비교합니다.', 'Integrate out a heavy mode and compare errors in a low-frequency expansion.'),
  prerequisites: t('가우시안 이차형식, 주변화와 등비급수를 복습하세요.', 'Review Gaussian quadratic forms, marginalization, and geometric series.'),
  prerequisiteIds: ['renormalization-and-real-space-coarse-graining', 'finite-gaussian-field-sources-connected-correlations'],
  eyebrow: 'EFFECTIVE THEORY',
  formula: 'Qₗ(ω) = ω² + m² − g²/(ω² + M²)',
  accent: 'block-order',
  questions: [
    {
      id: 'heavy-mode-expansion-error',
      question: t('ω/M이 1/2일 때 남길 차수를 하나 올리면 커널 오차는 어떻게 바뀌나요?', 'At ω/M = 1/2, how does one more retained order change the kernel error?'),
      options: [t('1/4배가 됩니다.', 'It is multiplied by 1/4.'), t('2배가 됩니다.', 'It doubles.'), t('변화가 없습니다.', 'It stays the same.')],
      answer: 0,
      explanation: t('오차의 차수별 비는 x=(ω/M)²입니다.', 'The error ratio between successive orders is x=(ω/M)².'),
    },
    {
      id: 'heavy-mode-positive-kernel',
      question: t('정합한 커널이 음수인 주파수에서 어떤 해석이 가능한가요?', 'What can be inferred when the matched kernel is negative at a frequency?'),
      options: [t('그 주파수의 가우시안 응답을 정의할 수 없습니다.', 'The Gaussian response is undefined at that frequency.'), t('분산이 음수인 가우시안이 생깁니다.', 'The Gaussian has negative variance.'), t('전체 모형이 반드시 불안정합니다.', 'The full model must be unstable.')],
      answer: 0,
      explanation: t('절단한 커널이 양수가 아니면 가우시안 적분이 수렴하지 않습니다. 전체 모형의 안정성은 전체 이차형식으로 판정합니다.', 'The truncated Gaussian integral fails to converge for a nonpositive kernel. Stability of the full model is determined by its full quadratic form.'),
    },
  ],
};

export const heavyModeMatchingGuide: LearningGuide = {
  intuition: t('무거운 모드를 적분하면 가벼운 모드에 주파수에 따라 달라지는 보정이 생깁니다.', 'Integrating out the heavy mode produces a frequency-dependent correction to the light mode.'),
  steps: [
    {
      title: t('완전제곱과 유효 커널', 'Complete the square and read off the kernel'),
      explanation: t('무거운 좌표를 평행이동하면 두 좌표의 교차항이 사라지고 가벼운 좌표의 계수가 바뀝니다.', 'Shifting the heavy coordinate removes the mixed term and changes the light coefficient.'),
      formula: 'Qₗ = Q₁₁ − Q₁₂²/Q₂₂',
      diagram: t('무거운 좌표를 적분한 뒤 남은 가벼운 커널입니다.', 'The light kernel after integrating the heavy coordinate.'),
    },
  ],
  experiments: [
    {
      title: t('전개 반경 안과 밖', 'Inside and outside the expansion radius'),
      predict: t('차수를 올리면 주파수비 0.5와 2에서 오차가 각각 어떻게 바뀔까요?', 'How does a higher order change errors at frequency ratios 0.5 and 2?'),
      action: t('두 주파수비에서 차수를 바꿔 커널 오차를 비교하세요.', 'Change the order at both frequency ratios and compare kernel errors.'),
      explain: t('오차는 차수마다 주파수비의 제곱을 곱한 값입니다.', 'The error is multiplied by the squared frequency ratio at each order.'),
      preset: 0,
    },
    {
      title: t('부드러운 모드의 응답', 'Response of a soft mode'),
      predict: t('혼합비가 1에 가까우면 커널 오차와 응답 오차가 비슷할까요?', 'As the mixing ratio approaches one, will kernel and response errors stay comparable?'),
      action: t('작은 양의 주파수비에서 혼합비를 키워 커널 오차와 응답 오차를 비교하세요.', 'At a small positive frequency ratio, increase mixing and compare the kernel and response errors.'),
      explain: t('역수를 취할 때 두 커널의 곱으로 나누므로 작은 커널은 응답 오차를 키울 수 있습니다.', 'Taking reciprocals divides by the product of the two kernels, so a small kernel can amplify response error.'),
      preset: 1,
    },
  ],
};
