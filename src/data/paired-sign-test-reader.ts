import type { LearningGuide, LearningUnit } from './interactive-book';

export const pairedSignTestId = 'seeds-uncertainty-and-paired-sign-tests' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const pairedSignTestUnit: LearningUnit = {
  id: pairedSignTestId,
  title: t('시드·불확실성과 쌍별 부호 검정', 'Seeds, Uncertainty and Paired Sign Tests'),
  subtitle: t('독립 단위를 정하고 부호 패턴을 정확히 열거해 꼬리확률을 계산합니다.', 'Define independent units and enumerate sign patterns to calculate an exact tail probability.'),
  prerequisites: t('이항 분포, 검정 가설과 쌍별 차이를 복습하세요.', 'Review binomial distributions, test hypotheses, and paired differences.'),
  prerequisiteIds: ['ablations-and-randomized-negative-controls'],
  eyebrow: 'UNCERTAINTY AND TESTING',
  formula: 'p = Σⱼ₌ₖⁿ C(n,j)2⁻ⁿ',
  accent: 'block-order',
  questions: [
    {
      id: 'sign-test-estimand',
      question: t('이 부호 검정이 직접 검사하는 양은?', 'What does this sign test directly examine?'),
      options: [t('쌍별 효과의 양의 부호 확률', 'The probability of a positive paired effect'), t('평균 효과의 크기', 'The magnitude of the mean effect'), t('예측 지연 시간', 'Prediction latency')],
      answer: 0,
      explanation: t('크기를 버리고 양수의 개수를 이항 모형과 비교합니다.', 'It uses the count of positive signs and a binomial null model.'),
    },
    {
      id: 'sign-test-independent-units',
      question: t('독립 재시작 6개의 연속 표본 600개를 얻었다면 무엇을 먼저 확인해야 하나요?', 'With 600 sequential readings from six independent restarts, what should be established first?'),
      options: [t('독립 반복의 단위와 사건 내 상관', 'The independent unit and within-episode dependence'), t('표본 600개 모두가 독립이라는 가정', 'Assume all 600 readings are independent'), t('가장 작은 p값을 주는 분할', 'The split yielding the smallest p-value')],
      answer: 0,
      explanation: t('사건 내 상관을 무시하면 독립 정보량을 과대평가할 수 있습니다.', 'Ignoring dependence within episodes can overstate the independent information.'),
    },
  ],
};

export const pairedSignTestGuide: LearningGuide = {
  intuition: t('반복 횟수와 유리한 부호 수를 알면 공정 부호 아래에서 얼마나 드문 결과인지 계산할 수 있습니다.', 'The repeat count and positive-sign count determine how rare the result is under fair signs.'),
  steps: [{
    title: t('귀무 부호의 꼬리', 'The tail under fair signs'),
    explanation: t('관측한 수 이상의 양수가 나올 확률을 모두 더합니다.', 'Add probabilities of obtaining at least as many positive signs as observed.'),
    formula: 'p = Pr[Binomial(n, 1/2) ≥ k]',
    diagram: t('양수 수의 분포와 포함하는 꼬리를 표시합니다.', 'Show the count distribution and the included tail.'),
  }],
  experiments: [
    {
      title: t('여섯 번과 다섯 번', 'Six positive signs versus five'),
      predict: t('6개 모두 양수인 경우와 5개가 양수인 경우의 꼬리는 얼마나 다를까요?', 'How different are the tails for six and five positives out of six?'),
      action: t('k를 6에서 5로 줄이세요.', 'Change k from six to five.'),
      explain: t('꼬리확률은 1/64에서 7/64로 커집니다.', 'The tail rises from 1/64 to 7/64.'),
      preset: 0,
    },
    {
      title: t('독립 반복 수', 'Number of independent repetitions'),
      predict: t('독립 반복을 더하고 모두 양수이면 꼬리확률은 어떻게 될까요?', 'What happens when more independent repetitions are added and all are positive?'),
      action: t('n을 12로 올리고 k도 12로 맞추세요.', 'Raise n and k to 12.'),
      explain: t('모든 부호가 양수일 확률은 2⁻¹²입니다.', 'The probability of all positive signs is 2⁻¹².'),
      preset: 1,
    },
  ],
};
