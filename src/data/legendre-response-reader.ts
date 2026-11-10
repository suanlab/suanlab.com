import type { LearningGuide, LearningUnit } from './interactive-book';

export const legendreResponseUnitId = 'connected-generators-and-legendre-effective-actions' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const legendreResponseUnit: LearningUnit = {
  id: legendreResponseUnitId,
  title: t('연결 생성함수와 르장드르 유효작용', 'Connected Generators and Legendre Effective Actions'),
  subtitle: t('외부장으로부터 평균과 공분산을 읽고, 평균 좌표에서의 유효 곡률을 구합니다.', 'Read means and covariances from sources, then recover effective curvature in mean coordinates.'),
  prerequisites: t('사차 깁스 장, 공분산과 볼록함수의 접선을 복습하세요.', 'Review quartic Gibbs fields, covariance, and supporting tangents of convex functions.'),
  prerequisiteIds: ['finite-quartic-scalar-gibbs-fields', 'finite-gaussian-field-sources-connected-correlations'],
  eyebrow: 'STATISTICAL FIELD THEORY',
  formula: 'm = ∇W(h), ∇²Γ(m) = [∇²W(h)]⁻¹',
  accent: 'block-order',
  questions: [
    {
      id: 'legendre-connected-second',
      question: t('로그 정규화 함수의 이차 미분은 무엇인가요?', 'What is the second derivative of the log normalizer?'),
      options: [t('분산', 'The variance'), t('평균의 제곱', 'The squared mean'), t('외부장의 제곱', 'The squared source')],
      answer: 0,
      explanation: t('첫 미분은 평균입니다. 그 비율을 다시 미분할 때 정규화 항이 평균의 제곱을 빼므로 분산을 얻습니다.', 'The first derivative is the mean. Differentiating its normalization subtracts the squared mean, giving the variance.'),
    },
    {
      id: 'legendre-dual-curvature',
      question: t('스핀이 포화에 접근해 감수율이 0에 가까워질 때 유효 곡률은 어떻게 되나요?', 'How does effective curvature behave when the spin approaches saturation and susceptibility tends to zero?'),
      options: [t('증가한다.', 'It increases.'), t('0으로 간다.', 'It tends to zero.'), t('일정하다.', 'It stays constant.')],
      answer: 0,
      explanation: t('곡률은 감수율의 역수입니다. 같은 평균 변화에 더 큰 외부장 변화가 필요합니다.', 'Curvature is the reciprocal susceptibility. The same change in mean requires a larger source change.'),
    },
  ],
};

export const legendreResponseGuide: LearningGuide = {
  intuition: t('생성함수에서 접선의 기울기는 평균입니다. 그 기울기를 새 좌표로 쓰면 유효작용에서 외부장을 읽을 수 있습니다.', 'The tangent slope of the generator is the mean. Using that slope as the new coordinate lets you recover the source from the effective action.'),
  steps: [
    {
      title: t('생성함수에서 평균 읽기', 'Read the mean from the generator'),
      explanation: t('외부장을 바꾸면 접선이 움직입니다. 그 지점의 기울기가 평균입니다.', 'Changing the source moves the point of tangency. The slope there equals the mean.'),
      formula: 'm = W′(h)',
      diagram: t('선택한 외부장에서 생성함수와 접선이 만납니다.', 'The tangent touches the generator at the selected source.'),
    },
    {
      title: t('유효작용에서 외부장 읽기', 'Recover the source from the effective action'),
      explanation: t('같은 상태를 평균 좌표에서 보면 유효작용의 기울기가 외부장입니다.', 'At the same state in mean coordinates, the effective action has slope equal to the source.'),
      formula: 'h = Γ′(m)',
      diagram: t('생성함수와 유효작용의 두 점은 같은 상태를 나타냅니다.', 'The points on the two graphs represent the same state.'),
    },
  ],
  experiments: [
    {
      title: t('가우시안과 스핀의 포화', 'Gaussian response and spin saturation'),
      predict: t('외부장이 커질 때 어느 모형의 평균이 먼저 포화될까요?', 'Which mean saturates as the source increases?'),
      action: t('가우시안과 스핀에서 h를 0에서 4까지 옮겨 χ와 Γ″를 비교하세요.', 'Move h from 0 to 4 in the Gaussian and spin models, and compare χ and Γ″.'),
      explain: t('가우시안의 응답은 일정합니다. 스핀 평균은 1에 접근하고 감수율이 낮아집니다.', 'Gaussian susceptibility is constant. The spin mean approaches one while susceptibility falls.'),
      preset: 0,
    },
    {
      title: t('사차 모형의 원점 곡률', 'Curvature of the quartic model at zero'),
      predict: t('미시 작용의 원점 곡률이 −1일 때 유효작용의 원점 곡률은 어떤 부호일까요?', 'If the microscopic curvature at zero is −1, what is the sign of the effective curvature there?'),
      action: t('사차 깁스 장을 선택하고 h=0의 χ와 Γ″를 확인하세요.', 'Choose the quartic Gibbs field and inspect χ and Γ″ at h=0.'),
      explain: t('유한 깁스 분포는 두 우물에 걸쳐 양의 분산을 갖습니다. 그 역수인 유효 곡률은 양수입니다.', 'The finite Gibbs distribution fluctuates across both wells, giving positive variance and positive reciprocal curvature.'),
      preset: 0,
    },
  ],
};
