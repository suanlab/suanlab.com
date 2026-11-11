import type { LearningGuide, LearningUnit } from './interactive-book';

export const rotationalCurvatureId = 'curvature-covariant-derivatives-and-holonomy' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const rotationalCurvatureUnit: LearningUnit = {
  id: rotationalCurvatureId,
  title: t('곡률, 공변 미분과 평행이동', 'Curvature, Covariant Derivatives and Parallel Transport'),
  subtitle: t('평면·구면·쌍곡 평면의 접속 계수와 한 바퀴 뒤 회전을 비교합니다.', 'Compare connection coefficients and loop rotations in the plane, sphere, and hyperbolic plane.'),
  prerequisites: t('계량, 접벡터와 곡선 좌표를 복습하세요.', 'Review metrics, tangent vectors, and curvilinear coordinates.'),
  prerequisiteIds: ['minkowski-metric-and-proper-time'],
  eyebrow: 'COVARIANT GEOMETRY',
  formula: 'K = −f″/f',
  accent: 'block-order',
  questions: [
    {
      id: 'rotational-curvature-polar-flat',
      question: t('극좌표 평면에서 접속 계수가 0이 아니면 곡률은 얼마인가요?', 'What is the curvature of a plane with nonzero polar connection coefficients?'),
      options: [t('0', '0'), t('항상 양수', 'Always positive'), t('항상 음수', 'Always negative')],
      answer: 0,
      explanation: t('극좌표의 접속 계수는 기저가 회전하는 효과를 포함합니다. f=r이라 f″=0입니다.', 'Polar connection coefficients account for the turning basis. Since f=r, its second derivative is zero.'),
    },
    {
      id: 'rotational-curvature-holonomy',
      question: t('작은 양의 방향 고리에서 음의 가우스 곡률은 어떤 잔여 회전을 주나요?', 'What signed rotation does negative curvature give around a small positively oriented loop?'),
      options: [t('음의 회전', 'Negative rotation'), t('양의 회전', 'Positive rotation'), t('곡률과 관계없이 0', 'Zero for every curvature')],
      answer: 0,
      explanation: t('이 일정한 곡률 모형에서 잔여 회전은 곡률과 면적의 곱입니다.', 'For these constant-curvature models the residual rotation is curvature times area.'),
    },
  ],
};

export const rotationalCurvatureGuide: LearningGuide = {
  intuition: t('기저가 회전하는 효과와 한 바퀴를 돌고도 남는 회전을 구분합니다.', 'Distinguish the turning of the basis from the residual rotation after a loop.'),
  steps: [{
    title: t('접속에서 곡률로', 'From connection to curvature'),
    explanation: t('접속 계수는 f와 f′를 포함하고 곡률은 −f″/f입니다.', 'The connection coefficients involve f and f′, while the curvature is −f″/f.'),
    formula: 'ds² = dr² + f(r)²dθ²',
    diagram: t('같은 좌표 표현에서 서로 다른 세 가지 곡면을 비교합니다.', 'Compare three different surfaces in the same coordinate form.'),
  }],
  experiments: [
    {
      title: t('극좌표 평면', 'The plane in polar coordinates'),
      predict: t('기저가 한 바퀴 회전하면 평행이동한 벡터가 원래 방향으로 돌아올까요?', 'If the basis makes a full turn, does the transported vector return to its starting direction?'),
      action: t('평면을 선택하고 한 바퀴 동안 기록한 각도와 잔여 회전을 비교하세요.', 'Choose the plane and compare the local angle after one loop with the residual rotation.'),
      explain: t('각도 변화 −2π는 처음 방향과 같습니다. 곡률에 의한 잔여 회전은 0입니다.', 'An angle change of −2π is the original direction. The curvature-induced residual rotation is zero.'),
      preset: 0,
    },
    {
      title: t('양과 음의 곡률', 'Positive and negative curvature'),
      predict: t('같은 길이 척도와 반지름에서 잔여 회전의 부호는 어떻게 달라질까요?', 'At the same scale and loop radius, how do the signs of the residual rotations compare?'),
      action: t('구면과 쌍곡 평면을 바꿔 곡률, 면적과 회전을 읽으세요.', 'Switch between the sphere and hyperbolic plane and inspect curvature, area and rotation.'),
      explain: t('면적은 양수이며 곡률의 부호가 잔여 회전의 방향을 정합니다.', 'Area is positive and the sign of curvature determines the direction of residual rotation.'),
      preset: 1,
    },
  ],
};
