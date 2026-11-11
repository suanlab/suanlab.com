import type { LearningGuide, LearningUnit } from './interactive-book';

export const greatCircleGeodesicId = 'geodesics-straight-lines-and-great-circles' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const greatCircleGeodesicUnit: LearningUnit = {
  id: greatCircleGeodesicId,
  title: t('측지선, 극좌표 직선과 대원', 'Geodesics, Polar Straight Lines and Great Circles'),
  subtitle: t('공변 가속도가 0인 경로와 전역 최단 경로를 비교합니다.', 'Compare paths of zero covariant acceleration with globally shortest paths.'),
  prerequisites: t('공변 미분과 평행이동을 복습하세요.', 'Review covariant derivatives and parallel transport.'),
  prerequisiteIds: ['curvature-covariant-derivatives-and-holonomy'],
  eyebrow: 'GEODESICS',
  formula: '∇Ẋ Ẋ = 0',
  accent: 'block-order',
  questions: [
    {
      id: 'geodesic-coordinate-acceleration',
      question: t('평면의 극좌표에서 직선의 r̈는 항상 0인가요?', 'Is r̈ always zero on a straight line in polar coordinates?'),
      options: [t('접속항과 상쇄되는 값이 생길 수 있습니다.', 'It can be nonzero and cancel the connection term.'), t('항상 0입니다.', 'It is always zero.'), t('직선에는 극좌표가 없습니다.', 'A straight line has no polar coordinates.')],
      answer: 0,
      explanation: t('측지선에서 0이 되는 것은 공변 가속도입니다.', 'The covariant acceleration vanishes along a geodesic.'),
    },
    {
      id: 'geodesic-long-arc',
      question: t('같은 두 점을 잇는 긴 대원 호도 측지선인가요?', 'Is the long great-circle arc between the same endpoints a geodesic?'),
      options: [t('예. 접선 방향 가속도가 0입니다.', 'Yes. Its tangential acceleration vanishes.'), t('아니요. 가장 짧은 호만 가능합니다.', 'No. Only the shortest arc can be a geodesic.'), t('길이가 0일 때만 가능합니다.', 'Only when its length is zero.')],
      answer: 0,
      explanation: t('측지선 식은 경로의 국소 조건입니다. 전역 최단 여부는 다른 경로와 비교해서 판단합니다.', 'The geodesic equation is local. Global minimization compares distinct paths.'),
    },
  ],
};

export const greatCircleGeodesicGuide: LearningGuide = {
  intuition: t('같은 출발점과 도착점을 잇는 두 대원 호는 모두 측지선이지만 길이는 다를 수 있습니다.', 'Two great-circle arcs can connect the same endpoints as geodesics and still have different lengths.'),
  steps: [{
    title: t('접선 방향 가속도', 'Tangential acceleration'),
    explanation: t('구면에서 중심을 향한 가속도는 접평면으로 투영하면 0입니다.', 'On the sphere, acceleration toward the center projects to zero in the tangent plane.'),
    formula: '∇Ẋ Ẋ = 0',
    diagram: t('대원 평면을 바꾸어도 같은 측지선 조건을 만족합니다.', 'Changing the great-circle plane preserves the geodesic condition.'),
  }],
  experiments: [
    {
      title: t('짧은 호와 긴 호', 'Short and long arcs'),
      predict: t('두 호는 끝점에서 일치할까요?', 'Do the two arcs reach the same endpoint?'),
      action: t('같은 분리각에서 호를 바꾸고 진행률을 1로 두어 끝점과 길이를 비교하세요.', 'At fixed separation, set progress to one and compare the endpoint and length of each arc.'),
      explain: t('두 호의 길이 합은 대원의 둘레이며 끝점은 같습니다.', 'The lengths add to the great-circle circumference and the endpoint is shared.'),
      preset: 0,
    },
    {
      title: t('대척점과 초기 방향', 'Antipodes and initial directions'),
      predict: t('분리각이 180도일 때 대원 평면을 바꾸면 중간점도 같을까요?', 'At 180° separation, do different great-circle planes have the same midpoint?'),
      action: t('분리각 180도와 진행률 0.5에서 대원 평면의 각도를 바꾸세요.', 'At 180° separation and progress 0.5, change the great-circle plane angle.'),
      explain: t('중간점은 바뀌고 호의 길이와 끝점은 유지됩니다.', 'The midpoint changes while the length and endpoints stay fixed.'),
      preset: 1,
    },
  ],
};
