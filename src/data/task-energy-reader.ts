import type { LearningGuide, LearningUnit } from './interactive-book';

export const taskEnergyId = 'energy-per-task-and-thermal-cost' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const taskEnergyUnit: LearningUnit = {
  id: taskEnergyId,
  title: t('작업당 에너지와 열 비용', 'Energy per Task and Thermal Cost'),
  subtitle: t('전력을 적분해 작업당 에너지를 구하고 유휴 기준의 영향을 살펴봅니다.', 'Integrate power to estimate energy per task and inspect the effect of the idle baseline.'),
  prerequisites: t('정적분과 와트·줄의 관계를 복습하세요.', 'Review definite integrals and the relation between watts and joules.'),
  prerequisiteIds: ['memory-traffic-and-the-roofline'],
  eyebrow: 'ENERGY AND THERMAL COST',
  formula: 'E = ∫P(t) dt',
  accent: 'block-order',
  questions: [
    {
      id: 'task-energy-power-time',
      question: t('40 W로 0.20초 실행한 작업과 60 W로 0.10초 실행한 작업 중 총 에너지가 작은 쪽은?', 'Which run uses less gross energy, 40 W for 0.20 seconds or 60 W for 0.10 seconds?'),
      options: [t('60 W 실행', 'The 60 W run'), t('40 W 실행', 'The 40 W run'), t('두 실행이 같습니다.', 'Both are equal.')],
      answer: 0,
      explanation: t('60 W 실행은 6 J, 40 W 실행은 8 J를 사용합니다.', 'The 60 W run uses 6 J and the 40 W run uses 8 J.'),
    },
    {
      id: 'task-energy-idle-boundary',
      question: t('기준 유휴 전력을 바꾸면 어떤 수치가 변하나요?', 'Which reading changes when you adjust the idle baseline?'),
      options: [t('유휴 대비 추가 에너지', 'Incremental energy above idle'), t('측정한 총 에너지', 'Measured gross energy'), t('완료 작업 수', 'The completed task count')],
      answer: 0,
      explanation: t('유휴 대비 에너지는 총 에너지에서 기준 전력과 실행 시간의 곱을 뺀 값입니다.', 'Incremental energy subtracts baseline power times duration from gross energy.'),
    },
  ],
};

export const taskEnergyGuide: LearningGuide = {
  intuition: t('전력 곡선 아래의 넓이가 실행 동안 사용한 에너지입니다.', 'Area under a power trace gives the energy consumed during execution.'),
  steps: [{
    title: t('곡선 아래 넓이', 'Area under the trace'),
    explanation: t('불균일한 표본은 각 시간 간격에서 사다리꼴 넓이를 계산해 합칩니다.', 'For uneven samples, add the trapezoid area on each time interval.'),
    formula: 'E ≈ Σ (Pᵢ + Pᵢ₊₁) Δtᵢ / 2',
    diagram: t('높고 짧은 곡선의 넓이가 낮고 긴 곡선보다 작을 수 있습니다.', 'A tall, short trace can enclose less area than a low, long trace.'),
  }],
  experiments: [
    {
      title: t('총 에너지와 유휴 대비 에너지', 'Compare gross and incremental energy'),
      predict: t('두 실행의 총 에너지가 달라도 유휴 대비 에너지는 같을 수 있을까요?', 'Can runs with different gross energies have equal incremental energies?'),
      action: t('유휴 기준을 20 W로 두고 두 열을 비교하세요.', 'Set the idle baseline to 20 W and compare the two columns.'),
      explain: t('두 실행의 총 에너지는 8 J와 6 J이고 유휴 대비 에너지는 모두 4 J입니다.', 'Gross energies are 8 J and 6 J, and both incremental energies equal 4 J.'),
      preset: 0,
    },
    {
      title: t('완료한 작업으로 나누기', 'Normalize by completed tasks'),
      predict: t('측정 파형을 유지하고 완료 수를 100에서 200으로 올리면 어떤 값이 절반이 될까요?', 'Which values halve if completed tasks increase from 100 to 200 at fixed power traces?'),
      action: t('완료 작업 수를 200으로 올리세요.', 'Raise the completed task count to 200.'),
      explain: t('총량과 실행 시간은 같고 작업당 에너지만 절반이 됩니다.', 'Totals and durations are unchanged, and energy per task halves.'),
      preset: 1,
    },
  ],
};
