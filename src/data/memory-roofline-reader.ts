import type { LearningGuide, LearningUnit } from './interactive-book';

export const memoryRooflineId = 'memory-traffic-and-the-roofline' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const memoryRooflineUnit: LearningUnit = {
  id: memoryRooflineId,
  title: t('메모리 이동과 Roofline', 'Memory Traffic and the Roofline'),
  subtitle: t('같은 연산을 할 때 데이터 이동량이 실행 시간에 주는 영향을 계산합니다.', 'Calculate how data movement affects the runtime of a fixed computation.'),
  prerequisites: t('처리율, 물리 단위와 차원 분석을 복습하세요.', 'Review throughput, physical units, and dimensional analysis.'),
  prerequisiteIds: ['photodiode-noise-and-equivalent-bandwidth'],
  eyebrow: 'LATENCY AND MEMORY TRAFFIC',
  formula: 'P_roof = min(P, BI)',
  accent: 'block-order',
  questions: [
    {
      id: 'roofline-traffic',
      question: t('같은 연산 수에서 외부 메모리 전송량이 절반이 되면 연산 집약도는?', 'At fixed operation count, what happens to intensity if external traffic halves?'),
      options: [t('두 배가 됩니다.', 'It doubles.'), t('절반이 됩니다.', 'It halves.'), t('같습니다.', 'It stays the same.')],
      answer: 0,
      explanation: t('집약도는 연산 수를 경계를 통과한 바이트 수로 나눈 값입니다.', 'Intensity is the number of operations divided by bytes crossing the boundary.'),
    },
    {
      id: 'roofline-bandwidth',
      question: t('집약도가 교차점보다 낮으면 어느 상한이 처리율을 제한하나요?', 'Which ceiling limits throughput below the ridge intensity?'),
      options: [t('대역폭과 집약도의 곱', 'Bandwidth times intensity'), t('최대 연산 처리율', 'Peak compute throughput'), t('실행 시간의 역수', 'The inverse of runtime')],
      answer: 0,
      explanation: t('교차점 아래에서는 BI가 최대 연산 처리율보다 작습니다.', 'Below the ridge, BI is smaller than peak compute throughput.'),
    },
  ],
};

export const memoryRooflineGuide: LearningGuide = {
  intuition: t('같은 연산이라도 각 바이트를 더 많이 재사용하면 전송 시간이 줄어듭니다.', 'More reuse of each transferred byte reduces transfer time for the same computation.'),
  steps: [{
    title: t('연산과 전송의 하한', 'Compute and transfer time floors'),
    explanation: t('연산과 전송이 완전히 겹쳐도 더 오래 걸리는 과정은 끝날 때까지 기다려야 합니다.', 'Even with ideal overlap, the slower of arithmetic and transfer must finish.'),
    formula: 't_min = max(F/P, D/B)',
    diagram: t('기울어진 대역폭 상한과 수평 연산 상한이 만납니다.', 'The sloped bandwidth ceiling meets the flat compute ceiling.'),
  }],
  experiments: [
    {
      title: t('전송량 줄이기', 'Reduce bytes transferred'),
      predict: t('연산 수를 유지하고 전송량을 100 MB에서 20 MB로 줄이면 병목은 어떻게 될까요?', 'How does the bottleneck change when traffic drops from 100 MB to 20 MB at fixed operation count?'),
      action: t('처리율을 8 TFLOP/s, 대역폭을 200 GB/s로 두고 전송량을 줄이세요.', 'Keep 8 TFLOP/s and 200 GB/s, and reduce bytes transferred.'),
      explain: t('집약도가 10에서 50 FLOP/byte로 올라가 교차점 40을 넘습니다.', 'Intensity rises from 10 to 50 FLOP/byte, passing the ridge at 40.'),
      preset: 0,
    },
    {
      title: t('교차점에서 대역폭 늘리기', 'Raise bandwidth at the ridge'),
      predict: t('전송량 25 MB에서 대역폭을 더 늘리면 시간 하한은 계속 낮아질까요?', 'With 25 MB transferred, will higher bandwidth keep lowering the runtime floor?'),
      action: t('전송량을 25 MB로 두고 대역폭을 200에서 400 GB/s로 올리세요.', 'Set traffic to 25 MB and increase bandwidth from 200 to 400 GB/s.'),
      explain: t('연산 시간이 0.125 ms여서 그 아래로는 내려가지 않습니다.', 'Compute time remains 0.125 ms, which sets the floor.'),
      preset: 1,
    },
  ],
};
