import type { LearningGuide, LearningUnit } from './interactive-book';

export const physicalValidationId = 'physical-computing-validation-by-ablation' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const physicalValidationUnit: LearningUnit = {
  id: physicalValidationId,
  title: t('물리적 계산 장치의 검증', 'Validating a Physical Computing Device'),
  subtitle: t('기저로 교정한 계산 장치를 중첩 입력으로 검증하고 오차의 원인을 비교합니다.', 'Calibrate a computing device on basis inputs and test explanations for errors on superpositions.'),
  prerequisites: t('행렬의 열, 선형성, 측정 잔차를 복습하세요.', 'Review matrix columns, linearity, and measurement residuals.'),
  prerequisiteIds: ['energy-bounds-and-system-measurements', 'photonic-interferometers-and-matrix-products'],
  eyebrow: 'DEVICE VALIDATION',
  formula: 'r = ỹ(x) − M_fit x',
  accent: 'block-order',
  questions: [
    {
      id: 'physical-validation-hidden-clipping',
      question: t('두 기저 출력이 정확해도 중첩 입력에서 오차가 나타날 수 있나요?', 'Can a superposition produce an error even when both basis outputs are exact?'),
      options: [t('중첩 출력이 읽기 범위를 넘으면 가능합니다.', 'Yes, if the combined output exceeds the readout range.'), t('어떤 장치에서도 불가능합니다.', 'No device can do this.'), t('오차는 기저 입력에서만 나타납니다.', 'Errors occur only for basis inputs.')],
      answer: 0,
      explanation: t('중첩으로 더 큰 출력이 생기면 기저 교정 때 보이지 않던 잘림이 드러납니다.', 'Superposition can produce a larger output and reveal clipping hidden during calibration.'),
    },
    {
      id: 'physical-validation-gain',
      question: t('고정된 선형 이득 오차는 기저 교정으로 어떻게 보이나요?', 'How does a fixed linear gain error appear after basis calibration?'),
      options: [t('교정 행렬이 이득을 포함하므로 검증 잔차가 작을 수 있습니다.', 'The fitted matrix absorbs the gain, so validation residuals can be small.'), t('항상 큰 비선형 잔차가 남습니다.', 'It always leaves a large nonlinear residual.'), t('교정 행렬의 모든 성분이 0이 됩니다.', 'All entries of the fitted matrix become zero.')],
      answer: 0,
      explanation: t('선형 이득은 교정한 열을 같은 배율로 바꿉니다.', 'A linear gain scales the fitted columns by the same factor.'),
    },
  ],
};

export const physicalValidationGuide: LearningGuide = {
  intuition: t('교정에 쓰지 않은 입력을 넣으면 제한된 시험에서는 보이지 않던 오차를 찾을 수 있습니다.', 'Held-out inputs reveal errors that a limited calibration can miss.'),
  steps: [{
    title: t('설정 출력과 교정 출력', 'Programmed and calibrated outputs'),
    explanation: t('측정값을 설정 행렬의 예측과 비교하고, 기저로 구한 행렬의 예측과도 비교합니다.', 'Compare the measurement with predictions from both the programmed and basis-fitted matrices.'),
    formula: 'r = ỹ(x) − M_fit x',
    diagram: t('설정 출력에 대한 오차와 교정 후 잔차를 두 채널에서 읽습니다.', 'Read programmed-output errors and calibrated residuals for both channels.'),
  }],
  experiments: [
    {
      title: t('기저 시험에서 숨은 잘림', 'Clipping hidden by basis tests'),
      predict: t('잘림 한계 0.8에서 기저와 보강 입력의 잔차는 어떻게 다를까요?', 'With a limit of 0.8, how do residuals differ for basis and constructive inputs?'),
      action: t('이득 1에서 기저 입력과 보강 입력을 번갈아 선택하세요.', 'At gain one, switch between basis and constructive inputs.'),
      explain: t('기저는 범위 안이지만 보강 입력의 목표 출력은 두 채널에서 1입니다.', 'Basis outputs are in range, while the constructive target reaches one in both channels.'),
      preset: 0,
    },
    {
      title: t('읽기 범위 가설 검증', 'Test the readout range'),
      predict: t('같은 보강 입력에서 한계를 올리면 잔차는 어떻게 바뀔까요?', 'At the same constructive input, how does a higher limit affect the residual?'),
      action: t('읽기 한계를 1 이상으로 올리세요.', 'Raise the readout limit to one or higher.'),
      explain: t('출력이 범위 안으로 들어와 잘림 잔차가 사라집니다.', 'The outputs now fit within the range, removing the clipping residual.'),
      preset: 1,
    },
  ],
};
