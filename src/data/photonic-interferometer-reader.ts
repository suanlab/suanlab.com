import type { LearningGuide, LearningUnit } from './interactive-book';

export const photonicInterferometerId = 'photonic-interferometers-and-matrix-products' as const;
const t = (ko: string, en: string) => ({ ko, en });

export const photonicInterferometerUnit: LearningUnit = {
  id: photonicInterferometerId,
  title: t('광 간섭계와 행렬 곱', 'Photonic Interferometers and Matrix Products'),
  subtitle: t('위상으로 회전을 구현하고 진폭 감쇠와 검출 세기를 구분합니다.', 'Implement a rotation with phase and distinguish amplitude attenuation from detected intensity.'),
  prerequisites: t('복소수의 위상, 행렬 곱과 특잇값 분해를 복습하세요.', 'Review complex phase, matrix products, and singular value decomposition.'),
  prerequisiteIds: ['photodiode-circuits-and-sensor-bandwidth'],
  eyebrow: 'PHOTONIC COMPUTING',
  formula: 'M = R(θ)DR(−θ)',
  accent: 'block-order',
  questions: [
    {
      id: 'photonic-amplitude-to-power',
      question: t('진폭 투과율이 0.5인 채널의 파워 투과율은 얼마인가요?', 'What power fraction is transmitted by a channel with amplitude transmission 0.5?'),
      options: [t('0.25입니다.', '0.25'), t('0.5입니다.', '0.5'), t('1입니다.', '1')],
      answer: 0,
      explanation: t('파워는 진폭 크기의 제곱에 비례합니다.', 'Power is proportional to the squared magnitude of the amplitude.'),
    },
    {
      id: 'photonic-global-sign',
      question: t('모든 입력 진폭의 부호를 바꾸면 검출 세기는 어떻게 되나요?', 'What happens to detected intensities when all input amplitudes change sign?'),
      options: [t('각 검출 세기는 같습니다.', 'Each detected intensity stays the same.'), t('각 검출 세기의 부호가 바뀝니다.', 'Every detected intensity changes sign.'), t('첫 채널만 변합니다.', 'Only the first channel changes.')],
      answer: 0,
      explanation: t('선형 회로의 출력 진폭이 모두 반전되며, 제곱 크기는 같습니다.', 'All linear output amplitudes reverse, preserving their squared magnitudes.'),
    },
  ],
};

export const photonicInterferometerGuide: LearningGuide = {
  intuition: t('빛을 섞으면 진폭과 위상이 합쳐지고, 광검출에서는 세기가 관측됩니다.', 'Optical mixing combines amplitudes and phases, and a direct detector observes intensity.'),
  steps: [{
    title: t('회전, 감쇠, 역회전', 'Rotation, attenuation, and inverse rotation'),
    explanation: t('입력을 회전해 두 채널을 만든 뒤 한 채널을 감쇠하고 반대로 회전합니다.', 'Rotate the input into two channels, attenuate one channel, and rotate back.'),
    formula: 'M = R(θ) diag(1, d) R(−θ)',
    diagram: t('감쇠 정도와 회전 각도가 함께 출력 진폭을 결정합니다.', 'The attenuation and rotation angle jointly determine the output amplitudes.'),
  }],
  experiments: [
    {
      title: t('부호와 세기 비교', 'Compare sign and intensity'),
      predict: t('입력을 모두 반전하면 어떤 수치가 그대로일까요?', 'Which readings stay the same when the entire input is inverted?'),
      action: t('입력 부호 반전 버튼을 눌러 출력 진폭과 파워를 비교하세요.', 'Use the input sign button and compare output amplitudes and powers.'),
      explain: t('진폭은 반전되고 파워는 보존됩니다.', 'The amplitudes reverse while the powers stay the same.'),
      preset: 0,
    },
    {
      title: t('감쇠를 없애기', 'Remove attenuation'),
      predict: t('감쇠가 없으면 두 회전의 합성은 무엇이 될까요?', 'What is the combined action of the two rotations without attenuation?'),
      action: t('d를 1로 올리고 각도를 바꿔 보세요.', 'Raise d to one and change the angle.'),
      explain: t('두 회전은 서로 역연산이므로 입력 진폭이 복원됩니다.', 'The two rotations are inverses and recover the input amplitudes.'),
      preset: 1,
    },
  ],
};
