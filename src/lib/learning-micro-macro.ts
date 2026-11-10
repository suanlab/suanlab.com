/** Reader configuration only; calculators remain in the worker bundle. */
import { isMicroMacroInput, snapshotMicroMacroInput, type MicroMacroInput } from './learning-micro-macro-protocol';
import type { BinaryObservation } from './learning-binary-products';
export type MicroMacroKind = MicroMacroInput['kind'];
export type MicroMacroText = { ko: string; en: string };
const t = (ko: string, en: string): MicroMacroText => ({ ko, en });
export const isMicroMacroKind = (v: string): v is MicroMacroKind => ['binary-shell-multiplicity-block-counts', 'finite-macro-partition-conditional-entropy', 'finite-noisy-macro-observation-channel'].includes(v);
const binary = (sites: number, excitations: number, blockASites: number): MicroMacroInput => ({ kind: 'binary-shell-multiplicity-block-counts', parameters: { sites, excitations, blockASites, energyGap: 1e-21 } });
const partition = (trialWeights: number[], targetWeights: number[], mapping: number[], groupCount: number): MicroMacroInput => ({ kind: 'finite-macro-partition-conditional-entropy', parameters: { trialWeights, targetWeights, mapping, groupCount } });
const noisy = (preparationWeights: number[], channelWeights: number[][]): MicroMacroInput => ({ kind: 'finite-noisy-macro-observation-channel', parameters: { preparationWeights, channelWeights } });
export const microMacroTitles: Record<MicroMacroKind, MicroMacroText> = {
  'binary-shell-multiplicity-block-counts': t('배치·껍질·블록의 확률', 'Patterns, shells and blocks'),
  'finite-macro-partition-conditional-entropy': t('관측과 균등 재구성', 'Observation and uniform reconstruction'),
  'finite-noisy-macro-observation-channel': t('잡음 관측과 사후 정보', 'Noisy readings and posterior information'),
};
export const microMacroPresets: Record<MicroMacroKind, { label: MicroMacroText; input: MicroMacroInput }[]> = {
  'binary-shell-multiplicity-block-counts': [
    { label: t('네 자리·여섯 배치', 'Four sites, six patterns'), input: binary(4, 2, 2) },
    { label: t('두 블록의 요동', 'Two-block fluctuations'), input: binary(8, 4, 4) },
    { label: t('홀수 자리의 동률', 'Odd-size tied modes'), input: binary(9, 4, 4) },
    { label: t('빈 계', 'Empty system'), input: binary(0, 0, 0) },
    { label: t('희귀한 단일 배치', 'Rare single-pattern shell'), input: binary(1080, 0, 540) },
    { label: t('큰 정확한 계수', 'Large exact counts'), input: binary(4096, 2048, 2048) },
  ],
  'finite-macro-partition-conditional-entropy': [
    { label: t('숨은 편향', 'Hidden bias'), input: partition([6, 2, 1, 1], [4, 4, 1, 1], [0, 0, 1, 1], 2) },
    { label: t('같은 관측·다른 준비', 'Same reading, different preparation'), input: partition([8, 0, 1, 1], [4, 4, 1, 1], [0, 0, 1, 1], 2) },
    { label: t('KL 보존·상태는 숨음', 'Preserved KL, hidden states'), input: partition([4, 4, 1, 1], [1, 1, 4, 4], [0, 0, 1, 1], 2) },
    { label: t('숨은 지지집합 불일치', 'Hidden support mismatch'), input: partition([1, 1], [1, 0], [0, 0], 1) },
    { label: t('드러난 지지집합 불일치', 'Visible support mismatch'), input: partition([1, 1], [1, 0], [0, 1], 3) },
    { label: t('확률0 군과 빈 군', 'Zero-mass and empty groups'), input: partition([1, 0, 0], [1, 0, 0], [0, 1, 1], 3) },
    { label: t('한 ULP의 양수 증가량', 'Positive one-ULP gain'), input: partition([1, 1 + Number.EPSILON], [1, 1], [0, 0], 1) },
  ],
  'finite-noisy-macro-observation-channel': [
    { label: t('대칭 오류 .1', 'Symmetric error .1'), input: noisy([1, 1], [[9, 1], [1, 9]]) },
    { label: t('편향된 사전3:1', 'Biased prior 3:1'), input: noisy([3, 1], [[9, 1], [1, 9]]) },
    { label: t('지워짐1/4', 'Erasure probability 1/4'), input: noisy([1, 1], [[3, 0, 1], [0, 3, 1]]) },
    { label: t('독립적인 램프', 'Independent lamp'), input: noisy([1, 0], [[1, 1], [1, 1]]) },
    { label: t('언제나 반대 레이블', 'Always reversed labels'), input: noisy([3, 1], [[0, 1], [1, 0]]) },
    { label: t('표시 아래의 양수 정보', 'Positive information below probability range'), input: noisy([Number.MIN_VALUE, 1024], [[1, 0], [0, 1]]) },
    { label: t('사용하지 않은 입력·출력', 'Unused input and output'), input: noisy([1, 0], [[1, 0, 0], [1, 1, 0]]) },
  ],
};
export interface MicroMacroGuide { title: MicroMacroText; predict: MicroMacroText; action: MicroMacroText; explain: MicroMacroText; preset: number }
export const microMacroGuides: Record<MicroMacroKind, MicroMacroGuide[]> = {
  'binary-shell-multiplicity-block-counts': [
    { title: t('세 확률의 분모', 'Three probability denominators'), predict: t('배치1100과 K=2 껍질의 확률은 같을까요?', 'Do pattern 1100 and the K=2 shell have the same probability?'), action: t('첫 예제의 개수와 세 확률을 비교하세요.', 'Compare the counts and three probabilities in the first example.'), explain: t('전체 준비의 배치 확률1/16,껍질6/16,선택 껍질 내부1/6은 다른 질문입니다.', 'Full-space pattern probability 1/16, shell probability 6/16 and within-shell probability 1/6 answer different questions.'), preset: 0 },
    { title: t('총합 제약의 요동', 'Fluctuations under a total constraint'), predict: t('비상호작용 자리도 고정 K에서 독립일까요?', 'Are noninteracting sites independent after K is fixed?'), action: t('두 블록 예제에서 A의 분포와 분산을 보세요.', 'Inspect the block A distribution and variance.'), explain: t('개수1·16·36·16·1,분모70이며 분산은4/7입니다. 제약이 조건부 의존을 만듭니다.', 'Counts are 1,16,36,16,1 over 70; variance is 4/7. The constraint creates conditional dependence.'), preset: 1 },
    { title: t('희귀 확률과 실제0', 'Rare probability versus true zero'), predict: t('SB=0인 껍질은 전체 공간에서도 흔할까요?', 'Does SB=0 make a shell common in the full space?'), action: t('희귀 단일 배치의 확률·기여·SB를 비교하세요.', 'Compare rare-shell probability, contribution and SB.'), explain: t('2⁻¹⁰⁸⁰은 양수지만 선형 범위 밖입니다. 정보 기여는 양수이고 ln1=0인 SB와 다릅니다.', '2^-1080 is positive outside the linear range. Its positive information contribution differs from SB=0 because ln 1=0.'), preset: 4 },
  ],
  'finite-macro-partition-conditional-entropy': [
    { title: t('같은 histogram의 차이', 'What the same histogram hides'), predict: t('군 확률이 같으면 H(X)도 같을까요?', 'Does the same group histogram imply the same H(X)?'), action: t('숨은 편향과 다른 준비 예제를 비교하세요.', 'Compare hidden bias with the alternative preparation.'), explain: t('둘 다(.8,.2)를 관측하지만 내부6:2와8:0의 불확실성이 다릅니다.', 'Both yield (.8,.2), but internal 6:2 and 8:0 splits have different uncertainty.'), preset: 1 },
    { title: t('비교 보존과 상태 식별', 'Preserving comparisons versus identifying states'), predict: t('KL을 보존하면 상태도 알아낼까요?', 'Does preserving KL identify the state?'), action: t('KL 보존 예제의 숨은 정보와 숨은 KL을 비교하세요.', 'Compare hidden entropy with hidden KL in the preserved-KL example.'), explain: t('같은 조건부라 숨은 KL은0이지만 두 상태의 불확실성 ln2는 남습니다.', 'Matching conditionals give hidden KL zero, but ln 2 uncertainty about two states remains.'), preset: 2 },
    { title: t('관측에 드러나지 않는 무한대', 'Infinity hidden by observation'), predict: t('관측 KL=0이면 미시 KL도0일까요?', 'Does observed KL zero imply microstate KL zero?'), action: t('숨은 불일치와 드러난 불일치 예제를 비교하세요.', 'Compare hidden and visible support mismatch.'), explain: t('한 군이면 무한대가 내부에 숨고,나누면 군 KL부터 무한대입니다. 미정 조건부를 무한대끼리 빼지 않습니다.', 'One group hides infinity internally; separate groups reveal infinite macro KL. Undefined conditionals are not computed by subtracting infinities.'), preset: 3 },
  ],
  'finite-noisy-macro-observation-channel': [
    { title: t('행에서 열로', 'From rows to columns'), predict: t('채널.9가 결합 확률도.9일까요?', 'Does channel probability .9 mean joint probability .9?'), action: t('대칭 오류의 채널·결합·사후를 비교하세요.', 'Compare channel, joint and posterior tables.'), explain: t('균등 입력의 결합 질량은.45이며 출력 열로 나누면 사후.9를 얻습니다.', 'Fair-input joint mass is .45; dividing by its output column gives posterior .9.'), preset: 0 },
    { title: t('Bayes의 조건 방향', 'The direction of Bayes conditioning'), predict: t('눈금1을 읽으면 입력1 확률이 항상.9일까요?', 'After reading 1, is input 1 always .9 likely?'), action: t('사전3:1에서 출력1의 사후를 선택하세요.', 'Select output 1 under the 3:1 prior.'), explain: t('.075와.225를 합.3으로 나누면 입력1 사후는.75입니다.', 'Joint masses .075 and .225 sum to .3, giving posterior .75 for input 1.'), preset: 1 },
    { title: t('지워졌음을 아는 관측', 'Knowing that a reading was erased'), predict: t('모든 출력이 같은 양의 상태 정보를 남길까요?', 'Do all readings leave the same state uncertainty?'), action: t('지워짐 예제에서0·1·? 출력의 사후를 비교하세요.', 'Compare posteriors for 0,1 and ? in the erasure example.'), explain: t('0과1은 상태를 식별하지만?는 균등 사전을 남깁니다. 평균 숨은 정보는 ln2/4입니다.', '0 and 1 identify the state; ? leaves the fair prior. Average hidden information is ln 2/4.'), preset: 2 },
  ],
};
export const validMicroMacroSavedInput = (kind: MicroMacroKind, value: unknown): value is MicroMacroInput => isMicroMacroInput(value) && value.kind === kind;
export const validBinaryMacrostateInput = (value: unknown): value is MicroMacroInput => validMicroMacroSavedInput('binary-shell-multiplicity-block-counts', value);
export const validMacroPartitionInput = (value: unknown): value is MicroMacroInput => validMicroMacroSavedInput('finite-macro-partition-conditional-entropy', value);
export const validNoisyMacroInput = (value: unknown): value is MicroMacroInput => validMicroMacroSavedInput('finite-noisy-macro-observation-channel', value);
export const microMacroSavedValidators = { 'binary-shell-multiplicity-block-counts': validBinaryMacrostateInput, 'finite-macro-partition-conditional-entropy': validMacroPartitionInput, 'finite-noisy-macro-observation-channel': validNoisyMacroInput };
export function microMacroPresetInput(kind: MicroMacroKind, index: number) {
  if (!Number.isInteger(index) || !microMacroPresets[kind][index]) throw new RangeError('Invalid micro/macro preset');
  return snapshotMicroMacroInput(microMacroPresets[kind][index].input);
}
export function microMacroPlotValue(q: BinaryObservation | null): number | null { return q === null || q.value === null || !Number.isFinite(q.value) ? null : q.value; }
/** Enumerate actual labeled patterns only within a deliberately small display.
 * This never replaces model counts for the general N<=4096 domain.
 */
export function binaryPatternExamples(sites: number, excitations: number): string[] | null {
  if (!Number.isInteger(sites) || sites < 0 || sites > 12 || !Number.isInteger(excitations) || excitations < 0 || excitations > sites) return null;
  return Array.from({ length: 2 ** sites }, (_, i) => i.toString(2).padStart(sites, '0')).filter(s => (s.match(/1/g) ?? []).length === excitations).map(s => sites === 0 ? '∅' : s);
}
/** Fixed conceptual examples, independent of saved experiment settings. */
export const microMacroConceptPresetIndices: Record<MicroMacroKind, readonly number[]> = {
  'binary-shell-multiplicity-block-counts': [0, 0, 1, 4],
  'finite-macro-partition-conditional-entropy': [0, 1, 0, 2],
  'finite-noisy-macro-observation-channel': [0, 1, 2, 3],
};
export function microMacroConceptInput(kind: MicroMacroKind, step: number) {
  if (!Number.isInteger(step) || step < 0 || step >= 4) throw new RangeError('Invalid micro/macro concept step');
  return microMacroPresetInput(kind, microMacroConceptPresetIndices[kind][step]);
}
