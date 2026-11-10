import { lrKinds, lrInputKey, isLrInput, snapshotLrInput, lrParameterKeys, type LrKind, type LrInput } from './learning-linear-response-protocol';
import { parseNumberDraft } from './learning-number-input';
export { lrKinds, lrInputKey }; export type { LrKind, LrInput };
export type LrText = { ko: string; en: string };
export const lrText = (ko: string, en: string): LrText => ({ ko, en });
const t = lrText, kind = lrKinds[0];
export const isLrKind = (value: string): value is LrKind => lrKinds.some(kind => kind === value);
export const lrTitles: Record<LrKind, LrText> = { [kind]: t('손잡이가 바꾸는 선형 응답', 'Linear response depends on the handle') };
const base: LrInput['parameters'] = { a: 1, b: 1, c: 1, d: 1, e: 1, f: 1, rateDenominator: 1, kineticNumerator: 0, kineticDenominator: 1, frequencyNumerator: 1, frequencyDenominator: 2, timeNumerator: 4, timeDenominator: 1, sampleSteps: 16, amplitudeNumerator: 1, amplitudeDenominator: 10 };
const driven = { a: 2, b: 3, c: 4, d: 1, e: 1, f: 1 };
const specs: [LrText, Partial<LrInput['parameters']>][] = [
  [t('평형 기준', 'Equilibrium reference'), {}],
  [t('같은 구동·손잡이 −1', 'Same drive; handle −1'), { ...driven, kineticNumerator: -1 }],
  [t('같은 구동·손잡이 0', 'Same drive; handle 0'), driven],
  [t('같은 구동·손잡이 +1', 'Same drive; handle +1'), { ...driven, kineticNumerator: 1 }],
  [t('평형·손잡이 −1', 'Equilibrium; handle −1'), { kineticNumerator: -1 }],
  [t('평형·손잡이 +1', 'Equilibrium; handle +1'), { kineticNumerator: 1 }],
  [t('반대 순환 배경', 'Opposite circulating background'), { a: 1, b: 1, c: 1, d: 2, e: 3, f: 4 }],
  [t('영 입력·커널은 남는다', 'Zero input; kernel remains'), { ...driven, amplitudeNumerator: 0 }],
  [t('음수 계단 입력', 'Negative step input'), { ...driven, amplitudeNumerator: -1 }],
  [t('큰 입력·접선과 차이', 'Large input and tangent error'), { ...driven, amplitudeNumerator: 1, amplitudeDenominator: 1 }],
  [t('작은 입력', 'Small input'), { ...driven, amplitudeDenominator: 100 }],
  [t('선형0·유한 양수 입력', 'Zero linear response; positive finite input'), { a: 3, b: 3, c: 3, kineticNumerator: -1, amplitudeDenominator: 2 }],
  [t('선형0·유한 음수 입력', 'Zero linear response; negative finite input'), { a: 3, b: 3, c: 3, kineticNumerator: -1, amplitudeNumerator: -1, amplitudeDenominator: 2 }],
  [t('영 시간·오른쪽 커널', 'Zero time; right-limit kernel'), { ...driven, timeNumerator: 0 }],
  [t('아주 짧은 시간·작은 입력', 'Very short time and small input'), { ...driven, timeNumerator: 1, timeDenominator: 1000000, amplitudeNumerator: 1, amplitudeDenominator: 1000000 }],
  [t('빠른 혼합·작은 커널', 'Fast mixing and tiny kernel'), { a: 1000000, b: 1000000, c: 1000000, d: 1000000, e: 1000000, f: 1000000, amplitudeDenominator: 2 }],
  [t('양의 지수 전이율 미해결', 'Unresolved positive exponential rate'), { ...driven, kineticNumerator: 1000000, amplitudeNumerator: 1, amplitudeDenominator: 1 }],
  [t('아주 작은 양의 전이율', 'Extremely small positive rates'), { kineticNumerator: -1000000, amplitudeNumerator: 1, amplitudeDenominator: 1 }],
  [t('분수 손잡이·음수 입력', 'Fractional handle and negative input'), { ...driven, kineticNumerator: 1, kineticDenominator: 3, amplitudeNumerator: -1, amplitudeDenominator: 2 }],
  [t('영 주파수·정적 극한', 'Zero frequency and static limit'), { ...driven, frequencyNumerator: 0 }],
  [t('빠른 정현파 감수성', 'Susceptibility to a fast sinusoid'), { ...driven, frequencyNumerator: 20, frequencyDenominator: 1 }],
  [t('두 배 시계·절반 시간', 'Twice the clock and half the time'), { a: 4, b: 6, c: 8, d: 2, e: 2, f: 2, timeNumerator: 2, frequencyNumerator: 1, frequencyDenominator: 1 }],
  [t('같은 전이율·다른 정수 표현', 'Same rates with another integer encoding'), { a: 4, b: 6, c: 8, d: 2, e: 2, f: 2, rateDenominator: 2 }],
  [t('촘촘한 시간 기록', 'Dense time records'), { ...driven, sampleSteps: 64 }],
];
export const lrPresets: Record<LrKind, { id: string; label: LrText; input: LrInput }[]> = { [kind]: specs.map(([label, values], i) => ({ id: `lr-${i}`, label, input: snapshotLrInput({ kind, parameters: { ...base, ...values } }) })) };
export function lrPresetInput(kind: LrKind, index: number) { const p = lrPresets[kind][index]; if (!Number.isInteger(index) || !p) throw new RangeError('Unknown linear response preset'); return snapshotLrInput(p.input); }
const labels: Record<typeof lrParameterKeys[number], LrText> = {
  a: t('a: 0→1 전이율 분자', 'a: 0→1 rate numerator'), b: t('b: 1→2 전이율 분자', 'b: 1→2 rate numerator'), c: t('c: 2→0 전이율 분자', 'c: 2→0 rate numerator'),
  d: t('d: 1→0 전이율 분자', 'd: 1→0 rate numerator'), e: t('e: 2→1 전이율 분자', 'e: 2→1 rate numerator'), f: t('f: 0→2 전이율 분자', 'f: 0→2 rate numerator'),
  rateDenominator: t('전이율 공통 분모 [s⁻¹]', 'Common rate denominator [s⁻¹]'), kineticNumerator: t('손잡이 η 분자', 'Handle η numerator'), kineticDenominator: t('손잡이 η 분모', 'Handle η denominator'),
  frequencyNumerator: t('각주파수 Ω 분자 [rad/s]', 'Angular frequency Ω numerator [rad/s]'), frequencyDenominator: t('각주파수 분모', 'Angular frequency denominator'),
  timeNumerator: t('최종 시간 분자 [s]', 'Final time numerator [s]'), timeDenominator: t('시간 분모', 'Time denominator'), sampleSteps: t('시간 기록 간격 수', 'Number of time intervals'),
  amplitudeNumerator: t('계단 입력 h 분자', 'Step input h numerator'), amplitudeDenominator: t('계단 입력 h 분모', 'Step input h denominator'),
};
export function lrFields(input: LrInput) { return lrParameterKeys.map(id => ({ id, label: labels[id], value: input.parameters[id], minimum: ['kineticNumerator', 'amplitudeNumerator'].includes(id) ? -1000000 : ['timeNumerator', 'frequencyNumerator'].includes(id) ? 0 : 1, maximum: id === 'sampleSteps' ? 64 : 1000000 })); }
export function editLrDraft(input: LrInput, id: string, raw: string): LrInput | null {
  const field = lrFields(input).find(f => f.id === id); if (!field) return null;
  const value = parseNumberDraft(raw, field.minimum, field.maximum, true); if (value === null) return null;
  const next = { kind: input.kind, parameters: { ...input.parameters, [id]: value } };
  return isLrInput(next) ? snapshotLrInput(next) : null;
}
