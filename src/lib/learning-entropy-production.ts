/** Reader-side preparations and editing; numerical work stays in Workers. */
import { entropyKind, entropyInputKey, isEntropyInput, snapshotEntropyInput, type EntropyKind, type EntropyInput } from './learning-entropy-production-protocol';
import { entropyParameterKeys } from './learning-entropy-production-input';
import { parseNumberDraft } from './learning-number-input';
export { entropyInputKey }; export type { EntropyKind, EntropyInput };
export const entropyKinds = [entropyKind] as const;
export const isEntropyKind = (value: string): value is EntropyKind => value === entropyKind;
export type EntropyText = { ko: string; en: string };
export const entropyText = (ko: string, en: string): EntropyText => ({ ko, en });
const t = entropyText, kind = entropyKind;
export const entropyTitles = { [kind]: t('엔트로피 생성과 경로의 시간 방향', 'Entropy production and the time direction of paths') };
const base: EntropyInput['parameters'] = { a: 1, b: 1, c: 1, d: 1, e: 1, f: 1, rateDenominator: 1, timeNumerator: 4, timeDenominator: 1, sampleSteps: 16, initialState: 0, pathCode: 0, jumpCount: 3, tiltNumerator: 1, tiltDenominator: 4 };
const driven = { a: 2, b: 3, c: 4, d: 1, e: 1, f: 1 }, homogeneous = { a: 3, b: 3, c: 3 };
const specs: [EntropyText, Partial<EntropyInput['parameters']>][] = [
  [t('균일 평형·통행은 계속', 'Uniform equilibrium; traffic continues'), {}],
  [t('비균일 평형·열린 경로', 'Nonuniform equilibrium; open path'), { a: 2, b: 3, c: 1, d: 1, e: 2, f: 3, jumpCount: 1 }],
  [t('구동·시계 방향 한 바퀴', 'Driven; clockwise loop'), driven],
  [t('구동·반시계 방향 한 바퀴', 'Driven; reverse loop'), { ...driven, pathCode: 7 }],
  [t('구동·열린 한 점프', 'Driven; one open jump'), { ...driven, jumpCount: 1 }],
  [t('구동·왕복 경로', 'Driven; out-and-back path'), { ...driven, jumpCount: 2, pathCode: 2 }],
  [t('균일 구동·Poisson 참조', 'Homogeneous drive; Poisson reference'), homogeneous],
  [t('균일 구동·음수 경로', 'Homogeneous drive; negative path'), { ...homogeneous, pathCode: 7 }],
  [t('반대 순환·양의 평균', 'Opposite circulation; positive mean'), { a: 1, b: 1, c: 1, d: 2, e: 3, f: 4 }],
  [t('거의 평형·양수 생성률', 'Near equilibrium; positive production'), { a: 1000001, b: 1000000, c: 1000000, d: 1000000, e: 1000000, f: 1000000, rateDenominator: 1000000 }],
  [t('시간0·점프 없음', 'Zero time; no jumps'), { ...driven, timeNumerator: 0, jumpCount: 0 }],
  [t('무점프 예시·평균은 양수', 'No-jump example; positive mean'), { ...driven, jumpCount: 0 }],
  [t('z=0·정규화', 'z=0; normalization'), { ...driven, tiltNumerator: 0 }],
  [t('z=1·역과정 정규화', 'z=1; reversed-law normalization'), { ...driven, tiltNumerator: 1, tiltDenominator: 1 }],
  [t('거울 tilt z=3/4', 'Mirror tilt z=3/4'), { ...driven, tiltNumerator: 3 }],
  [t('중간 tilt z=1/2', 'Midpoint tilt z=1/2'), { ...driven, tiltDenominator: 2 }],
  [t('음수 tilt z=−1', 'Negative tilt z=−1'), { ...driven, tiltNumerator: -1, tiltDenominator: 1 }],
  [t('구간 밖 tilt z=2', 'Outside tilt z=2'), { ...driven, tiltNumerator: 2, tiltDenominator: 1 }],
  [t('아주 큰 양의 평균·로그', 'Enormous positive moment; log'), { a: 1000000000, b: 1000000000, c: 1000000000, tiltNumerator: -1, tiltDenominator: 1 }],
  [t('아주 작은 양의 평균·로그', 'Tiny positive moment; log'), { a: 1000000000, b: 1000000000, c: 1000000000, tiltDenominator: 2 }],
  [t('Subnormal 양수 표시', 'Subnormal positive display'), { a: 214, b: 214, c: 214, tiltDenominator: 2 }],
  [t('반올림된 평형과 구별', 'Distinguish rounded equilibrium'), { a: 999999998, b: 999999998, c: 999999998, d: 999999997, e: 999999998, f: 999999999, rateDenominator: 1000000000, tiltDenominator: 2 }],
  [t('촘촘한 기록·같은 끝점', 'Dense records; same endpoint'), { ...driven, sampleSteps: 64 }],
  [t('분수 전이율·평형', 'Fractional rates at equilibrium'), { a: 2, b: 3, c: 1, d: 1, e: 2, f: 3, rateDenominator: 3, jumpCount: 1, tiltNumerator: 17, tiltDenominator: 31 }],
];
export const entropyPresets = { [kind]: specs.map(([label, values], i) => ({ id: `entropy-${i}`, label, input: snapshotEntropyInput({ kind, parameters: { ...base, ...values } }) })) };
export function entropyPresetInput(kind: EntropyKind, index: number): EntropyInput {
  const value = entropyPresets[kind][index]; if (!Number.isInteger(index) || !value) throw new RangeError('Unknown entropy preparation'); return snapshotEntropyInput(value.input);
}
const labels: Record<typeof entropyParameterKeys[number], EntropyText> = {
  a: t('a: 0→1 전이율 분자', 'a: 0→1 rate numerator'), b: t('b: 1→2 전이율 분자', 'b: 1→2 rate numerator'), c: t('c: 2→0 전이율 분자', 'c: 2→0 rate numerator'), d: t('d: 1→0 전이율 분자', 'd: 1→0 rate numerator'), e: t('e: 2→1 전이율 분자', 'e: 2→1 rate numerator'), f: t('f: 0→2 전이율 분자', 'f: 0→2 rate numerator'), rateDenominator: t('전이율 공통 분모 [s⁻¹]', 'Common rate denominator [s⁻¹]'), timeNumerator: t('최종 시간 분자 [s]', 'Final time numerator [s]'), timeDenominator: t('시간 분모', 'Time denominator'), sampleSteps: t('시간 기록 간격 수', 'Number of record intervals'), initialState: t('예시 경로의 시작 상태', 'Example path initial state'), pathCode: t('방향 비트열 정수', 'Direction word integer'), jumpCount: t('예시 경로의 점프 수', 'Example path jump count'), tiltNumerator: t('Tilt z 분자', 'Tilt z numerator'), tiltDenominator: t('Tilt z 분모', 'Tilt z denominator'),
};
export function entropyFields(input: EntropyInput) { return entropyParameterKeys.map(id => ({ id, label: labels[id], value: input.parameters[id], minimum: id === 'tiltNumerator' ? -1000000000 : ['timeNumerator', 'initialState', 'jumpCount', 'pathCode'].includes(id) ? 0 : 1, maximum: id === 'sampleSteps' ? 64 : id === 'initialState' ? 2 : id === 'jumpCount' ? 16 : 1000000000 })); }
export function editEntropyDraft(input: EntropyInput, id: string, raw: string): EntropyInput | null {
  const field = entropyFields(input).find(f => f.id === id); if (!field) return null;
  const value = parseNumberDraft(raw, field.minimum, field.maximum, true); if (value === null) return null;
  const next = { kind: input.kind, parameters: { ...input.parameters, [id]: value } }; return isEntropyInput(next) ? snapshotEntropyInput(next) : null;
}
