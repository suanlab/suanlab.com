export const scalarGibbsUnitId = 'finite-quartic-scalar-gibbs-fields' as const;

export interface ScalarGibbsInput {
  quartic: number;
  quadratic: number;
  sources: number[];
  coupling: number;
  radius: number;
  panels: number;
}

const coupled = { quartic: .25, quadratic: -.5, sources: [0, 0], coupling: 1, radius: 4, panels: 32 };

export const scalarGibbsPresets: ScalarGibbsInput[] = [
  coupled,
  { ...coupled, coupling: 0 },
  { ...coupled, sources: [.125, -.375] },
  { ...coupled, quartic: 1, quadratic: 0, sources: [0], coupling: 0 },
];

export function readScalarGibbsInput(value: unknown): ScalarGibbsInput | null {
  if (!value || typeof value !== 'object') return null;
  const input = value as ScalarGibbsInput;
  const gridValue = (x: number, lower: number, upper: number) => typeof x === 'number' && Number.isInteger(8 * x) && x >= lower && x <= upper;
  if (!gridValue(input.quartic, 1/8, 2) || !gridValue(input.quadratic, -2, 2) || !gridValue(input.coupling, 0, 2)) return null;
  if (!Array.isArray(input.sources) || ![1, 2].includes(input.sources.length)) return null;
  const sources = Array.from(input.sources);
  if (!sources.every(x => gridValue(x, -1, 1)) || sources.length === 1 && input.coupling !== 0) return null;
  if (!Number.isInteger(input.radius) || input.radius < 2 || input.radius > 8 || ![16,32,64,128].includes(input.panels)) return null;
  return { quartic: input.quartic, quadratic: input.quadratic, coupling: input.coupling, sources, radius: input.radius, panels: input.panels };
}
