/** Approximate plotting only. Verified rational tables remain authoritative. */
import type { RegularPerturbationResult } from './learning-regular-perturbation-protocol';
import { scalarFieldProjection } from './learning-scalar-field-view';
export type PerturbationWindow = 'equilibrium' | 'linear';

export function perturbationLogMagnitude(text: string): number | null {
  const [raw, denominator = '1'] = text.split('/'), numerator = raw.replace(/^-/, '');
  if (!/^\d+$/.test(numerator) || !/^[1-9]\d*$/.test(denominator) || /^0+$/.test(numerator)) return null;
  const n = numerator.replace(/^0+/, ''), a = n.slice(0, 15), b = denominator.slice(0, 15);
  return Math.log10(Number(a)) - Math.log10(Number(b)) + n.length - a.length - denominator.length + b.length;
}

export function regularPerturbationView(result: RegularPerturbationResult, window: PerturbationWindow, order: number) {
  const k = scalarFieldProjection(result.input.stiffness)!, lambda = scalarFieldProjection(result.input.coupling)!, h = scalarFieldProjection(result.input.source)!;
  const lower = scalarFieldProjection(result.root.lower)!, upper = scalarFieldProjection(result.root.upper)!;
  const root = (lower + upper) / 2, scale = Math.abs(h / k), chosen = result.approximations[Math.min(result.input.order, Math.max(0, order))];
  const partial = scalarFieldProjection(chosen.approximation);
  const width = (window === 'linear' ? scale : Math.abs(root)) * 1.5 || 1;
  const force = (x: number) => k * x + lambda * x ** 3 - h;
  const energy = (x: number, coupling = lambda) => k * x ** 2 / 2 + coupling * x ** 4 / 4 - h * x;
  const samples = Array.from({ length: 241 }, (_, i) => { const x = width * (2 * i / 240 - 1); return { x, force: force(x), energy: energy(x) }; });
  const surface = Array.from({ length: 21 }, (_, j) => Array.from({ length: 41 }, (_, i) => {
    const x = width * (2 * i / 40 - 1), coupling = lambda * j / 20;
    return { x, coupling, energy: energy(x, coupling) };
  }));
  const energyValues = surface.flat().map(p => p.energy), energyMin = Math.min(...energyValues, energy(root)), energyMax = Math.max(...energyValues);
  const orderRows = result.approximations.map(row => ({ order: row.order, term: perturbationLogMagnitude(row.term), residual: perturbationLogMagnitude(row.residual), bound: perturbationLogMagnitude(row.errorBound), distance: perturbationLogMagnitude(row.rootDistance.upper) }));
  const logs = orderRows.flatMap(row => [row.term, row.residual, row.bound, row.distance]).filter((x): x is number => x !== null);
  const logMin = logs.length ? Math.floor(Math.min(...logs)) - 1 : -1, logMax = logs.length ? Math.max(logMin + 1, Math.ceil(Math.max(...logs))) : 1;
  return { k, lambda, h, lower, upper, root, width, partial, chosen, samples, surface, energyMin, energyMax,
    forceMax: Math.max(...samples.map(p => Math.abs(p.force))), partialInside: partial !== null && Math.abs(partial) <= width,
    rootEnergy: energy(root), orderRows, logMin, logMax, force, energy };
}
export type RegularPerturbationView = ReturnType<typeof regularPerturbationView>;
