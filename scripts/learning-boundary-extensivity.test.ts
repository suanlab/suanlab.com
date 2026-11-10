import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { finiteBoundaryExtensivityModel as model, isBoundaryExtensivityParameters as valid, type BoundaryExtensivityParameters as Params } from '../src/lib/learning-boundary-extensivity';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const base: Params = { dimension: 2, lengthOne: 2, lengthTwo: 2, width: 3, growthMode: 'uniform-dilation', scale: 1, spinOne: 1, spinTwo: -1, couplingNumerator: 1, couplingDenominator: 1 };
const fixture = JSON.parse(readFileSync('scripts/fixtures/boundary-extensivity-oracle.json', 'utf8'));
const fraction = (q: Q) => { assert.ok(q.bounds); assert.deepEqual(q.bounds.lower, q.bounds.upper); assert.equal(q.bounds.lower.numerator.exponent, 0); assert.equal(q.bounds.lower.denominator.exponent, 0); return { n: BigInt(q.bounds.lower.numerator.coefficient), d: BigInt(q.bounds.lower.denominator.coefficient) }; };
function equal(q: Q, n: bigint | number, d: bigint | number = 1) { const a = fraction(q); assert.equal(a.n * BigInt(d), BigInt(n) * a.d); }
function at(value: unknown, path: string): Q { return path.split('.').reduce((v: any, k) => v[k], value) as Q; }
for (const c of fixture.cases) test(`independent Fraction / edge reference: ${c.id}`, () => {
 const result = model(c.parameters);
 for (const [name, value] of Object.entries(c.counts)) assert.equal((result.current.geometry as any)[name], value, name);
 for (const [path, expected] of Object.entries(c.observations) as [string, { numerator: string; denominator: string }][]) {
  const q = at(result, path); equal(q, BigInt(expected.numerator), BigInt(expected.denominator)); const n = BigInt(expected.numerator);
  assert.equal(q.resolution, n === BigInt(0) ? 'structural-zero' : 'exact-rational'); assert.equal(q.sign, n === BigInt(0) ? 0 : n > BigInt(0) ? 1 : -1);
  const projected = Number(expected.numerator) / Number(expected.denominator); assert.ok(q.value !== null); assert.ok(Math.abs(q.value - projected) <= Math.abs(projected) * 2e-15 || q.value === projected);
 }
 assert.equal(result.history.at(-1), result.current); assert.equal(result.current.scale, c.parameters.scale); assert.deepEqual(structuredClone(result), result); assert.doesNotThrow(() => JSON.stringify(result));
});
test('hand calculation distinguishes restricted cuts from separately renormalized blocks', () => {
 const r = model(base).current;
 assert.equal(r.geometry.siteCount, '12'); equal(r.modes['nearest-neighbor'].combinedEnergy, -11); equal(r.modes['nearest-neighbor'].crossEnergy, 3);
 equal(r.modes['complete-unscaled'].combinedEnergy, 6); equal(r.modes['complete-kac'].combinedEnergy, 1, 2); equal(r.modes['complete-kac'].crossEnergy, 3);
 equal(r.separatelyNormalizedKac.energySum, -5); equal(r.separatelyNormalizedKac.normalizationCounterterm, 5, 2); equal(r.separatelyNormalizedKac.combinedMinusSeparate, 11, 2);
 const aligned = model({ ...base, spinTwo: 1, scale: 1000 }).current; equal(aligned.separatelyNormalizedKac.combinedMinusSeparate, -1, 2); equal(aligned.modes['complete-kac'].crossEnergyPerSite, -1, 4);
});
test('uniform dilation and fixed thickness have different boundary limits despite growing site counts', () => {
 for (const growthMode of ['uniform-dilation', 'fixed-thickness'] as const) {
  const r = model({ ...base, growthMode, scale: 1000000 }); assert.equal(r.classification.vanHoveGeometryAlongPath, growthMode === 'uniform-dilation');
  assert.equal(r.limits.externalFacesPerSite.kind, 'finite'); if (r.limits.externalFacesPerSite.kind !== 'finite') throw Error(); equal(r.limits.externalFacesPerSite.value, growthMode === 'uniform-dilation' ? 0 : 1, growthMode === 'uniform-dilation' ? 1 : 2);
  const local = r.limits['nearest-neighbor']; if (local.energyPerSite.kind !== 'finite' || local.crossMagnitudePerSite.kind !== 'finite') throw Error(); equal(local.energyPerSite.value, growthMode === 'uniform-dilation' ? -2 : -5, growthMode === 'uniform-dilation' ? 1 : 4); equal(local.crossMagnitudePerSite.value, growthMode === 'uniform-dilation' ? 0 : 1, growthMode === 'uniform-dilation' ? 1 : 4);
  assert.equal(r.classification.localCutNegligiblePerSiteAlongPath, growthMode === 'uniform-dilation'); assert.equal(r.classification.kacCutNegligiblePerSiteAlongPath, false);
 }
});
test('balanced-state cancellation never changes the unscaled Hamiltonian norm growth', () => {
 const balanced = model({ ...base, scale: 1000000 }); equal(balanced.current.modes['complete-unscaled'].energyPerSite, 1, 2); assert.equal(balanced.classification.maximumEnergyGrowth['complete-unscaled'], 'quadratic-in-site-count'); assert.equal(balanced.limits['complete-unscaled'].energyPerSite.kind, 'finite'); assert.equal(balanced.limits['complete-unscaled'].maximumAbsoluteEnergyPerSite.kind, 'positive-infinity');
 const biased = model({ ...base, lengthTwo: 3 }); assert.equal(biased.limits['complete-unscaled'].energyPerSite.kind, 'negative-infinity'); assert.equal(biased.limits['complete-unscaled'].crossEnergyPerSite.kind, 'positive-infinity');
 assert.equal(model({ ...base, spinTwo: 1 }).limits['complete-unscaled'].crossEnergyPerSite.kind, 'negative-infinity');
 const zero = model({ ...base, couplingNumerator: 0, growthMode: 'fixed-thickness' }); assert.equal(zero.classification.vanHoveGeometryAlongPath, false); assert.equal(zero.classification.localCutNegligiblePerSiteAlongPath, true); for (const mode of Object.values(zero.limits).filter(v => typeof v === 'object' && 'energyPerSite' in v)) { assert.equal(mode.energyPerSite.kind, 'finite'); if (mode.energyPerSite.kind === 'finite') equal(mode.energyPerSite.value, 0); }
});
test('all dilation records preserve exact cuts, Kac densities and finite-size corrections', () => {
 for (const c of fixture.cases) {
  const r = model(c.parameters), p = r.parameters, J = BigInt(p.couplingNumerator), D = BigInt(p.couplingDenominator), a = BigInt(p.lengthOne), b = BigInt(p.lengthTwo), L = a + b, sign = BigInt(p.spinOne * p.spinTwo), d = BigInt(p.dimension);
  for (const row of r.history) {
   const N = BigInt(row.geometry.siteCount), scale = BigInt(row.scale), w = BigInt(row.geometry.width);
   for (const m of Object.values(row.modes)) { equal(m.reconstructionResidual, 0); const cut = fraction(m.cutDifference), cross = fraction(m.crossEnergy); assert.equal(cut.n * cross.d, cross.n * cut.d); }
   equal(row.separatelyNormalizedKac.reconstructionResidual, 0); equal(row.modes['complete-kac'].crossEnergyPerSite, -J * sign * a * b, D * L * L);
   const kac = fraction(row.modes['complete-kac'].energyPerSite), limit = r.limits['complete-kac'].energyPerSite; if (limit.kind !== 'finite') throw Error(); const l = fraction(limit.value); assert.equal((kac.n * l.d - l.n * kac.d) * BigInt(2) * D * N, J * kac.d * l.d);
   equal(row.modes['nearest-neighbor'].maximumAbsoluteEnergyPerSite, J * (d * BigInt(row.geometry.lengthOne) * w + d * BigInt(row.geometry.lengthTwo) * w - w - (d - BigInt(1)) * BigInt(row.geometry.lengthOne) - (d - BigInt(1)) * BigInt(row.geometry.lengthTwo)), D * BigInt(row.geometry.lengthOne) * w + D * BigInt(row.geometry.lengthTwo) * w);
   equal(row.geometry.interfaceFacesPerSite, 1, L * (p.growthMode === 'uniform-dilation' ? scale : BigInt(1)));
  }
 }
});
test('every finite path row has the derived exact correction to its analytic local limit', () => {
 for(const c of fixture.cases){const r=model(c.parameters),p=r.parameters,J=BigInt(p.couplingNumerator),D=BigInt(p.couplingDenominator),L=BigInt(p.lengthOne+p.lengthTwo),w=BigInt(p.width),d=BigInt(p.dimension),sign=BigInt(p.spinOne*p.spinTwo),limit=r.limits['nearest-neighbor'].energyPerSite;if(limit.kind!=='finite')throw Error();const l=fraction(limit.value);
  for(const row of r.history){const v=fraction(row.modes['nearest-neighbor'].energyPerSite),scale=BigInt(row.scale);const expectedN=p.growthMode==='uniform-dilation'?J*((BigInt(2)-sign)*w+(d-BigInt(1))*L):J*(d-BigInt(1))*L;const expectedD=D*L*w*scale;assert.equal((v.n*l.d-l.n*v.d)*expectedD,expectedN*v.d*l.d);}
 }
});
test('exact geometry strings survive counts beyond binary64 integer precision', () => {
 const r = model({ ...base, dimension: 3, lengthOne: 1000000, lengthTwo: 1000000, width: 1000000, scale: 1000000 }); assert.equal(r.current.geometry.siteCount, '2000000000000000000000000000000000000'); assert.ok(BigInt(r.current.geometry.siteCount) > BigInt(Number.MAX_SAFE_INTEGER)); assert.equal(r.current.geometry.externalFaceCount, '10000000000000000000000000');
 equal(r.current.geometry.interfaceFacesPerSite, 1, 2000000000000); assert.equal(r.history.length, 21); assert.equal(r.history.at(-1)!.scale, 1000000);
});
test('swapping blocks, global spin reversal and coupling representation preserve the proper quantities', () => {
 const p = { ...base, lengthOne: 1, lengthTwo: 3, couplingNumerator: 7, couplingDenominator: 11, scale: 7 }, r = model(p), swapped = model({ ...p, lengthOne: 3, lengthTwo: 1, spinOne: -1, spinTwo: 1 }), flipped = model({ ...p, spinOne: -1, spinTwo: 1 }), scaled = model({ ...p, couplingNumerator: 14, couplingDenominator: 22 });
 for (const name of ['nearest-neighbor', 'complete-unscaled', 'complete-kac'] as const) { assert.deepEqual(r.current.modes[name].combinedEnergy, swapped.current.modes[name].combinedEnergy); assert.deepEqual(r.current.modes[name], flipped.current.modes[name]); assert.deepEqual(r.current.modes[name], scaled.current.modes[name]); assert.deepEqual(r.current.modes[name].restrictedEnergyOne, swapped.current.modes[name].restrictedEnergyTwo); }
});
test('literal enumeration of every spin microstate attains the stated worst energy bounds', () => {
 for (const dimension of [1, 2, 3] as const) {
  const width = dimension === 1 ? 1 : 2, points: number[][] = [];
  for (let x = 0; x < 2; x++) for (let y = 0; y < (dimension >= 2 ? width : 1); y++) for (let z = 0; z < (dimension === 3 ? width : 1); z++) points.push([x, y, z]);
  let nearMax = 0, completeMax = 0;
  for (let mask = 0; mask < 2 ** points.length; mask++) { const spin = (i: number) => mask & (1 << i) ? 1 : -1; let near = 0, complete = 0; for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) { const energy = -spin(i) * spin(j); complete += energy; if (points[i].reduce((s, x, k) => s + Math.abs(x - points[j][k]), 0) === 1) near += energy; } nearMax = Math.max(nearMax, Math.abs(near)); completeMax = Math.max(completeMax, Math.abs(complete)); }
  const r = model({ ...base, dimension, width, lengthOne: 1, lengthTwo: 1 }).current; equal(r.modes['nearest-neighbor'].maximumAbsoluteEnergy, nearMax); equal(r.modes['complete-unscaled'].maximumAbsoluteEnergy, completeMax); equal(r.modes['complete-kac'].maximumAbsoluteEnergy, completeMax, points.length);
 }
});
test('invalid geometry, growth paths and rational parameters are rejected without mutation', () => {
 const bad = [null, {}, { ...base, extra: 0 }, { ...base, dimension: 4 }, { ...base, dimension: 1 }, { ...base, dimension: 1, width: 1, growthMode: 'fixed-thickness' }, { ...base, scale: 0 }, { ...base, scale: 1000001 }, { ...base, lengthOne: .5 }, { ...base, width: Infinity }, { ...base, spinOne: 0 }, { ...base, couplingNumerator: -1 }, { ...base, couplingDenominator: 0 }];
 for (const input of bad) { assert.equal(valid(input), false); assert.throws(() => model(input as Params), RangeError); }
 const p = { ...base }, r = model(p); p.lengthOne = 9; assert.equal(r.parameters.lengthOne, 2); assert.ok(Object.isFrozen(r.parameters)); assert.deepEqual(model({ ...base, scale: 7 }).history.map(s => s.scale), [1, 2, 4, 7]);
});
