import test from 'node:test';
import assert from 'node:assert/strict';
import { calculateGaussianField as gaussian } from '../src/lib/learning-gaussian-field';
import { gaussianFieldDefault } from '../src/lib/learning-gaussian-field-input';
import { calculateNongaussianField as spin } from '../src/lib/learning-nongaussian-field';
import { nongaussianFieldDefault, nongaussianFieldExamples } from '../src/lib/learning-nongaussian-field-input';
import { calculateSpatialSpectrum as spatial } from '../src/lib/learning-spatial-spectrum';
import { spatialSpectrumDefault, spatialSpectrumExamples } from '../src/lib/learning-spatial-spectrum-input';
import { Rational as Q } from '../src/lib/learning-graph-gauge-validation-math';
const read = Q.read, integer = (n: number) => new Q(BigInt(n));

test('24 shared finite models agree after explicitly converting physical spacing to energy coefficients', () => {
  for (const sites of [4, 7]) for (const boundary of ['open', 'periodic', 'dirichlet'] as const) for (const constraint of ['none', 'mean-zero'] as const) for (const spacing of ['1', '1/2']) {
    const r = spatial({ ...spatialSpectrumDefault, sites, boundary, constraint, spacing, massSquared: '2', temperature: '3/2', origin: 1 });
    const g = gaussian({ ...gaussianFieldDefault, sites, boundary, constraint, massSquared: r.onsiteCoefficient, stiffness: r.edgeCoefficient, temperature: '3/2', source: Array(sites).fill('0'), probes: [0, 1, sites - 2, sites - 1] });
    assert.deepEqual(g.precision, r.precision);
    assert.deepEqual(g.connectedTwo, r.covariance);
    assert.deepEqual(g.projector, r.projector);
    const densitySource = gaussian({ ...g.input, source: Array.from({ length: sites }, (_, i) => i === 0 ? spacing : '0') });
    for (let i = 0; i < sites; i++) assert.equal(densitySource.mean[i], read(r.covariance[i][0]).mul(read(spacing)).div(read(r.input.temperature)).text());
  }
});
test('a Gaussian and two positive spin distributions can share means and covariance with different fourth structure', () => {
  const g = gaussian({ ...gaussianFieldDefault, stiffness: '0', source: ['0', '0', '0', '0'] });
  const independent = spin({ ...nongaussianFieldDefault, pairFactor: '1' }), coupled = spin({ ...nongaussianFieldDefault, pairFactor: '1', fourBodyFactor: '2' });
  assert.deepEqual(g.mean, independent.mean); assert.deepEqual(g.mean, coupled.mean);
  assert.deepEqual(g.connectedTwo, independent.connectedTwo); assert.deepEqual(g.connectedTwo, coupled.connectedTwo);
  assert.equal(g.fourPoint.connectedCumulant, '0'); assert.equal(independent.fourPoint.connectedCumulant, '0'); assert.equal(coupled.fourPoint.connectedCumulant, '3/5');
  const gr = gaussian({ ...gaussianFieldDefault, stiffness: '0', source: ['0', '0', '0', '0'], probes: [0, 0, 0, 0] });
  const sr = spin({ ...nongaussianFieldDefault, pairFactor: '1', probes: [0, 0, 0, 0] });
  assert.equal(gr.fourPoint.centered, '3'); assert.equal(sr.fourPoint.centered, '1'); assert.equal(sr.fourPoint.connectedCumulant, '-2');
});
test('conditioning and sample centering remain distinct even for a normalizable massless pinned field', () => {
  const input = { ...gaussianFieldDefault, sites: 3, boundary: 'dirichlet' as const, massSquared: '0', source: ['1', '0', '0'], probes: [0, 1, 1, 2] as [number, number, number, number] };
  const full = gaussian(input), conditioned = gaussian({ ...input, constraint: 'mean-zero' });
  const weights = ['2/3', '-1/3', '-1/3'].map(read);
  let centered = integer(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) centered = centered.add(weights[i].mul(read(full.connectedTwo[i][j])).mul(weights[j]));
  assert.equal(centered.text(), '11/36'); assert.equal(conditioned.connectedTwo[0][0], '3/10'); assert.equal(centered.sub(read(conditioned.connectedTwo[0][0])).text(), '1/180');
  assert(conditioned.certificates.projectedMeanResidual.every(v => v === '0'));
  assert(conditioned.certificates.ambientMeanResidual.some(v => v !== '0'));
});
test('Gaussian heating and heating at fixed spin factors represent different paths', () => {
  const cold = gaussian(gaussianFieldDefault), hot = gaussian({ ...gaussianFieldDefault, temperature: '2' });
  assert.deepEqual(cold.mean, hot.mean); assert.deepEqual(cold.susceptibility, hot.susceptibility);
  assert.equal(read(hot.connectedTwo[0][0]).text(), read(cold.connectedTwo[0][0]).mul(integer(2)).text());
  const a = spin(nongaussianFieldDefault), b = spin({ ...nongaussianFieldDefault, temperature: '2' });
  assert.deepEqual(a.states, b.states); assert.deepEqual(a.connectedTwo, b.connectedTwo);
  assert.equal(read(b.susceptibility[0][0]).text(), read(a.susceptibility[0][0]).div(integer(2)).text());
  const fixed = nongaussianFieldExamples.find(c => c.id === 'fixed-energy-reference')!.input, heated = nongaussianFieldExamples.find(c => c.id === 'hotter-fixed-energy')!.input;
  for (const [x, y] of [[fixed.pairFactor, heated.pairFactor], [fixed.fourBodyFactor, heated.fourBodyFactor], ...fixed.sourceFactors.map((x, i) => [x, heated.sourceFactors[i]])]) assert.equal(read(x).text(), read(y).mul(read(y)).text());
  assert.notDeepEqual(spin(fixed).states.map(s => s.probability), spin(heated).states.map(s => s.probability));
});
test('raw-pair subtraction is wrong in both families and physical-source derivatives retain their temperature powers', () => {
  const g = gaussian(gaussianFieldDefault); assert.equal(g.fourPoint.raw, '2732/16875'); assert.equal(g.fourPoint.connectedCumulant, '0'); assert.equal(g.fourPoint.incorrectRawPairSubtraction, '-28/5625');
  const s = spin({ ...nongaussianFieldDefault, pairFactor: '1', sourceFactors: ['2', '1', '1', '1'], probes: [0, 0, 0, 0], temperature: '3/2' });
  assert.equal(s.fourPoint.centered, '832/625'); assert.equal(s.fourPoint.connectedCumulant, '64/625'); assert.equal(s.fourPoint.incorrectRawPairSubtraction, '-2');
  assert.equal(s.sourceDerivatives.meanSecondSelected, '-128/375'); assert.equal(s.sourceDerivatives.meanThirdSelected, '512/16875');
});
test('a local ratio and a finite constrained covariance cannot manufacture an infinite stationary field', () => {
  const m = spatial(spatialSpectrumExamples.find(c => c.id === 'massless-ring-constrained')!.input);
  assert.deepEqual(m.curve.slice(0, 3).map(p => p.covariance), ['21/32', '7/32', '-3/32']);
  assert.equal(m.localRatios[0].status, 'positive-local-ratio-only'); assert.equal(m.localRatios[1].status, 'nonpositive');
  assert.equal(m.infinite.status, 'massless-no-stationary-field'); assert.equal(m.infinite.correlationLength, null);
  const independent = spatial(spatialSpectrumExamples.find(c => c.id === 'independent-constrained')!.input);
  assert.equal(independent.covariance[0][1], '-1/8'); assert.equal(independent.infinite.correlationLength!.lower, '0');
});
test('fixed-length negative-wave refinement changes array indices while preserving physical length and wavenumber', () => {
  const r = spatial(spatialSpectrumExamples.find(c => c.id === 'negative-alias')!.input), levels = r.refinements!;
  assert.deepEqual(levels.map(p => p.arrayIndex), [7, 15, 31, 63]); assert.deepEqual(levels.map(p => p.sites), [8, 16, 32, 64]);
  assert(levels.every(p => p.fixedBoxLength === '8' && p.signedWave === -1));
  levels.forEach(p => assert.deepEqual(p.wavenumber, levels[0].wavenumber)); assert.equal(r.covariance.length, 8);
});
