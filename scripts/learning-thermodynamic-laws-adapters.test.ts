import test from 'node:test';
import assert from 'node:assert/strict';
import { carnotCycleState } from '../src/lib/learning-carnot-cycle';
import { reservoirCycle } from '../src/lib/learning-reservoir-cycle';
import { gappedSpectrum } from '../src/lib/learning-gapped-spectrum';
import { evaluateCarnotLaw as carnot, evaluateReservoirLaw as reservoir, evaluateSpectrumLaw as spectrum,
  validCarnotValues, validReservoirValues, validSpectrumValues, thermodynamicLawDefaults as defaults,
  thermodynamicLawKinds, isThermodynamicLawKind, lawObservationSegments } from '../src/lib/learning-thermodynamic-laws';

const initial = thermodynamicLawKinds.map(k => defaults[k]);
test('all defaults are valid; family guard rejects inherited/unknown names', () => {
  [validCarnotValues, validReservoirValues, validSpectrumValues].forEach((validate, i) => {
    assert.equal(validate(initial[i]), true);
    for (const value of [null, {}, [], Array(initial[i].length), [...initial[i], 1], initial[i].slice(1),
      initial[i].map((v, j) => j === 0 ? NaN : v), initial[i].map((v, j) => j === 0 ? '1' : v)])
      assert.equal(validate(value), false);
  });
  thermodynamicLawKinds.forEach(k => assert.equal(isThermodynamicLawKind(k), true));
  for (const k of ['constructor', '__proto__', '', 'thermodynamics']) assert.equal(isThermodynamicLawKind(k), false);
});
test('direction and degeneracies require integers; subnormal positive temperatures remain valid', () => {
  assert.equal(validCarnotValues([1, 600, 300, .01, 2, .5, 0]), false);
  for (let i = 0; i < 3; i++) {
    const v = [...initial[2]]; v[i] = 1.5; assert.equal(validSpectrumValues(v), false);
  }
  assert.equal(validSpectrumValues([64, 16, 16, Number.MIN_VALUE, Number.MIN_VALUE]), true);
  assert.throws(() => spectrum([1.5, 1, 1, 1, 1]), RangeError);
});
test('invalid reservoir order remains editable without silently swapping temperature roles', () => {
  const c = [1, 250, 800, .01, 2, 0, .5], r = [250, 800, 1000, .25];
  assert.equal(validCarnotValues(c), true); assert.equal(validReservoirValues(r), true);
  for (const result of [carnot(c), reservoir(r)]) {
    assert.equal(result.status, 'invalid-reservoir-order'); assert.equal(result.query, null);
    assert.deepEqual(result.history, []); assert.equal(result.parameters.hotTemperature, 250);
  }
  c[1] = 800; r[0] = 800;
  assert.equal(carnot(c).status, 'ready'); assert.equal(reservoir(r).status, 'ready');
});
test('every Carnot control reaches direct current-state calculation, independently of the plotting grid', () => {
  const alternatives = [1.5, 700, 400, .02, 3, 1, Number.MIN_VALUE];
  alternatives.forEach((v, i) => {
    const values = [...initial[0]]; values[i] = v;
    const result = carnot(values); assert.equal(result.status, 'ready');
    assert.deepEqual(result.query, carnotCycleState(result.parameters, result.direction, values[6]));
    assert.ok(result.history.some(q => q.progress === values[6]));
    assert.deepEqual(result.history.at(-1), result.cycle?.complete);
  });
});
test('Carnot curves include all corners and retain tiny reverse progress rather than subtracting it from one', () => {
  const values = [1, 600, 300, .01, 2, 1, Number.MIN_VALUE], result = carnot(values);
  assert.equal(result.status, 'ready'); assert.ok(result.query);
  assert.equal(result.query.workByGas.sign, 1); assert.equal(result.query.progress, Number.MIN_VALUE);
  assert.equal(result.cycle?.complete.workByGas.sign, -1);
  for (const s of [0, .25, .5, .75, 1]) assert.ok(result.history.some(q => q.progress === s));
  for (let i = 1; i < result.history.length; i++) assert.ok(result.history[i].progress > result.history[i - 1].progress);
});
test('Carnot equal temperature and ratio-one cases preserve defined ratio versus actual performance', () => {
  const equal = carnot([1, 300, 300, .01, 2, 0, 1]);
  assert.equal(equal.cycle?.workPerHotHeat?.value, 0); assert.equal(equal.cycle?.engineEfficiency, null);
  const collapsed = carnot([1, 600, 300, .01, 1, 0, 1]);
  assert.equal(collapsed.cycle?.workPerHotHeat, null); assert.equal(collapsed.cycle?.carnotTemperatureBound.value, .5);
});
test('every reservoir control reaches its port budget without invented trajectory or rate', () => {
  [700, 400, 2000, .75].forEach((v, i) => {
    const values = [...initial[1]]; values[i] = v; const result = reservoir(values);
    assert.equal(result.status, 'ready'); assert.deepEqual(result.query, reservoirCycle(result.parameters));
    assert.ok(result.history.some(q => q.parameters.relativeEntropyGeneration === values[3]));
    assert.equal('progress' in result.history[0], false); assert.equal('pressure' in result.history[0], false);
  });
});
test('reservoir sweep includes reversibility and both admissible/counterfactual budgets', () => {
  const result = reservoir(initial[1]);
  assert.ok(result.history.some(q => q.entropyGeneration.sign === 0));
  assert.ok(result.history.some(q => q.regime === 'counterfactual'));
  assert.ok(result.history.some(q => q.regime === 'work-driven-dissipation'));
  assert.equal(reservoir([600, 300, 0, -1]).query?.regime, 'no-transfer');
  const tiny = reservoir([600, 300, Number.MIN_VALUE, -Number.MIN_VALUE]).query!;
  assert.equal(tiny.entropyGeneration.value, null); assert.equal(tiny.entropyGeneration.sign, -1);
  assert.equal(tiny.regime, 'counterfactual');
});
test('every spectrum control reaches direct selected equilibrium with exact state counts', () => {
  [64, 16, 16, 400, 200].forEach((v, i) => {
    const values = [...initial[2]]; values[i] = v; const result = spectrum(values);
    assert.deepEqual(result.query, gappedSpectrum(result.parameters));
    assert.ok(result.history.some(q => q.parameters.temperature === values[4]));
    assert.equal(result.query.totalStateCount, (BigInt(values[1] + values[2]) ** BigInt(values[0])).toString());
  });
});
for (const [gap, temperature] of [[200, 0], [0, 100], [0, 0], [Number.MIN_VALUE, Number.MIN_VALUE],
  [Number.MIN_VALUE, 800], [1000, Number.MIN_VALUE], [1, 800], [1000, 800]]) {
  test(`positive-temperature mesh keeps T=0 separate and resolves gap scale: gap=${gap}, T=${temperature}`, () => {
    const result = spectrum([1, 1, 3, gap, temperature]);
    assert.equal(result.zeroTemperatureLimit.parameters.temperature, 0);
    assert.equal(result.zeroTemperatureLimit.entropyTemperatureDerivative, null);
    assert.ok(result.history.length > 1);
    const [low, high] = result.logTemperatureDomain;
    assert.ok(Number.isFinite(low) && Number.isFinite(high) && low < high);
    for (let i = 0; i < result.history.length; i++) {
      const T = result.history[i].parameters.temperature;
      assert.ok(T > 0 && T <= 800);
      assert.ok(Math.log10(T) >= low - 1e-12 && Math.log10(T) <= high + 1e-12);
      if (i) assert.ok(T > result.history[i - 1].parameters.temperature);
    }
    if (temperature > 0) assert.ok(result.history.some(q => q.parameters.temperature === temperature));
    if (gap > 0 && gap <= 800) assert.ok(result.history.some(q => q.parameters.temperature === gap));
    // For a normal-range gap, there is a locally refined mesh even if the query
    // is at MIN_VALUE. This avoids coarse linear interpolation across the peak.
    if (gap >= 1 && gap <= 800) assert.ok(result.history.filter(q => q.parameters.temperature >= gap / 2 && q.parameters.temperature <= gap * 2).length >= 10);
  });
}
test('curve adapter breaks unavailable/undefined observations while keeping actual zero and negative signs', () => {
  const samples = [
    { value: 2, sign: 1, logMagnitude: Math.log(2) },
    { value: null, sign: 1, logMagnitude: -900 },
    { value: 0, sign: 0, logMagnitude: null },
    null,
    { value: -3, sign: -1, logMagnitude: Math.log(3) },
  ];
  const rows = samples.map((q, i) => ({ q, i }));
  assert.deepEqual(lawObservationSegments(rows, r => r.i, r => r.q), [[[0, 2]], [[2, 0]], [[4, -3]]]);
  assert.deepEqual(lawObservationSegments(rows, r => r.i, r => r.q, 'logMagnitude'),
    [[[0, Math.log(2)], [1, -900]], [[4, Math.log(3)]]]);
});
test('zero-gap upper group stays populated; small positive gap is not relabelled as zero gap', () => {
  const zero = spectrum([1, 1, 3, 0, 0]);
  assert.equal(zero.query.upperGroupProbability.value, .75);
  assert.equal(zero.query.spectrum, 'exact-zero-gap');
  const positive = spectrum([1, 1, 3, Number.MIN_VALUE, 800]);
  assert.equal(positive.query.spectrum, 'positive-gap');
  assert.equal(positive.query.heatCapacityPerCopyInKB.value, null);
  assert.equal(positive.query.heatCapacityPerCopyInKB.sign, 1);
});
test('adapters leave input vectors unchanged and produce serializable finite displays/statuses', () => {
  const before = JSON.stringify(initial);
  for (const result of [carnot(initial[0]), reservoir(initial[1]), spectrum(initial[2])]) {
    const serialized = JSON.stringify(result);
    assert.ok(serialized.length > 1000); assert.doesNotMatch(serialized, /NaN|Infinity/);
    assert.deepEqual(JSON.parse(serialized), result);
  }
  assert.equal(JSON.stringify(initial), before);
});
