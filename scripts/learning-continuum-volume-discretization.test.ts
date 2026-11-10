import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { nonuniformVolumeSimulation, sharedVolumeRingStep, type NonuniformVolumeParameters, type VolumeObservation } from '../src/lib/learning-continuum-volume-discretization';
interface Reference {
    value: string;
    sign: number;
    logMagnitude: string | null;
}
interface Case {
    parameters: NonuniformVolumeParameters;
    cells: Record<string, Reference>[];
    correct: Record<string, Reference>;
    wrong: Record<string, Reference>;
    summary: Record<string, Reference>;
    faces: Record<string, Reference>[];
    localCfl: Reference[];
    positivitySufficient: boolean;
}
const fixture = JSON.parse(readFileSync('scripts/fixtures/continuum-volume-independent.json', 'utf8'));
const cases: Case[] = [...fixture.ordinaryCases, ...fixture.limitCases];
function compare(actual: VolumeObservation, expected: Reference, label: string, scaleLog = -Infinity) {
    assert.ok(actual.sign === 0 ? actual.value === 0 && actual.logMagnitude === null : actual.logMagnitude !== null && Number.isFinite(actual.logMagnitude), label + ' representation');
    const log = expected.logMagnitude === null ? -Infinity : Number(expected.logMagnitude);
    // Cancellation is conditioned on the scale of the actual terms, never on 1.
    if (log < scaleLog - 28 || !expected.sign) {
        assert.ok(!actual.sign || actual.logMagnitude! <= scaleLog - 24, label + ': cancellation residual ' + JSON.stringify(actual));
        return;
    }
    assert.equal(actual.sign, expected.sign, label + ' sign');
    assert.ok(Math.abs(actual.logMagnitude! - log) < 2e-8, label + ': log ' + actual.logMagnitude + ' vs ' + log);
    if (actual.value !== null)
        assert.equal(Math.sign(actual.value), actual.sign, label + ' ordinary sign');
}
const referenceScale = (xs: Reference[]) => Math.max(...xs.map(x => x.logMagnitude === null ? -Infinity : Number(x.logMagnitude)));
for (const [index, c] of cases.entries())
    test(`volume independent dense matrix powers and analytic integrals ${index + 1}`, () => {
        const result = nonuniformVolumeSimulation(c.parameters);
        for (const key of Object.keys(c.cells[0])) {
            const scale = referenceScale(c.cells.map(x => x[key]));
            c.cells.forEach((expected, i) => compare(result.cells[i][key as keyof typeof result.cells[number]] as VolumeObservation, expected[key], `case ${index} cell ${i} ${key}`, scale));
        }
        for (const side of ['correct', 'wrong'] as const)
            for (const [key, value] of Object.entries(c[side])) {
                if (key === 'assembledMass')
                    continue; // Direct cancellation diagnostics are tested against their condition number below.
                compare(result[side][key as keyof typeof result.correct] as VolumeObservation, value, `case ${index} ${side} ${key}`, Number(value.logMagnitude ?? -Infinity));
            }
        for (const [key, value] of Object.entries(c.summary)) {
            const scale = key === 'wrongMassRateFromFaces' && c.parameters.meshWarp === 0 ? referenceScale(c.cells.map(x => x.wrongRate)) : -Infinity;
            compare(result[key as keyof typeof result] as VolumeObservation, value, `case ${index} ${key}`, scale);
        }
        for (const key of ['full', 'departure', 'change']) {
            const scale = referenceScale(c.faces.map(x => x[key]));
            c.faces.forEach((f, i) => compare(result.faces[i][key as 'full'], f[key], `case ${index} face ${i} ${key}`, scale));
        }
        c.localCfl.forEach((v, i) => compare(result.localCfl[i], v, `case ${index} CFL ${i}`));
        assert.equal(result.positivitySufficient, c.positivitySufficient);
        assert.equal(result.history.length, c.parameters.steps + 1);
        assert.equal(result.correct.massChangeFromFaces.sign, 0);
    });
for (const [i, c] of fixture.ringCases.entries())
    test(`arbitrary ring independent one step ${i + 1}`, () => {
        const result = sharedVolumeRingStep(c.volumes, c.conductances, c.concentration, c.velocityArea, c.timeStep);
        for (const side of ['correct', 'wrong'] as const) {
            for (const key of ['rates', 'values'] as const)
                c[side][key].forEach((v: Reference, j: number) => compare(result[side][key][j], v, `ring ${i} ${side} ${key} ${j}`));
            for (const key of ['assembledPhysicalMass', 'stableNormChange'] as const)
                compare(result[side][key], c[side][key], `ring ${i} ${side} ${key}`);
        }
    });
test('nonuniform storage counterexample has the declared rational mass and Euler norm budgets', () => {
    const s = sharedVolumeRingStep([1, 2, 1], [1, 1, 1], [1, 0, 0], 0, .1);
    const near = (a: VolumeObservation, b: number) => assert.ok(Math.abs(a.value! - b) < 2e-14);
    s.correct.rates.forEach((v, i) => near(v, [-2, .5, 1][i]));
    s.wrong.rates.forEach((v, i) => near(v, [-1.5, .75, .75][i]));
    near(s.correct.assembledPhysicalMass, 1);
    near(s.wrong.assembledPhysicalMass, 1.075);
    near(s.correct.stableNormChange, -.1725);
    near(s.normRateFromFaces, -2);
    near(s.correct.positiveEulerCorrection, .0275);
});
test('prescribed excessive timestep is executed, preserves the face mass ledger and exposes instability', () => {
    const p = { ...cases[0].parameters, velocity: 2, diffusivity: 2, meshWarp: .75, timeStep: .25, steps: 16 };
    const s = nonuniformVolumeSimulation(p);
    assert.equal(s.parameters.timeStep, .25);
    assert.equal(s.positivitySufficient, false);
    assert.equal(s.correct.massChangeFromFaces.sign, 0);
    assert.equal(s.correct.minimumConcentration.sign, -1);
    assert.ok(s.correct.stableCenteredNormChange.sign > 0);
    const residual = s.correct.assembledMassChange;
    assert.ok(!residual.sign || residual.logMagnitude! < s.correct.absoluteCellMassSum.logMagnitude! - 24);
});
test('constant states, zero flow, zero steps and zero dt remain exact fixed states', () => {
    for (const patch of [{ relativeContrast: 0 }, { meanConcentration: 0 }, { velocity: 0, diffusivity: 0 }, { timeStep: 0 }, { steps: 0 }]) {
        const s = nonuniformVolumeSimulation({ ...cases[0].parameters, ...patch });
        for (const cell of s.cells) {
            assert.equal(cell.change.sign, 0);
            assert.equal(cell.wrongChange.sign, 0);
            assert.equal(cell.exactChange.sign, 0);
            assert.equal(cell.error.sign, 0);
        }
    }
});
test('small mean, contrast, time, diffusion and flow survive as signed logs without a null-to-zero conversion', () => {
    const s = nonuniformVolumeSimulation({ ...cases[0].parameters, meanConcentration: Number.MIN_VALUE, relativeContrast: Number.MIN_VALUE, timeStep: Number.MIN_VALUE });
    assert.ok(s.cells.some(c => c.change.sign !== 0 && c.change.value === null && c.change.logMagnitude! < -2000));
    const flow = nonuniformVolumeSimulation({ ...cases[0].parameters, velocity: Number.MIN_VALUE, area: .25, diffusivity: 0 });
    assert.ok(flow.upwindNormLoss.sign > 0);
    assert.equal(flow.upwindNormLoss.value, null);
    assert.ok(flow.cells.some(c => c.change.sign !== 0));
    const diffusion = nonuniformVolumeSimulation({ ...cases[0].parameters, velocity: 0, diffusivity: Number.MIN_VALUE, area: .25 });
    assert.ok(diffusion.mesh.conductances.every(c => c.sign > 0));
    assert.ok(diffusion.diffusiveNormLoss.sign > 0);
});
test('valid stable updates preserve positivity and contract the volume weighted quadratic norm', () => {
    for (const c of cases.slice(0, 24)) {
        const s = nonuniformVolumeSimulation(c.parameters);
        if (!s.positivitySufficient)
            continue;
        assert.ok(s.correct.minimumConcentration.sign >= 0);
        assert.ok(s.correct.stableCenteredNormChange.sign <= 0);
        const residual = s.correct.assembledMassChange;
        assert.ok(!residual.sign || residual.logMagnitude! < s.correct.absoluteCellMassSum.logMagnitude! - 25);
    }
});
test('uniform storage control gives the same actual correct and wrong evolution', () => {
    const s = nonuniformVolumeSimulation({ ...cases[0].parameters, meshWarp: 0 });
    s.cells.forEach(c => assert.deepEqual(c.change, c.wrongChange));
    assert.equal(s.wrong.massChangeFromFaces.sign, 0);
});
test('invalid and sparse input arrays and parameter bounds are rejected', () => {
    for (const patch of [{ cellCountExponent: 2 }, { cellCountExponent: 3.5 }, { diffusivity: -1 }, { timeStep: -1 }, { steps: 65 }, { meshWarp: 1 }, { velocity: Infinity }, { meanConcentration: NaN }])
        assert.throws(() => nonuniformVolumeSimulation({ ...cases[0].parameters, ...patch }), RangeError);
    assert.throws(() => sharedVolumeRingStep([1, , 1] as number[], [1, 1, 1], [1, 0, 0], 0, .1), RangeError);
    assert.throws(() => sharedVolumeRingStep([1, 2, 1], [1, 1], [1, 0, 0], 0, .1), RangeError);
});
test('mesh geometry, conductance units and weighted mass are checked independently of the face ledger', () => {
    for (const c of cases.slice(0, 24)) {
        const s = nonuniformVolumeSimulation(c.parameters), totalVolume = c.parameters.area * c.parameters.length;
        assert.ok(s.mesh.volumes.every(v => v > 0));
        assert.ok(s.mesh.distances.every(v => v > 0));
        assert.ok(Math.abs(s.mesh.volumes.reduce((a, b) => a + b, 0) - totalVolume) < 1e-13 * totalVolume);
        const expectedMass = c.parameters.meanConcentration * totalVolume;
        for (const side of ['correct', 'wrong'] as const) {
            const actual = s[side].assembledMass.value, expected = Number(c[side].assembledMass.value), scale = Number(c[side].absoluteCellMassSum.value);
            assert.ok(actual !== null && Math.abs(actual - expected) <= 2e-11 * scale + Number.MIN_VALUE);
        }
        if (s.positivitySufficient && expectedMass > 0)
            assert.ok(Math.abs(s.correct.assembledMass.value! - expectedMass) < 1e-12 * expectedMass);
    }
});
test('actual next Euler norm increment matches the face losses plus positive time correction', () => {
    for (const velocity of [-.5, 0, .5])
        for (const diffusivity of [0, .1]) {
            const p = { ...cases[0].parameters, velocity, diffusivity, timeStep: .001, steps: 8 };
            const current = nonuniformVolumeSimulation(p), next = nonuniformVolumeSimulation({ ...p, steps: 9 });
            const measured = next.correct.stableCenteredNormChange.value! - current.correct.stableCenteredNormChange.value!;
            const predicted = current.predictedNextNormChangeFromFaces.value!;
            assert.ok(Math.abs(measured - predicted) < 1e-12 * Math.max(Math.abs(predicted), 1e-4));
        }
});
test('physical cell-average error decreases under stable mesh refinement at the same final time', () => {
    const errors = [3, 4, 5, 6].map(cellCountExponent => {
        const s = nonuniformVolumeSimulation({ ...cases[0].parameters, cellCountExponent, timeStep: .0001, steps: 64 });
        assert.equal(s.positivitySufficient, true);
        return s.l1CellAverageError.value!;
    });
    for (let i = 1; i < errors.length; i++)
        assert.ok(errors[i] < errors[i - 1], JSON.stringify(errors));
});

test('all parameter fields reject nonfinite and missing values, and extreme result trees stay finite or explicitly unavailable', () => {
  for (const key of Object.keys(cases[0].parameters)) {
    for (const invalid of [NaN, Infinity, -Infinity, undefined]) {
      assert.throws(() => nonuniformVolumeSimulation({ ...cases[0].parameters, [key]: invalid } as NonuniformVolumeParameters), RangeError);
    }
  }
  const inspect = (value: unknown): void => {
    if (typeof value === 'number') assert.ok(Number.isFinite(value));
    else if (value && typeof value === 'object') Object.values(value).forEach(inspect);
  };
  for (const velocity of [-2, 2]) for (const meshWarp of [-.75, .75]) {
    const result = nonuniformVolumeSimulation({ ...cases[0].parameters, cellCountExponent: 6, length: .25, area: 4, meshWarp, velocity, diffusivity: 2, meanConcentration: 4, relativeContrast: -1, mode: 3, phaseCycles: 1, timeStep: .25, steps: 64 });
    inspect(result);
    assert.doesNotThrow(() => JSON.stringify(result));
    assert.ok(result.cells.some(c => c.concentration.value === null && c.concentration.sign !== 0));
  }
});
