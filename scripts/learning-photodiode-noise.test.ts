import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculatePhotodiodeNoise } from '../src/lib/learning-photodiode-noise';

test('integrated spectral densities reproduce the closed-form voltage variance', () => {
  const resistance = 100e3, capacitance = 50e-12, current = 1e-6, temperature = 300;
  const result = calculatePhotodiodeNoise(resistance, capacitance, current, temperature);
  const steps = 20000, limit = 100;
  const dx = limit / steps;
  let integral = Math.PI / 2 - Math.atan(limit);
  for (let i = 0; i < steps; i++) integral += dx / (1 + ((i + .5) * dx) ** 2);
  const density = 2 * 1.602176634e-19 * current + 4 * 1.380649e-23 * temperature / resistance;
  const integrated = resistance ** 2 * density * result.cutoff * integral;
  assert(Math.abs(integrated / result.rmsVoltage ** 2 - 1) < 1e-8);
  assert(Math.abs(result.equivalentBandwidth / result.cutoff - Math.PI / 2) < 1e-12);
});

test('doubling resistance halves equivalent bandwidth and doubles only shot variance', () => {
  const baseline = calculatePhotodiodeNoise(100e3, 50e-12, 1e-6, 300);
  const doubled = calculatePhotodiodeNoise(200e3, 50e-12, 1e-6, 300);
  assert.equal(doubled.shotVariance, 2 * baseline.shotVariance);
  assert.equal(doubled.thermalVariance, baseline.thermalVariance);
  assert.equal(doubled.equivalentBandwidth, baseline.equivalentBandwidth / 2);
  assert.equal(calculatePhotodiodeNoise(100e3, 50e-12, 0, 300).shotVariance, 0);
});
