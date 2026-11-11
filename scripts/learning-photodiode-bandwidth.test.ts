import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculatePhotodiodeResponse, calculatePhotodiodeStep } from '../src/lib/learning-photodiode-bandwidth';

const close = (a: number, b: number) => assert(Math.abs(a - b) < 1e-9, `${a} vs ${b}`);

test('one RC pole gives the expected rise time, cutoff amplitude, and phase', () => {
  const cutoff = 1 / (2 * Math.PI * 5e-6);
  const result = calculatePhotodiodeResponse(100e3, 50e-12, .5, 10e-6, cutoff);
  close(result.tau, 5e-6);
  close(result.finalVoltage, .5);
  close(result.cutoff, cutoff);
  close(result.relativeAmplitude, Math.SQRT1_2);
  close(result.phaseDegrees, -45);
  const t10 = -result.tau * Math.log(.9);
  close(calculatePhotodiodeStep(result.finalVoltage, result.tau, t10), .05);
  close(calculatePhotodiodeStep(result.finalVoltage, result.tau, t10 + result.riseTime), .45);
  assert.equal(calculatePhotodiodeStep(.5, result.tau, -1), 0);
});

test('resistance trades gain for bandwidth while capacitance changes bandwidth alone', () => {
  const readout = (r: number, c: number) => calculatePhotodiodeResponse(r, c, .5, 10e-6, 0);
  const baseline = readout(100e3, 50e-12);
  const highLoad = readout(200e3, 50e-12);
  const lowCapacitance = readout(100e3, 25e-12);
  assert.equal(highLoad.dcGain, 2 * baseline.dcGain);
  assert.equal(highLoad.cutoff, baseline.cutoff / 2);
  assert.equal(lowCapacitance.dcGain, baseline.dcGain);
  assert.equal(lowCapacitance.cutoff, 2 * baseline.cutoff);
  assert.equal(lowCapacitance.relativeAmplitude, 1);
  assert.equal(lowCapacitance.phaseDegrees, 0);
});
