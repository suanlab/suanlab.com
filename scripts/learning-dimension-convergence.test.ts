import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateOscillatorConvergence } from '../src/lib/learning-dimension-convergence';

test('energy growth agrees with the independently derived per-step identity', () => {
  for (const steps of [20, 40, 80, 160]) {
    const result = calculateOscillatorConvergence(steps);
    assert(Math.abs(result.energyRatio - (1 + result.h ** 2) ** steps) < 1e-12);
    assert(Math.abs(result.energyRatio - result.analyticalEnergyRatio) < 1e-12);
  }
});

test('refinement at a fixed horizon decreases trajectory error and energy drift', () => {
  let previousError = Infinity, previousEnergy = Infinity;
  for (const steps of [20, 40, 80, 160]) {
    const result = calculateOscillatorConvergence(steps);
    assert(result.stateError < previousError);
    assert(result.energyRatio < previousEnergy && result.energyRatio > 1);
    previousError = result.stateError;
    previousEnergy = result.energyRatio;
  }
});
