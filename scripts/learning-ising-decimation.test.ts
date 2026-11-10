import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calculateFourSpinMarginal, calculateIsingDecimation } from '../src/lib/learning-ising-decimation';

const close = (actual: number, expected: number) => assert(Math.abs(actual - expected) < 1e-10, `${actual} vs ${expected}`);

test('decimation preserves the enumerated partition and physical correlation length', () => {
  for (const coupling of [0, .01, .4, 2, 4]) {
    let partition = 0;
    for (let bits = 0; bits < 256; bits++) {
      const spin = (i: number) => bits & (1 << (i % 8)) ? 1 : -1;
      const bonds = Array.from({ length: 8 }, (_, i) => spin(i) * spin(i + 1)).reduce((a, b) => a + b, 0);
      partition += Math.exp(coupling * bonds);
    }
    const steps = calculateIsingDecimation(coupling, 8);
    for (const step of steps) {
      close(step.reconstructedLogPartition, Math.log(partition));
      close(step.physicalCorrelationLength, steps[0].physicalCorrelationLength);
    }
  }
});

test('four-spin coefficient reconstructs every traced weight and its omission changes probabilities', () => {
  for (const coupling of [0, .4, 1, 4]) {
    const result = calculateFourSpinMarginal(coupling);
    for (const row of result.distribution) {
      const magnetization = row.spins.reduce((a, b) => a + b, 0);
      const spinProduct = row.spins.reduce((a, b) => a * b, 1);
      const reconstructed = result.constant + result.pair * (magnetization ** 2 - 4) / 2 + result.fourSpin * spinProduct;
      close(reconstructed, Math.log(2 * Math.cosh(coupling * magnetization)));
    }
    assert(coupling === 0 ? result.totalVariation === 0 : result.totalVariation > 0);
  }
});

test('the first ring step reproduces every unnormalized retained weight', () => {
  const coupling = .4;
  const reduced = calculateIsingDecimation(coupling, 8)[1];
  for (let kept = 0; kept < 16; kept++) {
    const spin = (bits: number, i: number) => bits & (1 << (i % 4)) ? 1 : -1;
    const weight = (eliminated: number) => {
      let bonds = 0;
      for (let i = 0; i < 4; i++) {
        bonds += spin(eliminated, i) * (spin(kept, i) + spin(kept, i + 1));
      }
      return Math.exp(coupling * bonds);
    };
    const exact = Array.from({ length: 16 }, (_, hidden) => weight(hidden)).reduce((a, b) => a + b, 0);
    const coarseBonds = Array.from({ length: 4 }, (_, i) => spin(kept, i) * spin(kept, i + 1)).reduce((a, b) => a + b, 0);
    close(Math.log(exact), reduced.normalization + reduced.coupling * coarseBonds);
  }
});
