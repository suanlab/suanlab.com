import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { legendreResponseUnitId } from '../src/data/legendre-response-reader';
import { getLearningContent } from '../src/lib/learning-content';
import { calculateLegendreResponse } from '../src/lib/learning-legendre-response';

const number = (x: string) => {
  const [numerator, denominator = '1'] = x.split('/');
  return Number(numerator) / Number(denominator);
};

function check(range: { lower: string; upper: string } | null, reference: number) {
  assert(range);
  const epsilon = 8e-13 * Math.max(1, Math.abs(reference));
  assert(number(range.lower) - epsilon <= reference && reference <= number(range.upper) + epsilon,
    JSON.stringify({ range, reference }));
}

test('Gaussian dual curvature is constant and spin curvature diverges toward saturation', () => {
  for (const row of calculateLegendreResponse('gaussian')) {
    check(row.mean, row.source / 2);
    check(row.generator, row.source ** 2 / 4);
    check(row.effectiveAction, row.source ** 2 / 4);
    assert.deepEqual(row.effectiveCurvature, { lower: '2', upper: '2' });
  }
  const spin = calculateLegendreResponse('spin');
  for (const row of spin) {
    const mean = Math.tanh(row.source);
    const generator = Math.log(Math.cosh(row.source));
    check(row.mean, mean);
    check(row.generator, generator);
    check(row.effectiveAction, row.source * mean - generator);
    check(row.effectiveCurvature, Math.cosh(row.source) ** 2);
  }
  assert(number(spin[64].effectiveCurvature!.lower) > 700);
  assert(number(spin[64].effectiveAction.upper) < Math.log(2));
});

test('quartic generator and its dual contain independent Gauss–Legendre estimates', () => {
  const rows = calculateLegendreResponse('quartic');
  const run = spawnSync('python3', ['-c', `
import json
import numpy as np

nodes, weights = np.polynomial.legendre.leggauss(160)
x, w = 7*nodes, 7*weights
base = w*np.exp(-0.25*x**4+0.5*x**2)
z0 = base.sum()
rows = []
for tick in range(-8,9):
    h = tick/8
    tilted = base*np.exp(h*x)
    z = tilted.sum()
    p = tilted/z
    mean = p@x
    chi = p@(x-mean)**2
    generator = np.log(z/z0)
    rows.append([mean, generator, chi, h*mean-generator, 1/chi])
print(json.dumps(rows))
`], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const reference = JSON.parse(run.stdout) as number[][];
  assert.equal(rows.length, reference.length);
  rows.forEach((row, i) => {
    [row.mean, row.generator, row.susceptibility, row.effectiveAction, row.effectiveCurvature]
      .forEach((range, j) => check(range, reference[i][j]));
  });
  for (const range of [rows[8].mean, rows[8].generator, rows[8].effectiveAction]) {
    assert.deepEqual(range, { lower: '0', upper: '0' });
  }
  assert(number(rows[8].effectiveCurvature!.lower) > 0);
});

test('both manuscript locales render the duality equations and share the executed inversion example', async () => {
  const content = await getLearningContent(legendreResponseUnitId);
  const exercises = (['ko', 'en'] as const).map(locale => {
    assert.equal(content.outline[locale].length, 9);
    assert(content[locale].includes('<math'));
    assert(!content[locale].includes('katex-error'));
    const source = readFileSync('content/learning/physics-math-for-ai/' + legendreResponseUnitId + '.' + locale + '.md', 'utf8');
    return source.match(/\x60{3}python\n([\s\S]*?)\n\x60{3}/)?.[1];
  });
  assert(!/[가-힣]/.test(content.en));
  assert(exercises[0]);
  assert.equal(exercises[0], exercises[1]);
  const run = spawnSync('python3', ['-c', exercises[0]], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
});
