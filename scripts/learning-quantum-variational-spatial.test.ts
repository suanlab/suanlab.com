import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { build } from 'esbuild';
import { quantumVariationalKinds as kinds, quantumVariationalNewSpace as space, quantumVariationalPresetInput as preset } from '../src/lib/learning-quantum-variational';
import { evaluateQuantumVariationalRequest as evaluate } from '../src/lib/learning-quantum-variational-evaluator';
import { quantumRitzGeometry as geometry } from '../src/lib/learning-quantum-variational-geometry';
import { ritzSpaceCameraDistance } from '../src/components/learning/quantum-variational-space-engine';

test('real slices retain physical amplitude and residual lengths at every supported subspace dimension', () => {
  for (const aspect of [0.2, 0.5, 224 / 340, 1, 2, 4]) for (const radius of [1.4, 2.5, 10]) {
    const distance = ritzSpaceCameraDistance(radius, aspect), bound = Math.max(radius, 1.75);
    const apparentRadius = bound / Math.sqrt(distance * distance - bound * bound);
    assert(apparentRadius < Math.tan(21 * Math.PI / 180) * Math.min(1, aspect), 'the reference axes/labels must fit both camera dimensions');
  }
  for (const columns of [1, 2, 3]) {
    const response = evaluate({ version: 1, requestId: 1, input: space(kinds[1], 3, columns) });
    assert(response.ok && response.kind === kinds[1]);
    const g = geometry(response.result, 'full', 2)!;
    assert.equal(g.axes.length, columns);
    assert.equal(Math.hypot(...g.state), 1);
    assert.equal(Math.hypot(...g.residual), 0);
  }
});

test('geometry retains the residual decomposition and does not normalize residuals to unit length', () => {
  const response = evaluate({ version: 1, requestId: 1, input: preset(kinds[1], 1) });
  assert(response.ok && response.kind === kinds[1]);
  const full = geometry(response.result, 'full', 5)!, inside = geometry(response.result, 'inside', 5)!, outside = geometry(response.result, 'outside', 5)!;
  for (let j = 0; j < 3; j++) assert(Math.abs(full.residual[j] - inside.residual[j] - outside.residual[j]) < 1e-14);
  assert(Math.abs(Math.hypot(...full.residual) ** 2 - response.result.variance.value! / 25) < 1e-14);
  assert.notEqual(Math.hypot(...full.residual), 1);
  assert.equal(Math.hypot(...inside.residual), 0);
});

test('SSR emits accessible static projections and decorative covers without browser globals', async () => {
  const source = `import React from 'react';import{renderToStaticMarkup}from'react-dom/server';import Space from'./src/components/learning/quantum-variational-space';import Cover from'./src/components/learning/quantum-variational-cover';export function render(result,kind,language){return renderToStaticMarkup(<><Cover kind={kind}/><Space result={result} component="full" energyScale={5} language={language} scope="SSR"/></>)}`;
  const compiled = await build({ stdin: { contents: source, resolveDir: process.cwd(), loader: 'tsx' }, bundle: true, write: false, jsx: 'automatic', platform: 'node', format: 'cjs', external: ['react', 'react-dom/server'] });
  const module = { exports: {} as { render: (result: unknown, kind: string, language: string) => string } };
  new Function('require', 'module', 'exports', compiled.outputFiles[0].text)(createRequire(import.meta.url), module, module.exports);
  const response = evaluate({ version: 1, requestId: 1, input: preset(kinds[1], 0) });
  assert(response.ok);
  for (const kind of kinds) for (const language of ['ko', 'en']) {
    const html = module.exports.render(response.result, kind, language);
    assert(html.includes('data-qv-cover='));
    assert(html.includes('aria-hidden="true"'));
    assert(html.includes('data-qv-space="xy"'));
    assert(html.includes('role="img"'));
    assert(!html.includes('<canvas'));
    assert(!/NaN|Infinity/.test(html));
    if (language === 'en') assert(!/[가-힣]/.test(html));
  }
});

