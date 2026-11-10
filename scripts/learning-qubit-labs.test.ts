import { test } from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import QubitLab from '../src/components/learning/qubit-lab';
import QubitResult from '../src/components/learning/qubit-result';
import { qubitScalarText, qubitCoordinateAt } from '../src/components/learning/qubit-lab-view';
import { qubitTeachingIds, qubitTeachingPresets } from '../src/lib/learning-qubit-teaching-input';
import { evaluateQubitRequest } from '../src/lib/learning-qubit-evaluator';
Object.assign(globalThis, { React });

for (const language of ['ko', 'en'] as const) for (const kind of qubitTeachingIds) {
  test(`SSR ${kind}/${language} shows input controls without touching Worker/storage/window`, () => {
    let reads = 0;
    const names = ['Worker', 'localStorage', 'window'] as const;
    const prior = names.map(name => Object.getOwnPropertyDescriptor(globalThis, name));
    try {
      names.forEach(name => Object.defineProperty(globalThis, name, { configurable: true, get() { reads++; throw Error(name); } }));
      const html = renderToStaticMarkup(React.createElement(QubitLab, { kind, language }));
      assert(html.includes('data-qubit-phase="idle"'));
      assert(html.includes(language === 'ko' ? '정확 입력 JSON' : 'Exact input JSON'));
      assert(!html.includes('data-qubit-result'));
      assert.equal(reads, 0);
      if (language === 'en') assert(!/[가-힣]/.test(html));
    } finally {
      names.forEach((name, i) => { if (prior[i]) Object.defineProperty(globalThis, name, prior[i]!); else Reflect.deleteProperty(globalThis, name); });
    }
  });
}
for (const language of ['ko', 'en'] as const) for (const preset of Object.values(qubitTeachingPresets).flat()) {
  test(`verified result renders meaningful observations: ${preset.id}/${language}`, () => {
    const response = evaluateQubitRequest({ version: 1, requestId: 1, input: preset.input }); assert(response.ok);
    const html = renderToStaticMarkup(React.createElement(QubitResult, { response, language }));
    assert(html.includes(`data-qubit-result="${preset.input.kind}"`));
    assert(html.includes('<table>') && html.includes('<caption'));
    assert(html.includes(language === 'ko' ? '정확' : 'Exact') || html.includes(language === 'ko' ? '순도' : 'Purity'));
    if (language === 'en') assert(!/[가-힣]/.test(html));
    assert(!html.includes('NaN') && !html.includes('undefined'));
  });
}
test('exact display keeps huge and tiny rational strings without parsing coefficients', () => {
  const numerator = '9'.repeat(500), denominator = '1' + '0'.repeat(500);
  assert.equal(qubitScalarText({ real: [`${numerator}/${denominator}`, '0'], imaginary: ['0', '0'] }), `${numerator}/${denominator}`);
  assert.equal(qubitScalarText({ real: ['0', '1/2'], imaginary: ['-1/3', '0'] }), '(0) + (1/2)√2 + i(-1/3)');
});
test('reader coordinate lookup consumes the certified leaf without recomputing mathematics', () => {
  const response = evaluateQubitRequest({ version: 1, requestId: 1, input: qubitTeachingPresets[qubitTeachingIds[0]][2].input }); assert(response.ok);
  const value = qubitCoordinateAt(response.observations.coordinates, ['prefixes', 3, 'probabilities', 0]);
  assert(value.real.lower! < value.real.upper!);
  assert(value.real.coordinate! >= value.real.lower! && value.real.coordinate! <= value.real.upper!);
});
