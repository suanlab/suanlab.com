import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { forcedOscillatorState, oscillatorFrequencyResponse } from '../src/lib/learning-waves-oscillations';

const units = [
  { id: 'sinusoidal-forcing-transient-response', equations: 7, image: 'limit-clock', original: ['eb80d89828eaaff8ce2999dfa828331be43c896796d35817723487dd7580210a', '6cac84d103936e957a70c07a525ef514acbfc3864d9fb559874909516acc8175'], rows: [[.2,5,.36813,.399576],[.1,10,.419481,.399576],[.05,20,.407897,.399576],[.01,100,.397531,.399576]] },
  { id: 'oscillator-frequency-response-power-resonance', equations: 8, image: 'spectral-area', original: ['5bd9993765ebd7fee20a502f8b33823fa9d33daa5ffb97afff85322621381781', 'b64e32da3ff8829d1ab19af0b00c26f05d93744a95c62cc3cd78907e60d54146'], rows: [[.05,.920064,.995979,1],[.2,.702791,.98392,1],[1,.241925,.920008,1],[4,.063445,.702273,1]] },
];

for (const unit of units) test(`resonance research preserves originals and pairs proof, code and tables: ${unit.id}`, async () => {
  const pairs = ['en', 'ko'].map((lang, index) => {
    const source = readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`, 'utf8');
    assert.doesNotMatch(source, /[\u0000-\u0008\u000b-\u001f]/);
    if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/);
    const start = source.search(/^## (Research extension:|연구 확장:)/m);
    assert.ok(start >= 0);
    const headerEnd = source.indexOf('\n', start);
    const next = source.slice(headerEnd).search(/^## /m);
    assert.ok(next >= 0);
    const end = headerEnd + next;
    assert.equal(createHash('sha256').update(source.slice(0,start)+source.slice(end)).digest('hex'), unit.original[index]);
    assert.equal((source.slice(start,end).match(lang === 'en' ? /\*\*Answer:\*\*/g : /\*\*답:\*\*/g) || []).length,3);
    const research = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start,end));
    const math = research.children.flatMap(n => n.type === 'math' ? [n.value] : []);
    const code = research.children.flatMap(n => n.type === 'code' && n.lang === 'python' ? [n.value] : []);
    const tables = research.children.filter(n => n.type === 'table');
    assert.equal(math.length, unit.equations);
    assert.equal(code.length,1);
    assert.equal(tables.length,1);
    assert.equal(tables[0].children.length,5);
    assert.ok(tables[0].children.every(row => row.children.length===4));
    const stdout = execFileSync('python3',['-c',code[0]],{encoding:'utf8',timeout:20000});
    assert.deepEqual(stdout.trim().split('\n').map(row => row.split(/\s+/).map(Number)),unit.rows);
    return {math,code,stdout};
  });
  assert.deepEqual(pairs[0],pairs[1]);
  const content = await getLearningContent(unit.id);
  for (const lang of ['en','ko'] as const) {
    assert.ok(!content[lang].includes('katex-error'));
    assert.ok(content[lang].includes(`waves-resonance-${unit.image}.${lang}.svg`));
    assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length,1);
  }
});

test('independent ODE refinement, nonuniform limits, positive spectral area and tail bounds', () => {
  const result = JSON.parse(execFileSync('python3',['scripts/learning/verify-resonance-research.py','--json'],{encoding:'utf8',timeout:20000}));
  assert.deepEqual(result,{finiteWindowIntegrals:8,initialValueJets:144,positivePowerStates:225,rk4Refinements:36,scaledLimitStates:54,spectralIntegrals:45,tailBounds:135,transientBounds:144,weakDampingFixedTimes:6});
});

test('native finite-time resonant states agree with the supplement within supported times', () => {
  for (const damping of [.1,.2,.4]) for (const time of [1,10,20,30]) {
    const beta = damping/2, d=Math.sqrt(1-beta*beta), decay=Math.exp(-beta*time);
    const expectedX=(Math.sin(time)-decay*Math.sin(d*time)/d)/damping;
    const expectedV=(Math.cos(time)-decay*(Math.cos(d*time)-beta*Math.sin(d*time)/d))/damping;
    const actual=forcedOscillatorState({mass:1,stiffness:1,damping,forceAmplitude:1,frequency:1},{position:0,velocity:0},time);
    assert.ok(Math.abs(actual.position-expectedX)<2e-9);
    assert.ok(Math.abs(actual.velocity-expectedV)<2e-9);
  }
});

test('native mean power uses the stated passive sign across weak, critical and strong damping', () => {
  for (const damping of [.05,.2,1,2,4]) for (const frequency of [.1,.5,1,2,8]) {
    const actual=oscillatorFrequencyResponse({mass:1,stiffness:1,damping,forceAmplitude:2,frequency});
    const weight=damping*frequency*frequency/((1-frequency*frequency)**2+damping*damping*frequency*frequency);
    assert.ok(actual.susceptibility && actual.meanPower !== null);
    assert.ok(Math.abs(actual.meanPower-2*weight)<1e-12);
    assert.ok(Math.abs(weight+frequency*actual.susceptibility.imaginary)<1e-12);
  }
});
