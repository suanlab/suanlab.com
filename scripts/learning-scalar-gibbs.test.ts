import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { learningGuides, learningUnits } from '../src/data/interactive-book';
import { getLearningContent } from '../src/lib/learning-content';
import { calculateScalarGibbs } from '../src/lib/learning-scalar-gibbs';
import { readScalarGibbsInput, scalarGibbsUnitId } from '../src/lib/learning-scalar-gibbs-input';

const pair = { quartic: .25, quadratic: -.5, sources: [0,0], coupling: 1, radius: 4, panels: 32 };
const asNumber = (text: string) => {
  const [n,d='1'] = text.split('/');
  return Number(n)/Number(d);
};
const contains = (range: {lower: string; upper: string}, value: number) => {
  assert(asNumber(range.lower) <= value && value <= asNumber(range.upper), JSON.stringify({range,value}));
};

test('pure quartic mass and moments contain Gamma-integral values', () => {
  const result = calculateScalarGibbs({ ...pair,quartic: 1,quadratic: 0,sources: [0],coupling: 0 });
  contains(result.logPartition, Math.log(1.812804954110954));
  contains(result.secondMoments[0], .33798912003364236);
  contains(result.fourthMoments[0], .25);
  assert.deepEqual(result.means,[{lower:'0',upper:'0'}]);
});

test('coupled Gibbs enclosures contain independent quadrature under symmetric and nonuniform sources', () => {
  for (const sources of [[0,0],[.125,-.375]]) {
    const input = {...pair,sources}, result = calculateScalarGibbs(input);
    const reference = spawnSync('python3',['-c',`
import json, sys
import numpy as np
p=json.loads(sys.argv[1])
t,w=np.polynomial.legendre.leggauss(160)
x,y=7*t[:,None],7*t[None,:]
h0,h1=p['sources']
e=p['quartic']*(x**4+y**4)+p['quadratic']*(x*x+y*y)-h0*x-h1*y+p['coupling']*(x-y)**2/2
m=49*w[:,None]*w[None,:]*np.exp(-e)
z=m.sum()
mean=[(m*x).sum()/z,(m*y).sum()/z]
print(json.dumps([np.log(z),*mean,(m*x*x).sum()/z,(m*x**4).sum()/z,(m*x*y).sum()/z-mean[0]*mean[1]]))
`,JSON.stringify(input)],{encoding:'utf8'});
    assert.equal(reference.status,0,reference.stderr);
    const expected = JSON.parse(reference.stdout) as number[];
    [result.logPartition,...result.means,result.secondMoments[0],result.fourthMoments[0],result.covariance[0][1]]
      .forEach((range,i) => {
        if (sources.every(h => h === 0) && (i === 1 || i === 2)) {
          assert(Math.abs(expected[i]) < 1e-14);
          assert.deepEqual(range, { lower: '0', upper: '0' });
        } else {
          contains(range,expected[i]);
        }
      });
  }
});

test('panel refinement reduces box uncertainty at fixed tail and yields positive coupled covariance', () => {
  const coarse = calculateScalarGibbs(pair), fine = calculateScalarGibbs({...pair,panels:64});
  assert.equal(coarse.integrals[0].outsideAbsoluteUpper,fine.integrals[0].outsideAbsoluteUpper);
  const width = (x: {lower:string;upper:string}) => asNumber(x.upper)-asNumber(x.lower);
  assert(width(fine.integrals[0].boxIntegral) < width(coarse.integrals[0].boxIntegral)/4);
  assert(asNumber(fine.covariance[0][1].lower) > .49);
  assert(asNumber(fine.covariance[0][1].upper) < .55);
  assert.deepEqual(fine.originHessian,[[0,-1],[-1,0]]);
  assert.deepEqual(fine.means,[{lower:'0',upper:'0'},{lower:'0',upper:'0'}]);
});

test('supported controls preserve the hypotheses of the finite model', () => {
  for (const patch of [{quartic:0},{coupling:-1},{sources:[0]},{sources:[0,0,0]},{sources:[.1,0]},{panels:256},{radius:0}]) {
    assert.equal(readScalarGibbsInput({...pair,...patch}),null);
  }
});

test('the bilingual registered chapter renders mathematics and runs its shared Python exercise', async () => {
  const unit = learningUnits.find(item => item.id === scalarGibbsUnitId);
  assert(unit);
  assert.equal(learningGuides[unit.id].experiments.length, 2);
  const content = await getLearningContent(unit.id);
  const source = (locale: 'ko' | 'en') => readFileSync('content/learning/physics-math-for-ai/' + unit.id + '.' + locale + '.md', 'utf8');
  const exercise = (locale: 'ko' | 'en') => source(locale).match(/\x60{3}python\n([\s\S]*?)\n\x60{3}/)?.[1];
  for (const locale of ['ko', 'en'] as const) {
    assert.equal(content.outline[locale].length, 12);
    assert(content[locale].includes('<math'));
    assert(!content[locale].includes('katex-error'));
  }
  assert(!/[가-힣]/.test(content.en));
  const code = exercise('en');
  assert(code);
  assert.equal(code, exercise('ko'));
  const run = spawnSync('python3', ['-c', code], { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
});
