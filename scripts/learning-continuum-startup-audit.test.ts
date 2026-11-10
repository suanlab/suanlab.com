import test from 'node:test';
import assert from 'node:assert/strict';
import {viscoelasticAt,viscoelasticWindow} from '../src/lib/learning-continuum-constitutive';
import refs from './fixtures/continuum-startup-audit-independent.json';
function check(actual:number|null,e:{value:number|null;sign:number}) {
  if(e.value===null||e.sign===0){assert.equal(actual,e.value);return;}
  assert.notEqual(actual,null);assert.equal(Math.sign(actual!),e.sign);
  assert.ok(Math.abs(actual!-e.value)<=Math.abs(e.value)*2e-10+Number.MIN_VALUE,`${actual} vs ${e.value}`);
}
for(const [i,r] of refs.cases.entries())test(`independent Maxwell startup ${i}`,()=>{
  const q=viscoelasticAt(r.parameters,r.time),w=viscoelasticWindow(r.parameters,0,r.time);
  for(const k of ['stress','stressRate','energy','energyRate','dissipation','power'] as const)check(q.maxwell[k],r.expected[k]);
  check(q.stableMaxwellStressChange,r.expected.stressChange);
  check(w.maxwell.energyChange,r.expected.energyChange);
  if(r.expected.stressChange.logMagnitude!==null)
    assert.ok(Math.abs(q.logStressChangeMagnitude!-r.expected.stressChange.logMagnitude)<2e-10);
});
test('rounded equal drive and release retain the nonzero initial derivative',()=>{
  const p={elasticModulus:1,viscosity:1,strainAmplitude:.001,angularFrequency:.7,initialElasticStrain:.001*.7};
  assert.equal(p.strainAmplitude*p.angularFrequency-p.initialElasticStrain,0);
  const q=viscoelasticAt(p,0),later=viscoelasticAt(p,1e-20);
  assert.ok(q.maxwell.stressRate!<0);
  assert.ok(later.stableMaxwellStressChange!< -1e-41);
});
