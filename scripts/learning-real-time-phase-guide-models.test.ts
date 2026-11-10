import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdirSync,writeFileSync} from 'node:fs';
import {realTimePhaseGuidePreparations} from '../src/data/real-time-phase-teaching';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
import {evaluateRealTimePhaseValidationRequest} from '../src/lib/learning-real-time-phase-validation-evaluator';
import {realTimePhaseInputKey} from '../src/lib/learning-real-time-phase-protocol';
const records:Record<string,unknown>={},root='.runtime/learning/real-time-phase-lessons';mkdirSync(root,{recursive:true});
for(const prep of realTimePhaseGuidePreparations)test(`guide calculation and independent gate: ${prep.id}`,()=>{
  const input=prep.input,result=calculateRealTimePhase(input),response={version:1 as const,requestId:1,ok:true as const,kind:'real-time-phase' as const,input,inputKey:realTimePhaseInputKey(input),result};
  const receipt=evaluateRealTimePhaseValidationRequest({version:1,requestId:1,input,response});assert.equal(receipt.phase,'validated',prep.id);
  records[prep.id]=result.kind==='chain'?{kind:result.kind,stationary:result.exact.stationary,action:result.exact.stationaryAction,inertia:result.exact.inertia,map:result.exact.map,status:result.exact.stationaryStatus}:result.kind==='packet'?{kind:result.kind,covariance:result.exact.finalCovariance,map:result.exact.map}: {kind:result.kind,regime:result.regime,stationary:result.stationaryPoints.length,sign:result.scaledSign};
  writeFileSync(root+'/guide-results.json',JSON.stringify(records,null,2)+'\n');
});
test('independent Fraction hand calculations match stated guide readouts',()=>{
  const python=spawnSync('python3',['-c',`
import json
from fractions import Fraction as F
data=json.load(open('.runtime/learning/real-time-phase-lessons/guide-results.json'))
assert data['positive-hand']['stationary']==['1'] and data['positive-hand']['action']=='-1/4'
assert data['negative-hand']['stationary']==['-1'] and data['negative-hand']['action']=='1/4'
assert data['positive-hand']['inertia']=={'positive':1,'negative':0,'zero':0}
assert data['negative-hand']['inertia']=={'positive':0,'negative':1,'zero':0}
assert data['zero-pivot-inverse']['inertia']=={'positive':1,'negative':1,'zero':0}
assert data['zero-pivot-singular']['inertia']['zero']==1
assert data['caustic-compatible']['status']=='affine-family'
assert data['caustic-incompatible']['status']=='no-stationary-path'
assert data['packet-parity']['map']==[['-1','0'],['0','-1']]
for id,h,chirp,scaled in [('fixed-width',F(1,2),0,False),('fixed-half-hbar',F(1,4),0,False),('scaled-width',F(1,2),0,True),('scaled-half-hbar',F(1,4),0,True),('chirp-half',F(1,2),-1,False),('chirp-quarter',F(1,4),-1,False),('chirp-eighth',F(1,8),-1,False)]:
 variance=F(2,3)*h if scaled else F(2,3)
 pp=chirp*chirp*variance+h*h/(4*variance)
 qp=chirp*variance+pp
 qq=variance+2*chirp*variance+pp
 assert data[id]['covariance']==[[str(qq),str(qp)],[str(qp),str(pp)]],id
 assert qq*pp-qp*qp==h*h/4
assert data['fold-positive']['stationary']==2
assert data['fold-zero']['stationary']==1
assert data['fold-negative']['stationary']==0 and data['fold-negative']['sign']=='positive'
print('Guide hand calculations and covariance invariants verified.')
`],{encoding:'utf8'});assert.equal(python.status,0,python.stderr);assert(python.stdout.includes('verified'));
});
