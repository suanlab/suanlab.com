import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/continuum-stress-audit-independent.json';
import {affineDeformationState,planeStressTraction} from '../src/lib/learning-continuum-stress-strain';
function match(actual:number|null,expected:{value:number|null;sign:number},name:string){
 if(expected.value===null){assert.equal(actual,null,name);return;}
 assert.notEqual(actual,null,name);
 assert.equal(Math.sign(actual!),expected.sign,name+' sign');
 assert.ok(Math.abs(actual!-expected.value)<=2*Number.MIN_VALUE+2e-12*Math.abs(expected.value),`${name}: ${actual} vs ${expected.value}`);
}
for(const [i,c]of refs.strainCases.entries())test(`independent mapped-vector strain audit ${i+1}`,()=>{
 const q=affineDeformationState(c.model,c.angle);
 for(const [j,value]of [q.green[0][0],q.green[0][1],q.green[1][1]].entries())match(value,c.expected.green[j],'Green '+j);
 match(q.directionalGreen,c.expected.directionalGreen,'directional Green');match(q.stableEngineeringStrain,c.expected.engineering,'engineering');
});
for(const [i,c]of refs.principalCases.entries())test(`independent1600-digit principal stress audit ${i+1}`,()=>{
 const q=planeStressTraction(c.stress,0,0,0);
 q.principalValues.forEach((v,j)=>match(v,c.expected.values[j],'principal '+j));
 if(c.expected.direction===null)assert.equal(q.principalDirection,null);else {assert.notEqual(q.principalDirection,null);assert.ok(Math.abs(q.principalDirection!-c.expected.direction)<1e-14);}
});
test('a missing tiny tensor component does not erase a representable direction observation',()=>{
 const q=affineDeformationState({stretchX:1,stretchY:1,shear:4*Number.MIN_VALUE,rotation:0,translationX:0,translationY:0},Math.PI/4);
 assert.equal(q.green[1][1],null);assert.equal(q.directionalGreen,2*Number.MIN_VALUE);assert.equal(q.stableEngineeringStrain,2*Number.MIN_VALUE);
});
test('a tiny shear square does not poison a finite transverse extension',()=>{
 const q=affineDeformationState({stretchX:1,stretchY:1.5,shear:Number.MIN_VALUE,rotation:0,translationX:0,translationY:0},Math.PI/2);
 assert.equal(q.green[1][1],.625);assert.ok(q.stableEngineeringStrain!==null&&Math.abs(q.stableEngineeringStrain-.5)<1e-14);
});
