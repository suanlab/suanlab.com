import test from 'node:test';import assert from 'node:assert/strict';
import {waveVerticalDomain as domain}from'../src/lib/learning-wave-axis';
import{evaluateWaveEquation,waveEquationKinds}from'../src/lib/learning-wave-equation-lab';
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-13,`${a} != ${b}`);
test('existing default axes retain zero,positive,signed and empty plotting domains',()=>{
 assert.deepEqual(domain([]),{minimum:-.01,maximum:.01});assert.deepEqual(domain([0,0]),{minimum:-.01,maximum:.01});close(domain([1,2]).minimum,-.16);close(domain([1,2]).maximum,2.16);close(domain([-2,1]).minimum,-2.24);close(domain([-2,1]).maximum,1.24);assert.deepEqual(domain([.0002,.0006]),{minimum:-.01,maximum:.0106});
});
test('actual chain local-energy variations become legible without changing energy values or zero',()=>{
 const r=evaluateWaveEquation(waveEquationKinds[0],[4,4,1,5,2,3,0,8,.5]).chain!,e=r.query.assignedEnergy,old=domain(e),scaled=domain(e,1e-12),variation=Math.max(...e)-Math.min(...e);
 assert.ok(scaled.minimum<0&&scaled.maximum>Math.max(...e));assert.ok(240*variation/(scaled.maximum-scaled.minimum)>150);assert.ok(240*variation/(old.maximum-old.minimum)<15);assert.ok(scaled.maximum<.002);close(e.reduce((a,b)=>a+b,0),r.query.energy);
});
test('small signed flows and zero-motion diagrams remain finite and include zero',()=>{
 const a=domain([-2e-6,1e-6],1e-12);assert.ok(a.minimum<-2e-6&&a.maximum>1e-6);assert.deepEqual(domain([0,0],1e-12),{minimum:-1e-12,maximum:1e-12});const b=domain([1e-17,-2e-17],1e-12);assert.ok(Number.isFinite(240/(b.maximum-b.minimum)));
});
test('invalid padding and nonfinite plotted values are rejected rather than creating broken SVG paths',()=>{for(const p of[0,-1,NaN,Infinity])assert.throws(()=>domain([0,1],p));for(const v of[NaN,Infinity,-Infinity])assert.throws(()=>domain([v],1e-12));});
