import test from 'node:test';
import assert from 'node:assert/strict';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
const read=(s:string)=>{const[n,d='1']=s.split('/');return{n:BigInt(n),d:BigInt(d)};};
test('scaled representation is required when a resolved decay times a small prefactor loses the raw nonzero bound',()=>{
  const r=calculateRealTimePhase({mode:'chain',intervals:32,time:'16',mass:'1/100000',hbar:'1/999999999999',omegaSquared:'0',start:'15',end:'15',damping:'16',probe:16});
  assert(r.kind==='chain'&&r.wave.regulated);
  const wave=r.wave.regulated,decay=read(wave.negativeExponential.upper);
  // ln(2)>2*(1/3+1/81)=56/81>0.69, so decay<5630
  // proves exp(-decay)>2^-8192.
  // The additional small prefactor still makes the final raw bound include0.
  assert(decay.n<BigInt(5630)*decay.d);
  assert(wave.scaledAmplitude.scalePowerOfTwo<-200);
  assert(wave.rawValueRequiresScaledRepresentation);
  for(const v of [wave.value.real,wave.value.imaginary])assert(read(v.lower).n<=BigInt(0)&&read(v.upper).n>=BigInt(0));
  assert(read(wave.scaledAmplitude.lower).n>BigInt(0));assert(wave.logMagnitude);
});
