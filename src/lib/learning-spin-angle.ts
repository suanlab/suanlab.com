/** Exact turn reduction and a proved pi enclosure; never binary64 angle snapping. */
import {f,zero,add,sub,mul,div,type R,type F} from './learning-quantum-hilbert-core';
import type {createQuantumEvolutionContext} from './learning-quantum-evolution-interval';
let cached:R|undefined;
export function spinPiRange():R{
 if(cached)return cached;
 function atanInverse(d:number):R{const x=f(BigInt(1),BigInt(d)),x2=mul(x,x);let power=x,sum=zero;const terms=180;for(let k=0;k<terms;k++){const term=div(power,f(BigInt(2*k+1)));sum=k%2?sub(sum,term):add(sum,term);power=mul(power,x2);}return{lower:sum,upper:add(sum,div(power,f(BigInt(2*terms+1))))};}
 const a=atanInverse(5),b=atanInverse(239),range={lower:sub(mul(f(BigInt(16)),a.lower),mul(f(BigInt(4)),b.upper)),upper:sub(mul(f(BigInt(16)),a.upper),mul(f(BigInt(4)),b.lower))};
 cached=Object.freeze({lower:Object.freeze(range.lower),upper:Object.freeze(range.upper)});return cached;
}
/** Returns sin(pi*turns),cos(pi*turns): spinor half-angle of a 2pi*turns rotation. */
export function spinHalfAngleTrig(ctx:ReturnType<typeof createQuantumEvolutionContext>,turns:F){
 if(turns.d<=BigInt(0))throw new RangeError('Positive turn denominator required');const period=BigInt(2)*turns.d,r=f((turns.n%period+period)%period,turns.d),twice=BigInt(2)*r.n;
 if(twice%r.d===BigInt(0)){const k=Number(twice/r.d),values=[ctx.zero,ctx.one,ctx.zero,ctx.neg(ctx.one)];return{sin:values[k],cos:values[(k+1)%4]};}
 const phase=ctx.multiply(ctx.range(spinPiRange()),ctx.from(r));return ctx.sinCos(ctx.rationalRange(phase));
}
