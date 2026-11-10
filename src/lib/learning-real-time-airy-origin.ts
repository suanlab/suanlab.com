/** Certified origin constants computed in the Worker, not decimal literals.
 * Positive Gamma integrals, finite integrated Taylor bounds and analytic
 * infinite tails are documented in real-time-airy-origin-certificate.md.
 */
import {createRealTimeIntervalContext,realTimeCeil,realTimeIntegerRoot,type RealTimeInterval} from './learning-real-time-interval';
let cache:Readonly<{value:Readonly<RealTimeInterval>;derivative:Readonly<RealTimeInterval>}>|undefined;
export function certifiedRealTimeAiryOrigin() {
  if(cache)return cache;
  const ctx=createRealTimeIntervalContext(),S=ctx.S;
  let power={lower:S,upper:S};const lower=[BigInt(0),BigInt(0)],upper=[BigInt(0),BigInt(0)];
  const radius=11,T=BigInt(radius**3),terms=5000;
  for(let k=0;k<terms;k++) {
    for(let i=0;i<2;i++) {
      const d=BigInt(3*k+i+1),lo=power.lower/d,hi=realTimeCeil(power.upper,d);
      if(k%2){lower[i]-=hi;upper[i]-=lo;}else{lower[i]+=lo;upper[i]+=hi;}
    }
    power={lower:power.lower*T/BigInt(k+1),upper:realTimeCeil(power.upper*T,BigInt(k+1))};
  }
  const gamma=[1,2].map((r,i)=>{
    const factor=BigInt(3*radius**r);
    const bounds={lower:factor*lower[i],upper:factor*upper[i]+realTimeCeil(factor*power.upper,BigInt(3*terms+r))+(BigInt(1)<<BigInt(ctx.workBits-1862))};
    if(bounds.lower<=BigInt(0)||bounds.upper-bounds.lower>=(BigInt(1)<<BigInt(ctx.workBits-1800)))throw Error('Gamma origin certificate unresolved');
    return bounds;
  });
  const root=realTimeIntegerRoot(BigInt(3)*S**BigInt(3),3),rootThree={lower:root,upper:root+BigInt(1)};
  const value=ctx.divide(ctx.one,ctx.multiply(ctx.square(rootThree),gamma[1]));
  const derivative=ctx.neg(ctx.divide(ctx.one,ctx.multiply(rootThree,gamma[0])));
  cache=Object.freeze({value:Object.freeze(value),derivative:Object.freeze(derivative)});return cache;
}
