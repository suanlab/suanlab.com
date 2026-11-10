/** Independent Airy origin certificate from the POSITIVE incomplete-Gamma
 * series (DLMF8.7.1),not the producer's integrated alternating Taylor sum.
 */
import {createRealTimeCheckerInterval,type CheckerInterval as I} from './learning-real-time-checker-interval';
const O=BigInt(1);
let cached:ReturnType<typeof compute>|undefined;
function compute() {
  const ctx=createRealTimeCheckerInterval(6144),cutoff=1728,decay=ctx.expNegative(ctx.integer(cutoff));
  const target=O<<BigInt(ctx.bits-2600),terms:number[]=[];
  function gamma(numerator:1|2):Readonly<I> {
    // gamma(a,T)=T^a exp(-T) sum T^k/[a(a+1)...(a+k)],a=r/3.
    // T=12^3 makes T^a exactly12 or144. After the ratio drops below1/2,
    // the positive series remainder is at most twice the next term.
    let term=ctx.from({n:BigInt(3),d:BigInt(numerator)}),sum=term;
    const factor=numerator===1?12:144;
    for(let k=0;k<16000;k++) {
      const denominator=BigInt(3*(k+1)+numerator),next=ctx.divideInteger(ctx.scale(term,3*cutoff),denominator);
      const tail=ctx.scale(ctx.multiply(next,decay),2*factor);
      if(denominator>=BigInt(6*cutoff)&&tail.upper<target) {
        const partial=ctx.scale(ctx.multiply(sum,decay),factor);
        // For0<a<1 andt>=T>=1,t^(a-1)<=1,so the integral fromTtoinfinity
        // is <=exp(-T). No Decimal literal oruncertified quadrature is used.
        const result={lower:partial.lower,upper:partial.upper+tail.upper+decay.upper};
        if(result.lower<=BigInt(0)||result.upper-result.lower>=(O<<BigInt(ctx.bits-2400)))throw Error('Checker Gamma certificate width');
        terms.push(k+1);return Object.freeze(result);
      }
      sum=ctx.add(sum,next);term=next;
    }
    throw Error('Checker Gamma positive-series budget');
  }
  const oneThird=gamma(1),twoThirds=gamma(2),rootThree=ctx.root(ctx.integer(3),3);
  const value=ctx.divide(ctx.one,ctx.multiply(ctx.square(rootThree),twoThirds));
  const derivative=ctx.neg(ctx.divide(ctx.one,ctx.multiply(rootThree,oneThird)));
  for(const x of [value,derivative])if(x.upper-x.lower>=(O<<BigInt(ctx.bits-2400)))throw Error('Checker Airy origin width');
  return Object.freeze({bits:ctx.bits,value:Object.freeze(value),derivative:Object.freeze(derivative),
    gammaOneThird:oneThird,gammaTwoThirds:twoThirds,infiniteTailUpper:decay.upper,cutoff,terms:Object.freeze(terms),certifiedBits:2400});
}
export function checkerRealTimeAiryOrigin() {return cached??=compute();}
