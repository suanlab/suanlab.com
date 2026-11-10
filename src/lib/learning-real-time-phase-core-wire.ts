/** Lossless exact finite-core transport; no dense huge-denominator repetition.
 * These factors specify every matrix entry, including zeros and signs. Later
 * numerical observations/teaching views do not replace the exact factors.
 */
import type {buildRealTimeChainExact,buildRealTimePacketExact} from './learning-real-time-phase-exact';
import type {F,C} from './learning-quantum-hilbert-core';
const text=(x:F)=>x.d===BigInt(1)?String(x.n):`${x.n}/${x.d}`;
const complex=(x:C)=>({real:text(x.re),imaginary:text(x.im)});
const matrix=(a:F[][])=>a.map(row=>row.map(text));
export function realTimeChainCoreWire(r:ReturnType<typeof buildRealTimeChainExact>) {
  return {kind:'finite-chain-core' as const,input:r.input,dimension:r.d,
    hessian:{diagonal:text(r.diagonal),offDiagonal:text({n:-r.s.n,d:r.s.d})},
    determinant:text(r.determinant),forcing:r.forcing.map(text),constant:text(r.constant),
    inertia:r.inertia,stationaryStatus:r.stationaryStatus,stationary:r.stationary?.map(text)??null,
    stationaryAction:r.stationaryAction?text(r.stationaryAction):null,
    nullVector:r.nullVector?.map(text)??null,compatible:r.compatible,causticIndex:r.causticIndex,
    inverse:r.inertia.zero?null:{layout:'symmetric-continuant' as const,dimension:r.d,
      continuants:r.theta.map(text),scale:text(r.s)},
    map:matrix(r.map.total),refinements:r.refinements.map(row=>({multiplier:row.multiplier,intervals:row.intervals,map:matrix(row.map)})),
    regulator:r.regulator?{layout:'symmetric-complex-continuant' as const,dimension:r.d,
      continuants:r.regulator.theta.map(complex),distancePowers:r.regulator.distancePowers.map(complex),
      determinant:complex(r.regulator.determinant),pivotLayout:'successive-continuant-ratios' as const,
      pivotRootBranch:'positive-real-part-per-ratio' as const,
      mean:r.regulator.mean.map(complex),exponent:complex(r.regulator.exponent)}:null};
}
export function realTimePacketCoreWire(r:ReturnType<typeof buildRealTimePacketExact>) {
  return {kind:'finite-packet-core' as const,input:r.input,variance:text(r.variance),
    initialCovariance:matrix(r.initialCovariance),finalCovariance:matrix(r.finalCovariance),
    center:r.center.map(text),centralAction:text(r.centralAction),initialWidth:complex(r.initialWidth),finalWidth:complex(r.finalWidth),
    negativeModes:r.negativeModes,causticIndex:r.causticIndex,map:matrix(r.map.total),
    refinements:r.refinements.map(row=>({multiplier:row.multiplier,intervals:row.intervals,map:matrix(row.map)}))};
}
