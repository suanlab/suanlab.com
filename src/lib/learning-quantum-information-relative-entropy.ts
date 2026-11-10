/** Umegaki relative entropy of rational qubits, including exact support failure. */
import { binaryQubitDiscrimination } from './learning-quantum-information-discrimination';
import { f, zero, one, two, div, eq, q, observe, sqrtRange, type F, type R } from './learning-quantum-hilbert-core';
import { point, plus, minus, scale, divide, times, intersect, log2Range, qubitEntropyRange, dot } from './learning-quantum-information-entropies';
type Bloch=readonly[F,F,F];
export function qubitRelativeEntropy(rho:Bloch,sigma:Bloch) {
  binaryQubitDiscrimination(rho,sigma,div(one,two)); // Validate raw states before spectral operations.
  const equal=rho.every((x,i)=>eq(x,sigma[i])),radiusSquared=dot(sigma,sigma);
  const contract='Umegaki D(rho||sigma) in bits, exact rational support classification; directed512-bit roots and256-bit logarithms. The logarithm is spectral, not entrywise. No arbitrary-dimensional solver, Petz recovery, Worker, public input reader or UI.';
  if(eq(radiusSquared,one)&&!equal)return {status:'infinite' as const,value:null,reason:'support-not-contained',sigmaSpectrum:[q(one),q(zero)],contract};
  if(equal)return {status:'finite' as const,value:observe(point(zero)),reason:'equal-states',sigmaSpectrum:null,contract};
  const entropy=qubitEntropyRange(rho);
  if(eq(radiusSquared,zero))return {status:'finite' as const,value:observe(intersect(minus(point(one),entropy))),reason:'maximally-mixed-reference',sigmaSpectrum:[q(div(one,two)),q(div(one,two))],contract};
  const radius=sqrtRange(radiusSquared),sPlus=scale(plus(point(one),radius),div(one,two)),sMinus=scale(minus(point(one),radius),div(one,two));
  if(sMinus.lower.n<=BigInt(0))throw new RangeError('Reference spectrum unresolved at precision ceiling');
  const alignment=divide(point(dot(rho,sigma)),radius);
  const weights=[intersect(scale(plus(point(one),alignment),div(one,two))),intersect(scale(minus(point(one),alignment),div(one,two)))];
  const cross=plus(times(weights[0],log2Range(sPlus)),times(weights[1],log2Range(sMinus)));
  const raw=minus(scale(entropy,f(BigInt(-1))),cross);
  const value:R={lower:raw.lower.n<BigInt(0)?zero:raw.lower,upper:raw.upper};
  if(value.upper.n<BigInt(0))throw new RangeError('Relative entropy enclosure is negative');
  return {status:'finite' as const,value:observe(value),reason:'full-rank-reference',sigmaSpectrum:[observe(sPlus),observe(sMinus)],
    spectralWeights:weights.map(observe),rawValue:observe(raw),contract};
}
export function qubitDephasingDataProcessing(rho:Bloch,sigma:Bloch) {
  const before=qubitRelativeEntropy(rho,sigma),after=qubitRelativeEntropy([zero,zero,rho[2]],[zero,zero,sigma[2]]);
  return {channel:'complete-z-dephasing',before,after,
    interpretation:'The same CPTP channel acts on both states. Infinite input divergence may become finite. Separate enclosures are evidence for examples, not a proof of general data processing or recovery.'};
}
