/** Binary classical-quantum ensembles and a selected measurement, not Iacc optimization. */
import { binaryQubitDiscrimination } from './learning-quantum-information-discrimination';
import { f, zero, one, two, add, sub, mul, div, eq, q, observe, type F, type R } from './learning-quantum-hilbert-core';
import { point, plus, minus, scale, intersect, binaryEntropyRange, qubitEntropyRange } from './learning-quantum-information-entropies';
import type { PartitionObservation, PartitionRational } from './learning-partition-exact-core';
type Bloch=readonly[F,F,F];
function decode(x:PartitionRational):F {
  const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);
  return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));
}
const range=(x:PartitionObservation):R=>{
  if(!x.bounds)throw new RangeError('Finite observation required');
  return {lower:decode(x.bounds.lower),upper:decode(x.bounds.upper)};
};
export function binaryQubitEnsemble(r0:Bloch,r1:Bloch,prior:F,measurement:'helstrom'|'z'='helstrom') {
  if(measurement!=='helstrom'&&measurement!=='z')throw new RangeError('Select helstrom or z');
  const discrimination=binaryQubitDiscrimination(r0,r1,prior),other=sub(one,prior);
  const averageBloch=r0.map((x,i)=>add(mul(prior,x),mul(other,r1[i])));
  const s0=qubitEntropyRange(r0),s1=qubitEntropyRange(r1),sAverage=qubitEntropyRange(averageBloch);
  const priorEntropy=binaryEntropyRange(point(prior));
  const equal=r0.every((x,i)=>eq(x,r1[i])),deterministic=eq(prior,zero)||eq(prior,one);
  const rawChi=minus(sAverage,plus(scale(s0,prior),scale(s1,other)));
  const chi=equal||deterministic?point(zero):intersect(rawChi,zero,priorEntropy.upper);
  const probabilities=measurement==='helstrom'?discrimination.conditionalOutcomeProbabilities.map(row=>range(row[0]))
    :[r0,r1].map(r=>point(div(add(one,r[2]),two)));
  const outputZero=intersect(plus(scale(probabilities[0],prior),scale(probabilities[1],other)));
  const outputEntropy=binaryEntropyRange(outputZero);
  const conditionalEntropy=plus(scale(binaryEntropyRange(probabilities[0]),prior),scale(binaryEntropyRange(probabilities[1]),other));
  // Nonnegativity and Holevo's bound are analytic constraints, not a numerical proof of the theorem.
  const information=equal||deterministic?point(zero):intersect(minus(outputEntropy,conditionalEntropy),zero,chi.upper);
  const joint=probabilities.map((p,i)=>[scale(p,i===0?prior:other),scale(minus(point(one),p),i===0?prior:other)]);
  return {prior:q(prior),averageBloch:averageBloch.map(q),measurement,
    conditionalOutcomeProbabilities:probabilities.map(p=>[observe(p),observe(minus(point(one),p))]),jointProbabilities:joint.map(row=>row.map(observe)),
    stateEntropies:[observe(s0),observe(s1)],averageStateEntropy:observe(sAverage),classicalLabelEntropy:observe(priorEntropy),
    outputEntropy:observe(outputEntropy),conditionalOutputEntropy:observe(conditionalEntropy),
    selectedMeasurementInformation:observe(information),holevoInformation:observe(chi),holevoGap:observe(intersect(minus(chi,information),zero,one)),
    rawMeasurementInformation:observe(minus(outputEntropy,conditionalEntropy)),rawHolevoInformation:observe(rawChi),
    approximationContract:'Exact rational physicality; directed512-bit spectral roots and256-bit logarithms in bits. A selected measurement is evaluated, not optimized for accessible information. Nonnegative/Holevo intervals are intersected with known analytic bounds; raw measurement information is retained. No channel capacity, Worker, public input reader or UI.'};
}
