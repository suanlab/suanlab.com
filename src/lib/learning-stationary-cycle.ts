/** Three named states, exact integer preparation weights and discrete transitions.
 * Stationary reference flows and evolving preparation flows remain distinct.
 */
import {exactMarkovObservation as observe} from './learning-finite-markov-exact';
import {rationalGreatestCommonDivisor as gcd} from './learning-rational-gcd';
import type {PartitionObservation} from './learning-partition-exact-core';

export interface StationaryCycleParameters {
 forwardWeight:number;
 backwardWeight:number;
 stayWeight:number;
 initialWeights:readonly number[];
 stepCount:number;
}
export interface StationaryCycleStep {
 step:number;
 probabilities:PartitionObservation[];
 deviationFromUniform:PartitionObservation[];
 totalVariationFromUniform:PartitionObservation;
 contractionUpperBound:PartitionObservation;
}
const zero=BigInt(0),one=BigInt(1),three=BigInt(3),six=BigInt(6);
const abs=(n:bigint)=>n<zero?-n:n;
const integerWeight=(n:unknown):n is number=>typeof n==='number'&&Number.isInteger(n)&&n>=0&&n<=1000000;

export function stationaryThreeStateCycleModel(input:StationaryCycleParameters) {
 if(!input||![input.forwardWeight,input.backwardWeight,input.stayWeight].every(integerWeight)||
  input.forwardWeight+input.backwardWeight+input.stayWeight===0||
  !Array.isArray(input.initialWeights)||input.initialWeights.length!==3||!Array.from(input.initialWeights).every(integerWeight)||
  input.initialWeights.reduce((a,b)=>a+b,0)===0||!Number.isInteger(input.stepCount)||input.stepCount<0||input.stepCount>512)
  throw new RangeError('Expected nonzero integer transition/initial weights in 0..1000000, three initial states, and 0..512 steps');
 const parameters=Object.freeze({forwardWeight:input.forwardWeight+0,backwardWeight:input.backwardWeight+0,
  stayWeight:input.stayWeight+0,initialWeights:Object.freeze(Array.from(input.initialWeights,n=>n+0)),stepCount:input.stepCount+0});
 const f=BigInt(input.forwardWeight),b=BigInt(input.backwardWeight),s=BigInt(input.stayWeight),D=f+b+s;
 const weights=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>i===j?s:j===(i+1)%3?f:b));
 const irreducible=f+b>zero,period=irreducible&&s===zero&&(f===zero||b===zero)?3:1;
 const maximum=[f,b,s].reduce((a,b)=>a>b?a:b),minimum=[f,b,s].reduce((a,b)=>a<b?a:b);
 const delta=maximum-minimum;
 let mass=input.initialWeights.map(BigInt),denominator=mass.reduce((a,b)=>a+b,zero);
 const initialTVNumerator=mass.reduce((sum,n)=>sum+abs(three*n-denominator),zero),initialTVDenominator=six*denominator;
 let boundNumerator=initialTVNumerator,boundDenominator=initialTVDenominator;
 const history:StationaryCycleStep[]=[];
 const evolve=(numerators:bigint[])=>weights[0].map((_,j)=>numerators.reduce((sum,n,i)=>sum+n*weights[i][j],zero));
 for(let step=0;step<=input.stepCount;step++){
  const differences=mass.map(n=>three*n-denominator);
  history.push({step,probabilities:mass.map(n=>observe(n,denominator)),deviationFromUniform:differences.map(n=>observe(n,three*denominator)),
   totalVariationFromUniform:observe(differences.reduce((sum,n)=>sum+abs(n),zero),six*denominator),
   contractionUpperBound:observe(boundNumerator,boundDenominator)});
  if(step===input.stepCount)break;
  mass=evolve(mass);denominator*=D;
  const common=mass.reduce((factor,n)=>gcd(factor,n),denominator);
  mass=mass.map(n=>n/common);denominator/=common;
  boundNumerator*=delta;boundDenominator*=D;
  const boundFactor=gcd(boundNumerator,boundDenominator);boundNumerator/=boundFactor;boundDenominator/=boundFactor;
 }
 const transition=weights.map(row=>row.map(n=>observe(n,D)));
 const stationaryFlux=weights.map(row=>row.map(n=>observe(n,three*D)));
 const stationaryCurrent=weights.map((row,i)=>row.map((n,j)=>observe(n-weights[j][i],three*D)));
 const stationaryBalance=weights.map((row,i)=>({state:i,incoming:observe(row.reduce((sum,_,j)=>sum+(i===j?zero:weights[j][i]),zero),three*D),
  outgoing:observe(row.reduce((sum,n,j)=>sum+(i===j?zero:n),zero),three*D),holding:observe(row[i],three*D),netOutflow:observe(zero)}));
 const nextMass=evolve(mass),nextDenominator=denominator*D;
 const currentFlux=weights.map((row,i)=>row.map(n=>observe(mass[i]*n,denominator*D)));
 const currentNetOutflow=mass.map((n,i)=>observe(n*D-nextMass[i],nextDenominator));
 const currentIsStationary=mass.every((n,i)=>n*D===nextMass[i]);
 const initialIsUniform=input.initialWeights.every(n=>n===input.initialWeights[0]);
 return {
  parameters,arithmetic:'exact-bigint-rational' as const,fullyResolved:true,
  transition,timeReversedTransition:transition.map((_,i)=>transition.map(row=>row[i])),
  stationaryReference:[0,1,2].map(()=>observe(one,three)),stationaryResidual:[0,1,2].map(()=>observe(zero)),
  stationaryFlux,stationaryCurrent,stationaryBalance,clockwiseCurrent:observe(f-b,three*D),
  conditionalForwardCycleProbability:observe(f*f*f,D*D*D),conditionalBackwardCycleProbability:observe(b*b*b,D*D*D),
  cycleProductDifference:observe(f*f*f-b*b*b,D*D*D),
  contractionCoefficient:observe(delta,D),history,current:history[history.length-1],
  nextDistribution:nextMass.map(n=>observe(n,nextDenominator)),currentFlux,currentNetOutflow,
  classification:{irreducible,stationaryReferenceIsUnique:irreducible,
   stationaryFamily:irreducible?'unique-uniform' as const:'all-distributions' as const,
   detailedBalanceAtUniform:f===b,returnPeriods:[period,period,period],
   uniqueClassPeriod:irreducible?period:null,
   communicatingClasses:irreducible?[[0,1,2]]:[[0],[1],[2]],
   convergesToUniformFromEveryInitial:irreducible&&period===1,
   convergesToUniformForThisInitial:initialIsUniform||irreducible&&period===1,
   currentDistributionIsStationary:currentIsStationary},
  approximationContract:'Three named discrete states with cyclic forward/backward/holding integer weights in 0..1000000, nonzero sum. Three nonnegative initial integer weights have a nonzero sum; t=0..512. All probabilities, currents, finite-time distributions, total variations and contraction bounds are exact BigInt rationals. Equal rational bounds certify every value; a rounded Number or approximate magnitude logarithm is only a display projection. Nonzero outside-display observations are not zero. The uniform stationary reference, its fluxes and time reversal are separate from the evolving preparation and its fluxes. Identity transitions have all distributions stationary; no unique uniform equilibrium or convergence to it is inferred. A directed no-hold cycle has period three even though its uniform reference is stationary. The total-variation coefficient max(p,q,r)-min(p,q,r) gives an upper bound, not a universal exact mixing rate. No Monte Carlo samples, physical time, local detailed balance, heat, thermodynamic entropy production or sampler correctness in other state spaces is inferred.'
 };
}
