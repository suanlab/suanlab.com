/** Exact rational two-state distribution evolution, including absorbing,
 * alternating and identity preparations. No canonical equilibrium is invented.
 */
import {markovFraction as fraction,exactMarkovObservation as observe} from './learning-finite-markov-exact';
import {rationalGreatestCommonDivisor as gcd} from './learning-rational-gcd';
import type {PartitionObservation} from './learning-partition-exact-core';
export interface TwoStateMixingParameters {
 forwardNumerator:number;
 backwardNumerator:number;
 transitionDenominator:number;
 initialOneNumerator:number;
 initialDenominator:number;
 stepCount:number;
}
export interface TwoStateMixingStep {
 step:number;
 probabilities:PartitionObservation[];
 closedFormProbabilityOne:PartitionObservation;
 closedFormMatchesEvolution:boolean;
 deviationFromStationary:PartitionObservation[]|null;
 totalVariationFromStationary:PartitionObservation|null;
 worstCaseTotalVariation:PartitionObservation|null;
 forwardFlux:PartitionObservation;
 backwardFlux:PartitionObservation;
 netForwardFlux:PartitionObservation;
 cesaroProbabilityOne:PartitionObservation;
 cesaroDistanceFromStationary:PartitionObservation|null;
}
const Z=BigInt(0),O=BigInt(1),TWO=BigInt(2),FOUR=BigInt(4);
const abs=(n:bigint)=>n<Z?-n:n;
const integer=(n:unknown,positive=false):n is number=>typeof n==='number'&&Number.isInteger(n)&&n>=(positive?1:0)&&n<=1000000;

export function reversibleTwoStateMixingModel(input:TwoStateMixingParameters) {
 if(!input||!integer(input.transitionDenominator,true)||!integer(input.initialDenominator,true)||
  !integer(input.forwardNumerator)||!integer(input.backwardNumerator)||
  input.forwardNumerator>input.transitionDenominator||input.backwardNumerator>input.transitionDenominator||
  !integer(input.initialOneNumerator)||input.initialOneNumerator>input.initialDenominator||
  !Number.isInteger(input.stepCount)||input.stepCount<0||input.stepCount>512)
  throw new RangeError('Expected exact rational probabilities in [0,1], denominators 1..1000000, and 0..512 transitions');
 const parameters=Object.freeze({forwardNumerator:input.forwardNumerator+0,backwardNumerator:input.backwardNumerator+0,
  transitionDenominator:input.transitionDenominator,initialOneNumerator:input.initialOneNumerator+0,
  initialDenominator:input.initialDenominator,stepCount:input.stepCount+0});
 const A=BigInt(input.forwardNumerator),B=BigInt(input.backwardNumerator),D=BigInt(input.transitionDenominator),K=A+B;
 const U=BigInt(input.initialOneNumerator),V=BigInt(input.initialDenominator),L=D-K;
 const unique=K>Z,alternating=K===TWO*D,irreducible=A>Z&&B>Z;
 const initialStationary=A*(V-U)===B*U;
 const stationary=unique?[observe(B,K),observe(A,K)]:null;
 const transition=[[observe(D-A,D),observe(A,D)],[observe(B,D),observe(D-B,D)]];
 const history:TwoStateMixingStep[]=[];
 let mass=[V-U,U],denominator=V,powerL=O,powerD=O,sumOne=fraction(Z),firstQuarter:number|null=null;
 for(let step=0;step<=input.stepCount;step++){
  sumOne=fraction(sumOne.n*denominator+mass[1]*sumOne.d,sumOne.d*denominator);
  const count=BigInt(step+1),averageDen=sumOne.d*count;
  const deviation=unique?mass[1]*K-A*denominator:Z;
  const closedNumerator=unique?A*V*powerD+(U*K-A*V)*powerL:U;
  const closedDenominator=unique?V*K*powerD:V;
  const worstNumerator=unique?(A>B?A:B)*abs(powerL):Z,worstDenominator=unique?K*powerD:O;
  if(unique&&firstQuarter===null&&FOUR*worstNumerator<=worstDenominator)firstQuarter=step;
  history.push({step,probabilities:mass.map(n=>observe(n,denominator)),
   closedFormProbabilityOne:observe(closedNumerator,closedDenominator),
   closedFormMatchesEvolution:closedNumerator*denominator===mass[1]*closedDenominator,
   deviationFromStationary:unique?[observe(-deviation,denominator*K),observe(deviation,denominator*K)]:null,
   totalVariationFromStationary:unique?observe(abs(deviation),denominator*K):null,
   worstCaseTotalVariation:unique?observe(worstNumerator,worstDenominator):null,
   forwardFlux:observe(mass[0]*A,denominator*D),backwardFlux:observe(mass[1]*B,denominator*D),
   netForwardFlux:observe(mass[0]*A-mass[1]*B,denominator*D),
   cesaroProbabilityOne:observe(sumOne.n,averageDen),
   cesaroDistanceFromStationary:unique?observe(abs(sumOne.n*K-A*averageDen),averageDen*K):null});
  if(step===input.stepCount)break;
  mass=[(D-A)*mass[0]+B*mass[1],A*mass[0]+(D-B)*mass[1]];denominator*=D;
  const common=gcd(gcd(denominator,mass[0]),mass[1]);mass=mass.map(n=>n/common);denominator/=common;
  powerL*=L;powerD*=D;
 }
 const identity=!unique,hasPointwiseLimit=identity||!alternating||initialStationary;
 const initial=[observe(V-U,V),observe(U,V)],current=history[history.length-1];
 const classes=irreducible?[[0,1]]:[[0],[1]],closed=irreducible?[[0,1]]:identity?[[0],[1]]:A===Z?[[0]]:[[1]];
 const periods:(number|null)[]=irreducible?(alternating?[2,2]:[1,1]):[A===D?null:1,B===D?null:1];
 const uniqueSupport=unique?[...(B>Z?[0]:[]),...(A>Z?[1]:[])]:null;
 const stationaryTimeReversal=unique?[
  B>Z?[observe(D-A,D),observe(A,D)]:null,
  A>Z?[observe(B,D),observe(D-B,D)]:null,
 ]:null;
 const variance=unique?observe(A*B,K*K):null;
 const average=fraction(sumOne.n,sumOne.d*BigInt(input.stepCount+1)),averageN=average.n,averageD=average.d;
 return {parameters,arithmetic:'exact-bigint-rational' as const,fullyResolved:true,transition,initial,
  secondEigenvalue:observe(L,D),absoluteContractionCoefficient:observe(abs(L),D),
  signedSpectralGap:observe(K,D),absoluteSpectralGap:observe(D-abs(L),D),
  stationaryDistribution:stationary,stationaryFamily:unique?'unique-distribution' as const:'all-distributions' as const,
  stationaryFamilyDimension:unique?0:1,stationarySupport:uniqueSupport,
  stationaryBalanceFlow:unique?observe(A*B,K*D):null,stationaryTimeReversal,
  stationaryIndicatorVariance:variance,stationaryLagOneCovariance:unique?observe(A*B*L,K*K*D):null,
  stationaryLagOneCorrelation:irreducible?observe(L,D):null,
  history,current,nextDistribution:[observe((D-A)*mass[0]+B*mass[1],denominator*D),observe(A*mass[0]+(D-B)*mass[1],denominator*D)],
  cesaroAverage:[observe(averageD-averageN,averageD),observe(averageN,averageD)],
  limitDistribution:hasPointwiseLimit?(identity?initial:stationary):null,
  cesaroLimitDistribution:identity?initial:stationary,
  mixingAtQuarter:{criterion:'worst-initial total variation at most 1/4' as const,
   status:!unique?'not-applicable-nonunique-stationary' as const:alternating?'never-periodic' as const:
    firstQuarter===null?'not-reached-within-recorded-range' as const:'reached-within-recorded-range' as const,firstStep:firstQuarter},
  classification:{irreducible,communicatingClasses:classes,closedCommunicatingClasses:closed,returnPeriods:periods,
   uniqueClosedClassPeriod:unique?(alternating?2:1):null,stationaryIsUnique:unique,
   allStationaryLawsSatisfyDetailedBalance:true,initialDistributionIsStationary:initialStationary,
   currentDistributionIsStationary:mass[0]*A===mass[1]*B,
   hasPointwiseLimitForThisInitial:hasPointwiseLimit,
   convergesToUniqueStationaryFromEveryInitial:unique&&!alternating,
   convergesToUniqueStationaryForThisInitial:unique?!alternating||initialStationary:null},
  approximationContract:'Two named states and a fixed row-stochastic discrete-time kernel [[1-a,a],[b,1-b]]. Supplied integer numerators and denominators represent exact rationals, denominators 1..1000000 and probabilities in [0,1]; initial probability is independently specified, t=0..512. Exact integer mass evolution is compared with a separate closed recurrence formula. All observations, signed deviations, fluxes, Cesaro means, total variations and the worst-initial distance are exact rational points. Number and magnitude-log fields are display projections; nonzero outside-display values are never zero. Identity transitions have a stationary family and no selected reference distance, but each initial distribution is its own pointwise and Cesaro limit. The alternating chain has a unique stationary law without convergence from nonstationary initial distributions. One-way absorbing chains can converge to a unique stationary law despite reducibility. Stationary reverse rows are provided only on positive stationary support; degenerate indicator covariance is zero but normalized correlation is not provided. Cesaro averages include t=0 through the selected t and average distributions, not sampled occupations or their variance. The 1/4 mixing test uses worst initial states and distinguishes an unobserved finite crossing from periodic nonconvergence. No empirical MCMC error, universal effective sample size, continuous-time rate, physical heat or research-general convergence theorem is inferred.'};
}
