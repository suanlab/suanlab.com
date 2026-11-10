/** Stationary unit-variance Gaussian AR(1), including separately labeled
 * constant/alternating degenerate endpoints. Finite mean uncertainty uses
 * independent innovations, not cancellation of rounded autocovariances.
 */
import { binaryProduct, binaryMultiply, binarySum, binaryNegate, type BinaryValue } from './learning-binary-products';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';

export interface AR1AverageParameters { sampleCount: number; correlationRho: number }
export interface AR1AverageOptions { minimumPrecision?: number; maximumPrecision?: number }
interface I { lower: bigint; upper: bigint }
const Z=BigInt(0),O=BigInt(1),TWO=BigInt(2),one=binaryProduct(1);
const dyadic=(coefficient:bigint,exponent=0):BinaryValue=>({coefficient,exponent});
const floor=(n:bigint,d:bigint)=>n>=Z?n/d:-((-n+d-O)/d);
const ceil=(n:bigint,d:bigint)=>-floor(-n,d);
function normalized(v:BinaryValue):BinaryValue {
  if(v.coefficient===Z)return dyadic(Z);
  let {coefficient,exponent}=v;
  while(coefficient%TWO===Z){coefficient/=TWO;exponent++;}
  return {coefficient,exponent};
}
function squareRootFloor(n:bigint):bigint {
  if(n<Z)throw new RangeError('Nonnegative square root required');
  if(n<TWO)return n;
  let x=O<<BigInt(Math.ceil(n.toString(2).length/2));
  for(;;){const y=(x+n/x)/TWO;if(y>=x)return x;x=y;}
}
const exact=(n:BinaryValue,d=one):PartitionObservation=>partitionObservationFromRange({lower:{n,d},upper:{n,d}});
const infinity=():PartitionObservation=>({value:null,sign:1,logMagnitude:null,logResolution:'not-applicable',resolution:'mathematical-infinity',bounds:null});
const unavailable=():PartitionObservation=>({value:null,sign:1,logMagnitude:null,logResolution:'unresolved',resolution:'unresolved-at-precision-ceiling',bounds:null});
const precisions=[128,256,512,1024,2048,4096,8192];

export function ar1FiniteAverageModel(input:AR1AverageParameters,options:AR1AverageOptions={}) {
  if(!input||!Number.isInteger(input.sampleCount)||input.sampleCount<1||input.sampleCount>512||!Number.isFinite(input.correlationRho)||Math.abs(input.correlationRho)>1)
    throw new RangeError('Expected n=1..512 and stationary unit-variance AR(1) rho in [-1,1]');
  const minimum=options.minimumPrecision??128,maximum=options.maximumPrecision??8192;
  if(!precisions.includes(minimum)||!precisions.includes(maximum)||minimum>maximum)throw new RangeError('Invalid precision range');
  const parameters=Object.freeze({sampleCount:input.sampleCount,correlationRho:input.correlationRho}),n=input.sampleCount,rho=input.correlationRho,r=normalized(binaryProduct(rho));
  const innovation=normalized(binarySum([one,binaryNegate(binaryMultiply(r,r))]));
  const plus=normalized(binarySum([one,r])),minus=normalized(binarySum([one,binaryNegate(r)])),interior=Math.abs(rho)<1;
  const autocovariances:{lag:number;covariance:PartitionObservation}[]=[];
  let power=one;
  for(let lag=0;lag<n;lag++){autocovariances.push({lag,covariance:exact(power)});power=binaryMultiply(power,r);}
  const asymptoticIntegratedCorrelationTime=interior?exact(plus,minus):null;
  const asymptoticMeanVariance=interior?exact(plus,binaryMultiply(minus,binaryProduct(n))):null;
  const attempts:number[]=[];
  function calculate(P:number) {
    const S=O<<BigInt(P),fixedOne={lower:S,upper:S};
    const fixed=(v:BinaryValue):I=>{
      const shift=v.exponent+P;
      return shift>=0?{lower:v.coefficient<<BigInt(shift),upper:v.coefficient<<BigInt(shift)}:
        {lower:floor(v.coefficient,O<<BigInt(-shift)),upper:ceil(v.coefficient,O<<BigInt(-shift))};
    };
    const add=(a:I,b:I):I=>({lower:a.lower+b.lower,upper:a.upper+b.upper});
    const multiply=(a:I,b:I):I=>{
      const products=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];
      return {lower:floor(products.reduce((a,b)=>a<b?a:b),S),upper:ceil(products.reduce((a,b)=>a>b?a:b),S)};
    };
    const observe=(a:I,denominator=1):PartitionObservation=>partitionObservationFromRange({
      lower:{n:dyadic(a.lower,-P),d:binaryProduct(denominator)},upper:{n:dyadic(a.upper,-P),d:binaryProduct(denominator)},
    });
    const rf=fixed(r),lambda=fixed(innovation);
    let a:I={lower:Z,upper:Z},previousSquares:I={lower:Z,upper:Z},latestNumerator:I={lower:Z,upper:Z};
    const varianceBySampleCount:{sampleCount:number;meanVariance:PartitionObservation}[]=[];
    for(let m=1;m<=n;m++){
      a=add(fixedOne,multiply(rf,a));
      // The finite geometric sum is analytically in [0,m] for rho in [-1,1].
      a={lower:a.lower<Z?Z:a.lower,upper:a.upper>BigInt(m)*S?BigInt(m)*S:a.upper};
      const square=multiply(a,a);
      latestNumerator=add(square,multiply(lambda,previousSquares));
      varianceBySampleCount.push({sampleCount:m,meanVariance:observe(latestNumerator,m*m)});
      previousSquares=add(previousSquares,square);
    }
    const meanVariance=varianceBySampleCount[n-1].meanVariance;
    const varianceInflation=observe(latestNumerator,n),totalSumVariance=observe(latestNumerator);
    const lowRoot=squareRootFloor(latestNumerator.lower*S),highRoot=squareRootFloor(latestNumerator.upper*S);
    const standardError=partitionObservationFromRange({
      lower:{n:dyadic(lowRoot,-P),d:binaryProduct(n)},
      upper:{n:dyadic(highRoot*highRoot===latestNumerator.upper*S?highRoot:highRoot+O,-P),d:binaryProduct(n)},
    });
    const deterministicAverage=rho===-1&&n%2===0;
    const varianceEquivalentSampleCount=deterministicAverage?infinity():latestNumerator.lower>Z?partitionObservationFromRange({
      lower:{n:binaryProduct(n,n),d:dyadic(latestNumerator.upper,-P)},upper:{n:binaryProduct(n,n),d:dyadic(latestNumerator.lower,-P)},
    }):unavailable();
    const finiteToAsymptoticVarianceRatio=interior?partitionObservationFromRange({
      lower:{n:binaryMultiply(dyadic(latestNumerator.lower,-P),minus),d:binaryMultiply(binaryProduct(n),plus)},
      upper:{n:binaryMultiply(dyadic(latestNumerator.upper,-P),minus),d:binaryMultiply(binaryProduct(n),plus)},
    }):null;
    const observations=[meanVariance,totalSumVariance,varianceInflation,standardError,varianceEquivalentSampleCount,...varianceBySampleCount.map(v=>v.meanVariance),...(finiteToAsymptoticVarianceRatio?[finiteToAsymptoticVarianceRatio]:[])];
    return {meanExpectation:exact(binaryProduct(0)),meanVariance,totalSumVariance,varianceInflation,standardError,varianceEquivalentSampleCount,finiteToAsymptoticVarianceRatio,
      varianceBySampleCount,deterministicAverage,precisionBits:P,fullyResolved:observations.every(v=>v.resolution!=='unresolved-at-precision-ceiling')};
  }
  let latest:ReturnType<typeof calculate>|undefined;
  for(const P of precisions.filter(v=>v>=minimum&&v<=maximum)){attempts.push(P);latest=calculate(P);if(latest.fullyResolved)break;}
  return {scope:'stationary-unit-variance-gaussian-ar1-finite-average' as const,parameters,innovationVariance:exact(innovation),independentMeanVariance:exact(one,binaryProduct(n)),autocovariances,
    asymptoticIntegratedCorrelationTime,asymptoticMeanVariance,asymptoticStatus:interior?'mixing-interior' as const:'not-applicable-nonmixing-endpoint' as const,
    processMode:rho===1?'constant-degenerate' as const:rho===-1?'alternating-degenerate' as const:'stationary-gaussian-ar1' as const,...latest!,attempts,
    approximationContract:'Stationary centered Gaussian AR(1), marginal variance one, independent standard-normal initial value and innovations. Exact supplied binary64 rho in [-1,1], n=1..512. Autocovariances rho^lag are exact dyadics. Finite mean variance uses positive innovation decomposition with directed arithmetic, never subtraction of rounded moment displays. Standard error uses directed integer square-root bounds. Resolved observations have relative enclosure width <=2^-48. General magnitude-log projections are approximate. Unavailable at a precision ceiling is not zero. rho=+/-1 are separately labeled stationary nonmixing degenerate preparations with no ordinary summed-IAT report. Variance-equivalent sample count is 1/Var(mean), may exceed n or be infinite, and is not an independent-data count or a guarantee for other observables. Asymptotic variance tau/n requires fixed |rho|<1 and is not the exact finite variance. No fitted rho, empirical MCMC diagnostic, nonstationary training-stream guarantee or spatial correlation length is inferred.'};
}
