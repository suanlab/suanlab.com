/** Fixed finite spectrum in canonical equilibrium, with independent copies.
 * Energies:0 (g0 states), kB*gapTemperature (g1 states). Temperature0 denotes
 * the equilibrium limit, not a finite-time cooling process. SI throughout.
 */
import {binaryProduct,binaryRatioObservation,binaryMultiply,binarySum,binaryNegate,type BinaryValue,type BinaryObservation} from './learning-binary-products';
export const boltzmannConstant = 1.380649e-23; // binary64 approximation of the exact SI value, J/K
export interface GappedSpectrumParameters {
  copies:number;
  groundDegeneracy:number;
  upperDegeneracy:number;
  gapTemperature:number;
  temperature:number;
}
export interface SpectrumQuantity extends BinaryObservation {
  /** A further scale for negative logarithms; statuses distinguish unavailable levels. */
  logNegativeLogMagnitude:number|null;
  logMagnitudeStatus:'finite'|'not-applicable'|'outside-number-range';
  logNegativeLogMagnitudeStatus:'finite'|'not-applicable'|'outside-number-range';
}
const zero=():SpectrumQuantity=>({value:0,sign:0,logMagnitude:null,logNegativeLogMagnitude:null,logMagnitudeStatus:'not-applicable',logNegativeLogMagnitudeStatus:'not-applicable'});
function fromLog(log:number):SpectrumQuantity {
  if(!Number.isFinite(log))throw new RangeError('Expected a finite logarithm');
  const linear=Math.exp(log);
  return{value:linear===0||!Number.isFinite(linear)?null:linear,sign:1,logMagnitude:log,
    logNegativeLogMagnitude:log<0?Math.log(-log):null,logMagnitudeStatus:'finite',logNegativeLogMagnitudeStatus:log<0?'finite':'not-applicable'};
}
/** x is too large for binary64; log quantities are -x plus much smaller terms.
 * ln(-ln(quantity)) rounds to ln(x) in this branch. This is not physical zero.
 */
function extreme(logX:number):SpectrumQuantity {
  return{value:null,sign:1,logMagnitude:null,logNegativeLogMagnitude:logX,logMagnitudeStatus:'outside-number-range',logNegativeLogMagnitudeStatus:'finite'};
}
const positive=(v:number):SpectrumQuantity=>v===0?zero():fromLog(Math.log(v));
const one=binaryProduct(1);
const scaleRaw=(q:BinaryValue|null,...factors:number[])=>q===null?null:binaryMultiply(q,binaryProduct(...factors));
function project(log:number,raw:BinaryValue|null,denominator:BinaryValue=one):SpectrumQuantity {
  const result=fromLog(log);
  return raw===null?result:{...result,value:binaryRatioObservation(raw,denominator).value};
}
function rawProbability(log:number):BinaryValue|null {
  if(log < -2000)return null; // No non-residual observation can be range-rescued here.
  if(log >= -700)return binaryProduct(Math.exp(log));
  const root=Math.exp(log/4);return binaryProduct(root,root,root,root);
}
function addLogs(a:number,b:number){
  if(a===-Infinity)return b;if(b===-Infinity)return a;
  const top=Math.max(a,b);return top+Math.log1p(Math.exp(Math.min(a,b)-top));
}
function softplus(y:number){return y>0?y+Math.log1p(Math.exp(-y)):Math.log1p(Math.exp(y));}
function logSoftplus(y:number){return y < -37?y:Math.log(Math.log1p(Math.exp(y)));}
function validate(p:GappedSpectrumParameters){
  for(const[key,max]of[['copies',64],['groundDegeneracy',16],['upperDegeneracy',16]]as const){const v=p[key];if(!Number.isInteger(v)||v<1||v>max)throw new RangeError(`Invalid ${key}`);}
  for(const[key,max]of[['gapTemperature',1000],['temperature',800]]as const){const v=p[key];if(!Number.isFinite(v)||v<0||v>max)throw new RangeError(`Invalid ${key}`);}
}
export function gappedSpectrum(p:GappedSpectrumParameters){
  validate(p);
  const {copies:n,groundDegeneracy:g0,upperDegeneracy:g1,gapTemperature:gap,temperature:T}=p;
  const logScale=Math.log(n)+Math.log(boltzmannConstant),effectiveGroundStates=gap===0?g0+g1:g0;
  const residualNorm=Math.log(effectiveGroundStates);
  const residual=residualNorm===0?zero():fromLog(logScale+Math.log(residualNorm));
  const gapEnergy=gap===0?zero():project(Math.log(boltzmannConstant)+Math.log(gap),binaryProduct(boltzmannConstant,gap));
  const common={parameters:{...p},mode:T===0?'zero-temperature-limit' as const:'positive-temperature' as const,
    spectrum:gap===0?'exact-zero-gap' as const:'positive-gap' as const,
    effectiveGroundDegeneracy:effectiveGroundStates,
    groundStateCount:(BigInt(effectiveGroundStates)**BigInt(n)).toString(),
    totalStateCount:(BigInt(g0+g1)**BigInt(n)).toString(),gapEnergy,
    residualEntropy:residual,residualEntropyPerCopyInKB:positive(residualNorm),
    positiveGapLowTemperatureEntropy:g0===1?zero():fromLog(logScale+Math.log(Math.log(g0))),
    zeroGapThenLowTemperatureEntropy:fromLog(logScale+Math.log(Math.log(g0+g1))),
  };
  if(gap===0||T===0){
    const upper=gap===0?g1/(g0+g1):0,ground=gap===0?g0/(g0+g1):1;
    return{...common,gapToTemperatureRatio:T===0?null:{...binaryRatioObservation(binaryProduct(0),binaryProduct(T))},
      upperGroupProbability:positive(upper),groundGroupProbability:positive(ground),
      logPartitionPerCopy:positive(Math.log(effectiveGroundStates)),partitionFunctionPerCopy:effectiveGroundStates,
      meanEnergy:zero(),meanEnergyPerCopyInKB:zero(),energyVariance:zero(),
      entropy:residual,entropyPerCopyInKB:positive(residualNorm),excitationEntropy:zero(),excitationEntropyPerCopyInKB:zero(),
      heatCapacity:zero(),heatCapacityPerCopyInKB:zero(),entropyTemperatureDerivative:T===0?null:zero()};
  }
  const ratio=binaryRatioObservation(binaryProduct(gap),binaryProduct(T));
  const logX=ratio.logMagnitude!;
  if(ratio.value===null&&logX>0){
    const tail=extreme(logX),norm=Math.log(g0);
    return{...common,gapToTemperatureRatio:ratio,upperGroupProbability:tail,groundGroupProbability:{value:1,sign:1,logMagnitude:null,logNegativeLogMagnitude:null,logMagnitudeStatus:'outside-number-range' as const,logNegativeLogMagnitudeStatus:'outside-number-range' as const},
      logPartitionPerCopy:norm===0?tail:positive(norm),partitionFunctionPerCopy:g0,
      meanEnergy:tail,meanEnergyPerCopyInKB:tail,energyVariance:tail,
      entropy:norm===0?tail:residual,entropyPerCopyInKB:norm===0?tail:positive(norm),
      excitationEntropy:tail,excitationEntropyPerCopyInKB:tail,
      heatCapacity:tail,heatCapacityPerCopyInKB:tail,entropyTemperatureDerivative:tail};
  }
  // If x is below the number range, exp(-x) rounds to1; retain ln(x) separately.
  const x=ratio.value??0,y=Math.log(g1/g0)-x,normalizer=softplus(y);
  const logQ=y-normalizer,logP=-normalizer;
  const logCapacity=2*logX+logQ+logP;
  const logCorrection=logSoftplus(y);
  const groundProbability:SpectrumQuantity={...fromLog(logP),logMagnitude:normalizer===0?null:logP,logMagnitudeStatus:normalizer===0?'outside-number-range':'finite',logNegativeLogMagnitude:logCorrection,logNegativeLogMagnitudeStatus:'finite'};
  const logExcitationEntropy=addLogs(logCorrection,logX+logQ);
  const groundNorm=Math.log(g0),groundLog=groundNorm===0?-Infinity:Math.log(groundNorm);
  const logEntropyNorm=addLogs(groundLog,logExcitationEntropy);
  const logPartitionNorm=addLogs(groundLog,logCorrection);
  const rawQ=rawProbability(logQ),pLinear=Math.exp(logP);
  const softplusPerQ=y < -37?1:normalizer/Math.exp(logQ);
  const rawExc=rawQ===null?null:binaryMultiply(rawQ,binarySum([binaryProduct(gap),binaryProduct(T,softplusPerQ)]));
  const rawEntropy=rawExc===null?null:binarySum([binaryProduct(groundNorm,T),rawExc]);
  const rawCapacity=scaleRaw(rawQ,gap,gap,pLinear),temperatureSquare=binaryProduct(T,T);
  const rawPartition=rawQ===null?null:binarySum([binaryProduct(groundNorm),scaleRaw(rawQ,softplusPerQ)!]);
  if(rawQ!==null)groundProbability.value=binaryRatioObservation(binarySum([one,binaryNegate(rawQ)]),one).value;
  return{...common,gapToTemperatureRatio:ratio,
    upperGroupProbability:project(logQ,rawQ),groundGroupProbability:groundProbability,
    logPartitionPerCopy:project(logPartitionNorm,rawPartition),partitionFunctionPerCopy:g0+g1*Math.exp(-x),
    meanEnergy:project(logScale+Math.log(gap)+logQ,scaleRaw(rawQ,n,boltzmannConstant,gap)),meanEnergyPerCopyInKB:project(Math.log(gap)+logQ,scaleRaw(rawQ,gap)),
    energyVariance:project(Math.log(n)+2*(Math.log(boltzmannConstant)+Math.log(gap))+logQ+logP,scaleRaw(rawQ,n,boltzmannConstant,boltzmannConstant,gap,gap,pLinear)),
    entropy:project(logScale+logEntropyNorm,scaleRaw(rawEntropy,n,boltzmannConstant),binaryProduct(T)),entropyPerCopyInKB:project(logEntropyNorm,rawEntropy,binaryProduct(T)),
    excitationEntropy:project(logScale+logExcitationEntropy,scaleRaw(rawExc,n,boltzmannConstant),binaryProduct(T)),excitationEntropyPerCopyInKB:project(logExcitationEntropy,rawExc,binaryProduct(T)),
    heatCapacity:project(logScale+logCapacity,scaleRaw(rawCapacity,n,boltzmannConstant),temperatureSquare),heatCapacityPerCopyInKB:project(logCapacity,rawCapacity,temperatureSquare),
    entropyTemperatureDerivative:project(logScale+logCapacity-Math.log(T),scaleRaw(rawCapacity,n,boltzmannConstant),binaryProduct(T,T,T))};
}
