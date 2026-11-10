/** Per-cycle two-reservoir budgets. This is not an internal piston trajectory.
 * Negative relative generation is an explicitly counterfactual comparison.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negative,
  binaryMultiply as multiply, binaryRatioObservation as observe } from './learning-binary-products';
export interface ReservoirCycleParameters {
  hotTemperature: number;
  coldTemperature: number;
  heatHot: number;
  relativeEntropyGeneration: number;
}
function bound(v:number,low:number,high:number,name:string){
  if(!Number.isFinite(v)||v<low||v>high)throw new RangeError(`Invalid ${name}`);
}
export function reservoirCycle(p:ReservoirCycleParameters){
  bound(p.hotTemperature,250,800,'hot temperature');bound(p.coldTemperature,250,800,'cold temperature');
  bound(p.heatHot,0,4000,'hot heat');bound(p.relativeEntropyGeneration,-1,4,'relative entropy generation');
  if(p.hotTemperature<p.coldTemperature)throw new RangeError('Hot reservoir is colder than cold reservoir');
  const th=p.hotTemperature,tc=p.coldTemperature,qh=p.heatHot,sigma=p.relativeEntropyGeneration;
  const denominator=product(th),onePlusSigma=sum([product(1),product(sigma)]);
  const temperatureDifference=sum([product(th),negative(product(tc))]);
  const workBracket=sum([temperatureDifference,negative(product(tc,sigma))]);
  const work=observe(multiply(product(qh),workBracket),denominator);
  const generation=observe(product(qh,sigma),denominator);
  const admissible=generation.sign>=0;
  const regime=qh===0?'no-transfer' as const:!admissible?'counterfactual' as const:work.sign>0?'heat-engine' as const:work.sign<0?'work-driven-dissipation' as const:'heat-transfer' as const;
  const fraction=qh===0?null:observe(workBracket,denominator);
  return{
    parameters:{...p},regime,secondLawAdmissible:admissible,
    heatHotIntoDevice:observe(product(qh),product(1)),
    heatColdIntoDevice:observe(negative(multiply(product(qh,tc),onePlusSigma)),denominator),
    workByDevice:work,entropyGeneration:generation,
    hotReservoirEntropyChange:observe(product(-qh),denominator),
    coldReservoirEntropyChange:observe(multiply(product(qh),onePlusSigma),denominator),
    reversibleWork:observe(multiply(product(qh),temperatureDifference),denominator),
    workDeficitFromReversible:observe(product(qh,tc,sigma),denominator),
    carnotTemperatureBound:observe(temperatureDifference,denominator),
    workFraction:fraction,engineEfficiency:regime==='heat-engine'?fraction:null,
    cycleEnergyChange:0,cycleEntropyChange:0,
  };
}
