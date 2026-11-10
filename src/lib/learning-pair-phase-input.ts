/** Reader-safe preparation contract. No numerical engine imports. */
export interface PairPhaseParameters{omega1Numerator:number;omega1Denominator:number;omega2Numerator:number;omega2Denominator:number;couplingNumerator:number;couplingDenominator:number;initialHalfA:number;initialHalfB:number;meanPhaseNumerator:number;meanPhaseDenominator:number;timeNumerator:number;timeDenominator:number;sampleSteps:number;}
export const pairPhaseParameterKeys=['omega1Numerator','omega1Denominator','omega2Numerator','omega2Denominator','couplingNumerator','couplingDenominator','initialHalfA','initialHalfB','meanPhaseNumerator','meanPhaseDenominator','timeNumerator','timeDenominator','sampleSteps'] as const;
export function isPairPhaseParameters(value:unknown):value is PairPhaseParameters{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==pairPhaseParameterKeys.length)return false;
 const p=value as Record<string,number>,signed=['omega1Numerator','omega2Numerator','initialHalfA','initialHalfB','meanPhaseNumerator'];
 return pairPhaseParameterKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&p[k]>=(signed.includes(k)?-1000000000:['couplingNumerator','timeNumerator'].includes(k)?0:1)&&p[k]<=(k==='sampleSteps'?64:1000000000))&&(p.initialHalfA!==0||p.initialHalfB!==0)&&Math.abs(p.omega1Numerator)<=8*p.omega1Denominator&&Math.abs(p.omega2Numerator)<=8*p.omega2Denominator&&p.couplingNumerator<=16*p.couplingDenominator&&Math.abs(p.meanPhaseNumerator)<=8*p.meanPhaseDenominator&&p.timeNumerator<=4*p.timeDenominator;
}
