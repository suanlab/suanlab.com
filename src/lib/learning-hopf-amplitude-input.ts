/** Reader-safe preparation contract. No numerical engine imports. */
export interface HopfAmplitudeParameters {
 muNumerator:number;muDenominator:number;gNumerator:number;gDenominator:number;
 shearNumerator:number;shearDenominator:number;omegaNumerator:number;omegaDenominator:number;
 xNumerator:number;xDenominator:number;yNumerator:number;yDenominator:number;
 timeNumerator:number;timeDenominator:number;sampleSteps:number;
}
export const hopfAmplitudeKeys=['muNumerator','muDenominator','gNumerator','gDenominator','shearNumerator','shearDenominator','omegaNumerator','omegaDenominator','xNumerator','xDenominator','yNumerator','yDenominator','timeNumerator','timeDenominator','sampleSteps'] as const;
export function isHopfAmplitudeParameters(value:unknown):value is HopfAmplitudeParameters {
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==hopfAmplitudeKeys.length)return false;
 const p=value as Record<string,number>,signed=['muNumerator','shearNumerator','omegaNumerator','xNumerator','yNumerator'];
 if(!hopfAmplitudeKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&p[k]>=(signed.includes(k)?-1000000000:k==='timeNumerator'?0:1)&&p[k]<=(k==='sampleSteps'?64:1000000000)))return false;
 return Math.abs(p.muNumerator)<=8*p.muDenominator&&p.gNumerator<=8*p.gDenominator&&Math.abs(p.shearNumerator)<=8*p.shearDenominator&&Math.abs(p.omegaNumerator)<=8*p.omegaDenominator&&Math.abs(p.xNumerator)<=4*p.xDenominator&&Math.abs(p.yNumerator)<=4*p.yDenominator&&p.timeNumerator<=4*p.timeDenominator;
}
