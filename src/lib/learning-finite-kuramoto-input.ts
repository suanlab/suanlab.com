/** Reader-safe exact preparations; finite numerical grid and continuum comparison are separate. */
export interface FiniteKuramotoParameters{
 omegaNumerators:readonly number[];phasePiNumerators:readonly number[];omegaDenominator:number;phasePiDenominator:number;
 couplingNumerator:number;couplingDenominator:number;timeNumerator:number;timeDenominator:number;integrationSteps:number;sampleSteps:number;referenceWidthNumerator:number;referenceWidthDenominator:number;
}
export const finiteKuramotoKeys=['omegaNumerators','phasePiNumerators','omegaDenominator','phasePiDenominator','couplingNumerator','couplingDenominator','timeNumerator','timeDenominator','integrationSteps','sampleSteps','referenceWidthNumerator','referenceWidthDenominator'] as const;
const integer=(v:unknown,min=1,max=1000000000):v is number=>typeof v==='number'&&Number.isSafeInteger(v)&&v>=min&&v<=max;
const vector=(x:unknown):x is number[]=>Array.isArray(x)&&Object.getPrototypeOf(x)===Array.prototype&&x.length>=1&&x.length<=32&&Reflect.ownKeys(x).length===x.length+1&&Array.from({length:x.length},(_,i)=>Object.hasOwn(x,i)&&integer(x[i],-1000000000)).every(Boolean);
export function isFiniteKuramotoParameters(value:unknown):value is FiniteKuramotoParameters{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==finiteKuramotoKeys.length)return false;
 const p=value as Record<string,unknown>;if(!finiteKuramotoKeys.every(k=>Object.hasOwn(p,k)&&(k==='omegaNumerators'||k==='phasePiNumerators'?vector(p[k]):integer(p[k],k==='couplingNumerator'||k==='timeNumerator'?0:1,k==='integrationSteps'?1024:k==='sampleSteps'?64:1000000000))))return false;
 const v=p as unknown as FiniteKuramotoParameters;return v.omegaNumerators.length===v.phasePiNumerators.length&&v.omegaNumerators.every(n=>Math.abs(n)<=8*v.omegaDenominator)&&v.phasePiNumerators.every(n=>Math.abs(n)<=4*v.phasePiDenominator)&&v.couplingNumerator<=16*v.couplingDenominator&&v.timeNumerator<=4*v.timeDenominator&&v.referenceWidthNumerator<=8*v.referenceWidthDenominator&&v.sampleSteps<=v.integrationSteps;
}
export function snapshotFiniteKuramotoParameters(value:FiniteKuramotoParameters):Readonly<FiniteKuramotoParameters>{
 if(!isFiniteKuramotoParameters(value))throw new RangeError('Twelve owned fields; dense equal-length integer arrays,N1..32,|omega|<=8,|phase/pi|<=4,K0..16,T0..4s,width0..8positive,steps1..1024,records1..64<=steps,integer magnitudes<=1e9');
 return Object.freeze({...value,omegaNumerators:Object.freeze(value.omegaNumerators.map(n=>n+0)),phasePiNumerators:Object.freeze(value.phasePiNumerators.map(n=>n+0))});
}
