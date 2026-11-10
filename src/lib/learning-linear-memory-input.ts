/** Stable/marginal two-component linear family. States dimensionless, time in s.
 * dx/dt=-a*x+b*y, epsilon*dy/dt=c*x-d*y; a,d>0, ad-bc>=0. */
export interface LinearMemoryParameters{
 aNumerator:number;aDenominator:number;bNumerator:number;bDenominator:number;cNumerator:number;cDenominator:number;dNumerator:number;dDenominator:number;
 epsilonNumerator:number;epsilonDenominator:number;xNumerator:number;xDenominator:number;yNumerator:number;yDenominator:number;timeNumerator:number;timeDenominator:number;sampleSteps:number;
}
export const linearMemoryKeys=['aNumerator','aDenominator','bNumerator','bDenominator','cNumerator','cDenominator','dNumerator','dDenominator','epsilonNumerator','epsilonDenominator','xNumerator','xDenominator','yNumerator','yDenominator','timeNumerator','timeDenominator','sampleSteps'] as const;
export function isLinearMemoryParameters(x:unknown):x is LinearMemoryParameters{
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==linearMemoryKeys.length)return false;const p=x as Record<string,number>,signed=['bNumerator','cNumerator','xNumerator','yNumerator'];
 if(!linearMemoryKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&p[k]>=(signed.includes(k)?-1000000000:k==='timeNumerator'?0:1)&&p[k]<=(k==='sampleSteps'?64:1000000000)))return false;
 for(const k of ['a','b','c','d'])if(Math.abs(p[k+'Numerator'])>8*p[k+'Denominator'])return false;
 for(const k of ['epsilon','x','y','time'])if(Math.abs(p[k+'Numerator'])>4*p[k+'Denominator'])return false;
 const n=(k:string)=>BigInt(p[k+'Numerator']),d=(k:string)=>BigInt(p[k+'Denominator']);return n('a')*n('d')*d('b')*d('c')>=n('b')*n('c')*d('a')*d('d');
}
export function snapshotLinearMemoryParameters(x:LinearMemoryParameters):Readonly<LinearMemoryParameters>{if(!isLinearMemoryParameters(x))throw new RangeError('Seventeen owned integers; positive denominators,a,d,epsilon; |a,b,c,d|<=8,|x0,y0|<=4,epsilon<=4,time0..4s,records1..64,ad-bc>=0,magnitudes<=1e9');return Object.freeze(Object.fromEntries(linearMemoryKeys.map(k=>[k,x[k]+0]))) as unknown as Readonly<LinearMemoryParameters>;}
