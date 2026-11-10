/** Three micro states, blocks A={0,1},B={2}; discrete updates, no physical clock. */
export interface MarkovCoarseParameters{
 weight01:number;weight02:number;weight10:number;weight12:number;weight20:number;weight21:number;transitionDenominator:number;
 initialWeights:readonly number[];liftNumerator:number;liftDenominator:number;steps:number;memoryDepth:number;
}
export const markovCoarseKeys=['weight01','weight02','weight10','weight12','weight20','weight21','transitionDenominator','initialWeights','liftNumerator','liftDenominator','steps','memoryDepth']as const;
const integer=(x:unknown,min=0,max=1000000):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
export function isMarkovCoarseParameters(x:unknown):x is MarkovCoarseParameters{
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==markovCoarseKeys.length)return false;
 const p=x as Record<string,unknown>;
 if(!markovCoarseKeys.every(k=>Object.hasOwn(p,k)))return false;
 for(const k of markovCoarseKeys)if(k!=='initialWeights'&&!integer(p[k],k==='transitionDenominator'||k==='liftDenominator'?1:0,k==='steps'||k==='memoryDepth'?32:1000000))return false;
 const w=p.initialWeights;if(!Array.isArray(w)||Object.getPrototypeOf(w)!==Array.prototype||w.length!==3||Reflect.ownKeys(w).length!==4||![0,1,2].every(i=>Object.hasOwn(w,i)&&integer(w[i]))||w.every(n=>n===0))return false;
 const v=x as MarkovCoarseParameters;return v.weight01+v.weight02<=v.transitionDenominator&&v.weight10+v.weight12<=v.transitionDenominator&&v.weight20+v.weight21<=v.transitionDenominator&&v.liftNumerator<=v.liftDenominator;
}
export function snapshotMarkovCoarseParameters(x:MarkovCoarseParameters):Readonly<MarkovCoarseParameters>{
 if(!isMarkovCoarseParameters(x))throw new RangeError('Twelve owned fields; nonnegative integer weights<=1e6,positive denominators<=1e6,outgoing sums<=denominator,three nonzero-total initial weights,0<=alpha<=1,steps and memory depth0..32');
 return Object.freeze({...x,initialWeights:Object.freeze(x.initialWeights.map(n=>n+0))});
}
