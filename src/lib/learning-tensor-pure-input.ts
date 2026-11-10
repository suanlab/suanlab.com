/** Arbitrary bounded complex coefficient matrix on C^dA tensor C^dB. */
export const tensorPureKeys=['dimensionA','dimensionB',...Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>[`stateReal${i}${j}`,`stateImag${i}${j}`])).flat(2),...['a','b'].flatMap(s=>Array.from({length:4},(_,i)=>[`${s}ProbeReal${i}`,`${s}ProbeImag${i}`]).flat())]as readonly string[];
export type TensorPureParameters=Record<string,number>;
export function isTensorPureParameters(x:unknown):x is TensorPureParameters{
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==50)return false;const p=x as TensorPureParameters;
 if(!tensorPureKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000)||p.dimensionA<1||p.dimensionA>4||p.dimensionB<1||p.dimensionB>4)return false;
 if(!Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i<p.dimensionA&&j<p.dimensionB||p[`stateReal${i}${j}`]===0&&p[`stateImag${i}${j}`]===0)).flat().every(Boolean))return false;
 if(!Array.from({length:p.dimensionA},(_,i)=>Array.from({length:p.dimensionB},(_,j)=>p[`stateReal${i}${j}`]!==0||p[`stateImag${i}${j}`]!==0)).flat().some(Boolean))return false;
 return ['a','b'].every(s=>{const d=p[s==='a'?'dimensionA':'dimensionB'];return Array.from({length:d},(_,i)=>p[`${s}ProbeReal${i}`]!==0||p[`${s}ProbeImag${i}`]!==0).some(Boolean)&&Array.from({length:4-d},(_,i)=>i+d).every(i=>p[`${s}ProbeReal${i}`]===0&&p[`${s}ProbeImag${i}`]===0);});
}
export function snapshotTensorPureParameters(x:TensorPureParameters):Readonly<TensorPureParameters>{if(!isTensorPureParameters(x))throw new RangeError('50 owned bounded integers; dimensions1..4,nonzero global/probe vectors and zero inactive coordinates');return Object.freeze(Object.fromEntries(tensorPureKeys.map(k=>[k,x[k]+0])));}
