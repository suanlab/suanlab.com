/** One finite spin irrep; unused raw coordinates must be zero, never silently clipped. */
export const spinLadderKeys=['twoJ','hbarNumerator','hbarDenominator',...Array.from({length:9},(_,i)=>[`stateReal${i}`,`stateImag${i}`]).flat()] as readonly string[];
export type SpinLadderParameters={twoJ:number;hbarNumerator:number;hbarDenominator:number}&Record<`state${'Real'|'Imag'}${0|1|2|3|4|5|6|7|8}`,number>;
export function isSpinLadderParameters(x:unknown):x is SpinLadderParameters{
 if(!x||typeof x!=='object'||Object.getPrototypeOf(x)!==Object.prototype||Reflect.ownKeys(x).length!==21)return false;const p=x as Record<string,number>;
 return spinLadderKeys.every(k=>Object.hasOwn(p,k)&&Number.isSafeInteger(p[k])&&Math.abs(p[k])<=1000000)&&p.twoJ>=0&&p.twoJ<=8&&p.hbarNumerator>0&&p.hbarDenominator>0&&Array.from({length:9},(_,i)=>i).every(i=>i<=p.twoJ||p[`stateReal${i}`]===0&&p[`stateImag${i}`]===0)&&Array.from({length:p.twoJ+1},(_,i)=>i).some(i=>p[`stateReal${i}`]!==0||p[`stateImag${i}`]!==0);
}
export function snapshotSpinLadderParameters(x:SpinLadderParameters):Readonly<SpinLadderParameters>{if(!isSpinLadderParameters(x))throw new RangeError('Twenty-one owned integers of magnitude<=1e6; twoJ0..8,positive hbar ratio,nonzero active state and zero inactive coordinates');return Object.freeze(Object.fromEntries(spinLadderKeys.map(k=>[k,(x as unknown as Record<string,number>)[k]+0]))) as SpinLadderParameters;}
