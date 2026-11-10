/** Small, fixed drawing references only. No current-input or high-precision engine. */
export const realTimeReferenceSquare = [[0,0],[1,0],[1,1],[0,1],[0,0]] as const;
const sample = (start: number, end: number, count: number, value: (x: number) => number) =>
  Array.from({length: count}, (_, i) => {const x=start+(end-start)*i/(count-1);return {x,y:value(x)};});
export const realTimeReferencePhases = [-1,-.5,0,.5,1].map(x => ({
  x, linear:4*x, quadratic:2*x*x,
}));
export const realTimeReferenceAction = sample(-3,1,81,q=>-q*q/2-q-.25);
export const realTimeReferenceDamping = [-2,-1,0,1,2].map(x=>({x,y:Math.exp(-x*x/2)}));
export const realTimeReferenceDensities = sample(-4,5,121,x=>Math.exp(-x*x/2)/Math.sqrt(2*Math.PI))
  .map(p=>({...p,propagated:Math.exp(-((p.x-1)**2)/(2*1.25))/Math.sqrt(2*Math.PI*1.25)}));
export const realTimeReferenceVariances = [
  {hbar:'1/2',fixed:17/16,scaled:5/8,fixedText:'17/16',scaledText:'5/8'},
  {hbar:'1/8',fixed:257/256,scaled:5/32,fixedText:'257/256',scaledText:'5/32'},
] as const;
export const realTimeReferenceFolds = [1,0,-1].map(control=>({
  control,points:sample(-2,2,81,x=>x*x*x/3-control*x),
  stationary:control>0?[-1,1]:control===0?[0]:[],
}));
