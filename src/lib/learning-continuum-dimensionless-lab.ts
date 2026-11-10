import { pecletAt, damkohlerAt, type PecletParameters, type DamkohlerParameters } from './learning-continuum-dimensionless';
export const continuumDimensionlessKinds = ['advection-diffusion-peclet-boundary-layer', 'reaction-diffusion-damkohler-penetration'] as const;
export type ContinuumDimensionlessKind = typeof continuumDimensionlessKinds[number];
export const isContinuumDimensionlessKind = (id: string): id is ContinuumDimensionlessKind => (continuumDimensionlessKinds as readonly string[]).includes(id);
export const pecletBounds = [[-2,2],[.0001,4],[.25,4],[0,4],[0,4],[.25,4],[0,1],[.25,2]];
export const damkohlerBounds = [[.0001,4],[.25,4],[0,128],[0,4],[.25,4],[0,1],[.25,2],[.25,2]];
function validate(values: number[], bounds: number[][]) {
  if (values.length !== bounds.length) throw new RangeError('Wrong dimensionless control count');
  for (let i=0;i<bounds.length;i++) if (!Number.isFinite(values[i]) || values[i] < bounds[i][0] || values[i] > bounds[i][1]) throw new RangeError(`Invalid dimensionless control ${i}`);
}
const unique = (values: number[]) => [...new Set(values.filter(x => Number.isFinite(x) && x >= 0 && x <= 1))].sort((a,b) => a-b);
/** Uniform points plus both boundary layers, exact boundaries and the actual query. */
function positions(xi: number, layer: number) {
  return unique([0,1,xi,...Array.from({length:161},(_,i)=>i/160),
    ...Array.from({length:81},(_,i)=>Math.min(1,layer*12*i/80)),
    ...Array.from({length:81},(_,i)=>1-Math.min(1,layer*12*i/80))]);
}
function logSamples(selected: number, maximum: number) {
  const start = Math.min(-24, selected > 0 ? Math.log10(selected) : -24);
  return [...new Set([...Array.from({length:61},(_,i)=>10**(start+(Math.log10(maximum)-start)*i/60)),selected])]
    .filter(x=>x>0&&x<=maximum).sort((a,b)=>a-b);
}
function peSupported(p: PecletParameters) {
  return p.length>=.25&&p.length<=4&&p.diffusivity>=.0001&&p.diffusivity<=4&&Math.abs(p.velocity)<=2&&p.area>=.25&&p.area<=4;
}
function daSupported(p: DamkohlerParameters) {
  return p.length>=.25&&p.length<=4&&p.diffusivity>=.0001&&p.diffusivity<=4&&p.reactionRate>=0&&p.reactionRate<=128&&p.area>=.25&&p.area<=4;
}
export function pecletLab(v: number[]) {
  validate(v,pecletBounds);
  const parameters: PecletParameters = {velocity:v[0],diffusivity:v[1],length:v[2],leftConcentration:v[3],rightConcentration:v[4],area:v[5]};
  const xi=v[6],scale=v[7],query=pecletAt(parameters,xi);
  const coordinates=positions(xi,parameters.velocity===0?1:parameters.diffusivity/(Math.abs(parameters.velocity)*parameters.length));
  const comparisonParameters={...parameters,length:scale*parameters.length,velocity:parameters.velocity/scale,area:scale*scale*parameters.area};
  const arithmeticSupported=parameters.velocity===0||comparisonParameters.velocity!==0&&Math.abs(Math.log(Math.abs(comparisonParameters.velocity))-(Math.log(Math.abs(parameters.velocity))-Math.log(scale)))<1e-10;
  const comparisonStatus=!arithmeticSupported?'nonzero-coefficient-outside-arithmetic-range' as const:peSupported(comparisonParameters)?'supported' as const:'outside-declared-parameter-range' as const;
  const comparisonSupported=comparisonStatus==='supported';
  const comparison=comparisonSupported?pecletAt(comparisonParameters,xi):null;
  const spatial=coordinates.map(xi=>({xi,value:pecletAt(parameters,xi),comparison:comparisonSupported?pecletAt(comparisonParameters,xi):null}));
  const corrections=logSamples(Math.abs(parameters.velocity),.01).map(velocity=>({logVelocity:Math.log10(velocity),value:pecletAt({...parameters,velocity:parameters.velocity<0?-velocity:velocity},xi)}));
  return {parameters,xi,scale,query,spatial,corrections,comparison,comparisonParameters,comparisonStatus,comparisonDiffusivityScale:1};
}
export function damkohlerLab(v: number[]) {
  validate(v,damkohlerBounds);
  const parameters: DamkohlerParameters = {diffusivity:v[0],length:v[1],reactionRate:v[2],surfaceConcentration:v[3],area:v[4]};
  const xi=v[5],lengthScale=v[6],diffusivityScale=v[7],query=damkohlerAt(parameters,xi);
  const layer=parameters.reactionRate===0?1:1/Math.exp(query.thiele.logMagnitude!);
  const comparisonParameters={...parameters,length:lengthScale*parameters.length,diffusivity:diffusivityScale*parameters.diffusivity,reactionRate:diffusivityScale/(lengthScale*lengthScale)*parameters.reactionRate,area:lengthScale*lengthScale*parameters.area};
  const arithmeticSupported=parameters.reactionRate===0||comparisonParameters.reactionRate!==0&&Math.abs(Math.log(comparisonParameters.reactionRate)-(Math.log(parameters.reactionRate)+Math.log(diffusivityScale)-2*Math.log(lengthScale)))<1e-10;
  const comparisonStatus=!arithmeticSupported?'nonzero-coefficient-outside-arithmetic-range' as const:daSupported(comparisonParameters)?'supported' as const:'outside-declared-parameter-range' as const;
  const comparisonSupported=comparisonStatus==='supported',comparison=comparisonSupported?damkohlerAt(comparisonParameters,xi):null;
  const spatial=positions(xi,layer).map(xi=>({xi,value:damkohlerAt(parameters,xi),comparison:comparisonSupported?damkohlerAt(comparisonParameters,xi):null}));
  const depletion=logSamples(parameters.reactionRate,.01).map(reactionRate=>({logRate:Math.log10(reactionRate),value:damkohlerAt({...parameters,reactionRate},xi)}));
  const surface=damkohlerAt(parameters,0),sealed=damkohlerAt(parameters,1);
  const response=Array.from({length:81},(_,i)=>{const logRate=-8+(8+Math.log10(128))*i/80,reactionRate=i===80?128:10**logRate;return{logRate,value:damkohlerAt({...parameters,reactionRate},xi)};});
  return {parameters,xi,lengthScale,diffusivityScale,query,surface,sealed,spatial,depletion,response,comparison,comparisonParameters,comparisonStatus};
}
