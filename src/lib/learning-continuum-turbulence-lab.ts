import { reynoldsEnsembleAt, reynoldsInstantaneousAt, reynoldsEnsembleBudgets, type ReynoldsEnsembleParameters } from './learning-continuum-turbulence';
import { gaussianBurgersAt, gaussianBurgersBudgets, type GaussianBurgersParameters } from './learning-continuum-gaussian-filter';
export const continuumTurbulenceKinds = ['reynolds-ensemble-stress-closure-defect', 'gaussian-filtered-burgers-subgrid-transfer'] as const;
export type ContinuumTurbulenceKind = typeof continuumTurbulenceKinds[number];
export const isContinuumTurbulenceKind = (id: string): id is ContinuumTurbulenceKind => (continuumTurbulenceKinds as readonly string[]).includes(id);
export const reynoldsBounds = [[.25,4],[.25,4],[.25,4],[.25,4],[0,2],[-2,2],[-1,1],[-2,2],[-2,2],[0,4],[0,1],[0,1],[0,1]];
export const gaussianFilterBounds = [[.25,4],[0,4],[0,2],[-2,2],[-2,2],[.25,4],[.25,4],[-2,2],[0,4],[0,1],[0,1]];
function validate(v: number[], bounds: number[][]) {
  if (v.length !== bounds.length) throw new RangeError('Wrong turbulence control count');
  for (let i=0;i<bounds.length;i++) if (!Number.isFinite(v[i]) || v[i]<bounds[i][0] || v[i]>bounds[i][1]) throw new RangeError(`Invalid turbulence control ${i}`);
}
const samples = (values: number[], max: number, min=0) => [...new Set(values.filter(x=>Number.isFinite(x)&&x>=min&&x<=max))].sort((a,b)=>a-b);
const grid = (count: number, max=1) => Array.from({length:count+1},(_,i)=>i===count?max:max*i/count);
function tinySamples(query: number) { const low=Math.min(-24,query>0?Math.log10(query):-24);return samples([...Array.from({length:33},(_,i)=>10**(low+(-2-low)*i/32)),query],.01).filter(x=>x>0); }
export function reynoldsLab(v: number[]) {
  validate(v,reynoldsBounds);
  const parameters: ReynoldsEnsembleParameters = {height:v[0],wavenumber:v[1],spanWidth:v[2],density:v[3],dynamicViscosity:v[4],bulkVelocity:v[5],meanAsymmetry:v[6],firstStreamfunctionAmplitude:v[7],secondStreamfunctionAmplitude:v[8],candidateEddyViscosity:v[9]};
  const xi=v[10],xCycles=v[11],phaseCycles=v[12],query=reynoldsEnsembleAt(parameters,xi),point=reynoldsInstantaneousAt(parameters,xi,xCycles,phaseCycles),budget=reynoldsEnsembleBudgets(parameters);
  const spatial=samples([...grid(120),.5,xi],1).map(xi=>({xi,value:reynoldsEnsembleAt(parameters,xi),point:reynoldsInstantaneousAt(parameters,xi,xCycles,phaseCycles),quiet:reynoldsEnsembleAt({...parameters,firstStreamfunctionAmplitude:0,secondStreamfunctionAmplitude:0},xi)}));
  const viscositySegments:(typeof spatial)[]=[];let branch:typeof spatial=[];
  for(const sample of spatial){
    if(sample.value.inferredEddyViscosity?.value==null){if(branch.length)viscositySegments.push(branch);branch=[];continue;}
    if(branch.length&&branch.at(-1)!.value.meanShear.sign*sample.value.meanShear.sign<0){viscositySegments.push(branch);branch=[];}
    branch.push(sample);
  }
  if(branch.length)viscositySegments.push(branch);
  const ensemble=Array.from({length:8},(_,j)=>({phase:j/8,value:reynoldsInstantaneousAt(parameters,xi,xCycles,j/8)}));
  const phases=samples([...grid(64),phaseCycles],1).map(phase=>({phase,value:reynoldsInstantaneousAt(parameters,xi,xCycles,phase)}));
  const asymmetry=samples([...grid(40,2).map(x=>x-1),parameters.meanAsymmetry],1,-1).map(c=>({c,value:reynoldsEnsembleBudgets({...parameters,meanAsymmetry:c})}));
  const small=tinySamples(xi).map(xi=>({xi,logPosition:Math.log10(xi),value:reynoldsEnsembleAt(parameters,xi)}));
  return {parameters,xi,xCycles,phaseCycles,query,point,budget,spatial,viscositySegments,ensemble,phases,asymmetry,small};
}
export function gaussianFilterLab(v: number[]) {
  validate(v,gaussianFilterBounds);
  const parameters: GaussianBurgersParameters = {wavenumber:v[0],filterWidth:v[1],kinematicViscosity:v[2],initialFirstAmplitude:v[3],initialSecondAmplitude:v[4],density:v[5],area:v[6],frameVelocity:v[7]};
  const time=v[8],xCycles=v[9],phaseCycles=v[10],query=gaussianBurgersAt(parameters,time,xCycles,phaseCycles),budget=gaussianBurgersBudgets(parameters,time);
  const spatial=samples([...grid(128),xCycles],1).map(xCycles=>({xCycles,value:gaussianBurgersAt(parameters,time,xCycles,phaseCycles)}));
  const temporal=samples([...grid(80,4),time],4).map(time=>({time,value:gaussianBurgersBudgets(parameters,time)}));
  const widths=samples([...grid(48,4),parameters.filterWidth],4).map(width=>({width,value:gaussianBurgersAt({...parameters,filterWidth:width},time,xCycles,phaseCycles)}));
  const small=tinySamples(parameters.filterWidth).map(width=>({width,logWidth:Math.log10(width),value:gaussianBurgersAt({...parameters,filterWidth:width},time,xCycles,phaseCycles)}));
  const frames=samples([...grid(32,4).map(x=>x-2),parameters.frameVelocity],2,-2).map(frameVelocity=>({frameVelocity,value:gaussianBurgersBudgets({...parameters,frameVelocity},time)}));
  return {parameters,time,xCycles,phaseCycles,query,budget,spatial,temporal,widths,small,frames};
}
