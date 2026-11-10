import { reynoldsEnsembleAt } from '@/lib/learning-continuum-turbulence';
import { gaussianBurgersAt } from '@/lib/learning-continuum-gaussian-filter';
import type { ContinuumTurbulenceKind } from '@/lib/learning-continuum-turbulence-lab';
export default function TurbulenceCover({kind}:{kind:ContinuumTurbulenceKind}) {
  const reynolds=kind==='reynolds-ensemble-stress-closure-defect';
  const values=Array.from({length:81},(_,i)=>{const x=i/80;
    if(reynolds){const q=reynoldsEnsembleAt({height:2,wavenumber:1.5,spanWidth:1,density:1,dynamicViscosity:.1,bulkVelocity:.8,meanAsymmetry:.25,firstStreamfunctionAmplitude:1,secondStreamfunctionAmplitude:.7,candidateEddyViscosity:.01},x);return[x,q.covarianceUU.value!,q.covarianceVV.value!,q.covarianceUV.value!];}
    const q=gaussianBurgersAt({wavenumber:1,filterWidth:.5,kinematicViscosity:.1,initialFirstAmplitude:1,initialSecondAmplitude:-.5,density:1,area:1,frameVelocity:0},.3,x);return[x,q.velocity.value!,q.filteredVelocity.value!];});
  const top=Math.max(...values.flatMap(v=>v.slice(1).map(Math.abs))),path=(index:number)=>values.map((v,i)=>`${i?'L':'M'}${45+270*v[0]} ${130-95*v[index]/top}`).join(' ');
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art"><path d="M45 30V225M45 130H320" fill="none" stroke="var(--learning-neutral)"/>{(reynolds?[1,2,3]:[1,2]).map((index)=><path key={index} d={path(index)} fill="none" stroke={`var(--learning-${['blue','orange','green'][index-1]})`} strokeWidth="3" strokeDasharray={index>1?'5 3':undefined}/>)}<text x="45" y="24" fill="currentColor" fontSize="14">{reynolds?'Ruu · Rvv · Ruv':'Gaussian · u / ū'}</text></svg>;
}
