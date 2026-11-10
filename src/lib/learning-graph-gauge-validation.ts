/** Independent scalar-form polarization and exact polynomial-difference verifier.
 * Reconstructs every response claim. Numerical engines stay outside the reader.
 */
import {readGraphGaugeInput,graphGaugeInputKey,type GraphGaugeInput} from './learning-graph-gauge-input';
import {graphGaugeContract} from './learning-graph-gauge-contract';
import {snapshotQubitWire} from './learning-qubit-wire';
import type {GraphGaugeResult} from './learning-graph-gauge';
import {Rational as R,Z,O,T,CZ,CO,CI,complex,plus,minus,times,conjugate,norm,wire,sum,equal,unitPhase,type Complex} from './learning-graph-gauge-validation-math';
function same(actual:unknown,expected:unknown):boolean{
 if(expected===null||typeof expected!=='object')return actual===expected;
 if(!actual||typeof actual!=='object'||Array.isArray(actual)!==Array.isArray(expected))return false;
 if(Array.isArray(expected))return Array.isArray(actual)&&actual.length===expected.length&&expected.every((x,i)=>same(actual[i],x));
 const a=actual as Record<string,unknown>,e=expected as Record<string,unknown>,keys=Object.keys(e);
 return Object.keys(a).length===keys.length&&keys.every(k=>Object.hasOwn(a,k)&&same(a[k],e[k]));
}
export function validateGraphGaugeResult(source:GraphGaugeInput,value:unknown):GraphGaugeResult|null{
 try{
  const input=readGraphGaugeInput(source);if(!input)return null;const owned=snapshotQubitWire(value);
  if(!owned||typeof owned!=='object')return null;const attributed=readGraphGaugeInput((owned as Record<string,unknown>).input);
  if(!attributed||graphGaugeInputKey(attributed)!==graphGaugeInputKey(input))return null;
  const n=input.matter.length,psi=input.matter.map(z=>complex(R.read(z.real),R.read(z.imaginary))),g=input.frames.map(unitPhase),U=input.edges.map(unitPhase),changedPsi=psi.map((z,i)=>times(z,g[i])),changedU=U.map((z,k)=>times(z,times(g[input.edges[k].from],conjugate(g[input.edges[k].to])))),weights=input.edges.map(e=>R.read(e.weight)),a=R.read(input.quadratic),b=R.read(input.quartic),kappa=R.read(input.loopCoupling);
  const differences=(z:Complex[],links:Complex[])=>input.edges.map((e,k)=>minus(z[e.from],times(links[k],z[e.to])));
  const hopping=(z:Complex[],links:Complex[])=>sum(differences(z,links).map((d,k)=>norm(d).mul(weights[k])));
  const quadratic=(z:Complex[],links:Complex[])=>hopping(z,links).add(a.mul(sum(z.map(norm))));
  const quartic=(z:Complex[])=>b.div(T).mul(sum(z.map(x=>norm(x).mul(norm(x)))));
  const walkProduct=(links:Complex[],steps:GraphGaugeInput['walks'][number]['steps'])=>steps.reduce((z,s)=>times(z,s.forward?links[s.edge]:conjugate(links[s.edge])),CO);
  const loopEnergy=(links:Complex[])=>sum(input.walks.map(w=>kappa.mul(O.sub(walkProduct(links,w.steps)[0]))));
  const energy=(z:Complex[],links:Complex[])=>quadratic(z,links).add(quartic(z)).add(loopEnergy(links));
  const matrix=(links:Complex[])=>{
   // Probe scalar energies rather than assembling the producer's incidence rows.
   const basis=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?CO:CZ)),diagonal=basis.map(z=>quadratic(z,links)),H=Array.from({length:n},()=>Array.from({length:n},()=>CZ));
   for(let i=0;i<n;i++){
    H[i][i]=complex(diagonal[i]);
    for(let j=i+1;j<n;j++){
     const real=quadratic(basis[i].map((z,k)=>plus(z,basis[j][k])),links).sub(diagonal[i]).sub(diagonal[j]).div(T);
     const imaginary=quadratic(basis[i].map((z,k)=>plus(z,times(CI,basis[j][k]))),links).sub(diagonal[i]).sub(diagonal[j]).div(T).neg();
     H[i][j]=complex(real,imaginary);H[j][i]=conjugate(H[i][j]);
    }
   }return H;
  };
  const gradient=(z:Complex[],links:Complex[])=>{
   // For a degree-four energy, the central derivative error is exactly
   // 2*b*coordinate*h^2. Remove it before taking the Wirtinger half.
   const h=new R(BigInt(2),BigInt(7));
   return z.map((_,i)=>{
    // Terms not incident to i are constant under this probe. Excluding them
    // avoids forming and cancelling enormous unrelated denominators.
    const localEnergy=(state:Complex[])=>sum(input.edges.flatMap((e,k)=>e.from===i||e.to===i?[norm(minus(state[e.from],times(links[k],state[e.to]))).mul(weights[k])]:[])).add(a.mul(norm(state[i]))).add(b.div(T).mul(norm(state[i]).mul(norm(state[i]))));
    const parts=[0,1].map(axis=>{
     const step=axis===0?complex(h):complex(Z,h),right=[...z],left=[...z];right[i]=plus(z[i],step);left[i]=minus(z[i],step);
     return localEnergy(right).sub(localEnergy(left)).div(T.mul(h)).sub(T.mul(b).mul(z[i][axis]).mul(h).mul(h)).div(T);
    });return complex(parts[0],parts[1]);
   });
  };
  const H=matrix(U),changedH=matrix(changedU),grad=gradient(psi,U),changedGrad=gradient(changedPsi,changedU),D=differences(psi,U),changedD=differences(changedPsi,changedU),fieldOnlyD=differences(changedPsi,U);
  const edges=input.edges.map((e,k)=>{
   const dressed=times(conjugate(psi[e.from]),times(U[k],psi[e.to])),changedDressed=times(conjugate(changedPsi[e.from]),times(changedU[k],changedPsi[e.to]));
   return{from:e.from,to:e.to,link:wire(U[k]),transformedLink:wire(changedU[k]),unitNormResidual:norm(U[k]).sub(O).text(),transformedUnitNormResidual:norm(changedU[k]).sub(O).text(),difference:wire(D[k]),transformedDifference:wire(changedD[k]),covarianceResidual:wire(minus(changedD[k],times(g[e.from],D[k]))),energy:norm(D[k]).mul(weights[k]).text(),transformedEnergy:norm(changedD[k]).mul(weights[k]).text(),fieldOnlyEnergy:norm(fieldOnlyD[k]).mul(weights[k]).text(),dressed:wire(dressed),transformedDressed:wire(changedDressed),dressedResidual:wire(minus(changedDressed,dressed))};
  });
  const walks=input.walks.map(w=>{
   const z=walkProduct(U,w.steps),tz=walkProduct(changedU,w.steps),reverse=[...w.steps].reverse().map(s=>({edge:s.edge,forward:!s.forward}));
   return{start:w.start,stepCount:w.steps.length,holonomy:wire(z),transformedHolonomy:wire(tz),invarianceResidual:wire(minus(tz,z)),unitNormResidual:norm(z).sub(O).text(),reverseHolonomy:wire(walkProduct(U,reverse)),penalty:kappa.mul(O.sub(z[0])).text(),transformedPenalty:kappa.mul(O.sub(tz[0])).text()};
  });
  // Independent graph traversal, rather than the producer's union-find.
  const unseen=new Set(Array.from({length:n},(_,i)=>i));let components=0;
  while(unseen.size){components++;const start=unseen.values().next().value as number;unseen.delete(start);const stack=[start];while(stack.length){const i=stack.pop()!;for(const e of input.edges)for(const[u,v]of [[e.from,e.to],[e.to,e.from]])if(u===i&&unseen.delete(v))stack.push(v);}}
  const total=energy(psi,U),transformedTotal=energy(changedPsi,changedU),fieldOnlyTotal=energy(changedPsi,U);
  const matrixEnergy=H.reduce((acc,row,i)=>plus(acc,times(conjugate(psi[i]),row.reduce((z,x,j)=>plus(z,times(x,psi[j])),CZ))),CZ);
  const covariance=changedH.map((row,i)=>row.map((z,j)=>wire(minus(z,times(g[i],times(H[i][j],conjugate(g[j]))))))),hermiticity=H.map((row,i)=>row.map((z,j)=>wire(minus(z,conjugate(H[j][i])))));
  const expected={input,sites:n,edgeCount:input.edges.length,componentCount:components,cycleRank:input.edges.length-n+components,frames:g.map(wire),nodes:psi.map((z,i)=>({matter:wire(z),transformedMatter:wire(changedPsi[i]),density:norm(z).text(),transformedDensity:norm(changedPsi[i]).text(),densityResidual:norm(changedPsi[i]).sub(norm(z)).text(),gradient:wire(grad[i]),transformedGradient:wire(changedGrad[i]),gradientCovarianceResidual:wire(minus(changedGrad[i],times(g[i],grad[i])))})),edges,walks,
   energy:{hopping:hopping(psi,U).text(),transformedHopping:hopping(changedPsi,changedU).text(),fieldOnlyHopping:hopping(changedPsi,U).text(),onsiteQuadratic:a.mul(sum(psi.map(norm))).text(),onsiteQuartic:quartic(psi).text(),loops:loopEnergy(U).text(),transformedLoops:loopEnergy(changedU).text(),total:total.text(),transformedTotal:transformedTotal.text(),fieldOnlyTotal:fieldOnlyTotal.text(),invarianceResidual:transformedTotal.sub(total).text(),quadraticOperatorResidual:wire(minus(matrixEnergy,complex(quadratic(psi,U))))},
   operator:{original:H.map(row=>row.map(wire)),transformed:changedH.map(row=>row.map(wire)),covarianceResidual:covariance,hermiticityResidual:hermiticity,trace:sum(H.map((row,i)=>row[i][0])).text(),transformedTrace:sum(changedH.map((row,i)=>row[i][0])).text(),frobeniusSquared:sum(H.flat().map(norm)).text(),transformedFrobeniusSquared:sum(changedH.flat().map(norm)).text()},
   classification:{globalUniformFrame:g.every(z=>equal(z,g[0])),componentwiseConstantFrame:input.edges.every(e=>equal(g[e.from],g[e.to])),linksUnchanged:U.every((z,k)=>equal(z,changedU[k])),fieldOnlyEnergyUnchanged:fieldOnlyTotal.text()===total.text(),completeLoopBasisClaimed:false,loopInvariantsCompleteForMatter:false,dynamicsSpecified:false,continuumLimitProvided:false,gaussConstraintImplemented:false,boundaryAnchorsSpecified:false},contract:graphGaugeContract};
  return same(owned,expected)?owned as GraphGaugeResult:null;
 }catch{return null;}
}
