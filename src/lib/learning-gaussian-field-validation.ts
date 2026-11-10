/** Independent full response reconstruction. Only owned input supplies arithmetic
 * operands. Untrusted response fractions are compared as strings, never parsed.
 * Uses a bordered Gauss-Jordan inverse and determinant minors instead of producer LDL solves.
 */
import {Rational as Q,Z,O,sum} from './learning-graph-gauge-validation-math';
import {readGaussianFieldInput,type GaussianFieldInput} from './learning-gaussian-field-input';
import type {GaussianFieldResult} from './learning-gaussian-field-protocol';
import {snapshotQubitWire} from './learning-qubit-wire';
type Matrix=Q[][];
const integer=(v:number)=>new Q(BigInt(v)),zeros=(n:number,m=n):Matrix=>Array.from({length:n},()=>Array.from({length:m},()=>Z));
const dot=(a:Q[],b:Q[])=>sum(a.map((x,i)=>x.mul(b[i]))),transpose=(a:Matrix)=>a[0].map((_,j)=>a.map(row=>row[j]));
const multiply=(a:Matrix,b:Matrix)=>{const columns=transpose(b);return a.map(row=>columns.map(col=>dot(row,col)));};
const wire=(a:Matrix)=>a.map(row=>row.map(v=>v.text()));
/** Row-pivoted Bareiss determinant. Rational arithmetic retains exact divisions. */
function determinant(source:Matrix):Q{
 if(!source.length)return O;const a=source.map(row=>row.slice()),n=a.length;let previous=O,sign=O;
 for(let k=0;k<n-1;k++){let row=k;while(row<n&&a[row][k].n===BigInt(0))row++;if(row===n)return Z;if(row!==k){[a[row],a[k]]=[a[k],a[row]];sign=sign.neg();}const pivot=a[k][k];
  for(let i=k+1;i<n;i++)for(let j=k+1;j<n;j++)a[i][j]=pivot.mul(a[i][j]).sub(a[i][k].mul(a[k][j])).div(previous);
  for(let i=k+1;i<n;i++)a[i][k]=Z;previous=pivot;
 }return sign.mul(a[n-1][n-1]);
}
function inverse(source:Matrix):Matrix{
 const n=source.length,a=source.map((row,i)=>[...row,...Array.from({length:n},(_,j)=>i===j?O:Z)]);
 for(let k=0;k<n;k++){let at=k;while(at<n&&a[at][k].n===BigInt(0))at++;if(at===n)throw Error('Singular Gaussian precision preparation');[a[k],a[at]]=[a[at],a[k]];const pivot=a[k][k];a[k]=a[k].map(v=>v.div(pivot));for(let i=0;i<n;i++)if(i!==k){const scale=a[i][k];a[i]=a[i].map((v,j)=>v.sub(scale.mul(a[k][j])));}}
 return a.map(row=>row.slice(n));
}
function reconstructGaussianField(value:GaussianFieldInput){
 const input=readGaussianFieldInput(value);if(!input)throw Error('Invalid Gaussian input');const n=input.sites,mass=Q.read(input.massSquared),kappa=Q.read(input.stiffness),temperature=Q.read(input.temperature),source=input.source.map(Q.read),incidence:Q[][]=[];
 for(let i=0;i<n-1;i++){const row=Array.from({length:n},()=>Z);row[i]=O;row[i+1]=O.neg();incidence.push(row);}
 if(input.boundary==='periodic'){const row=Array.from({length:n},()=>Z);row[0]=O;row[n-1]=O.neg();incidence.push(row);}
 if(input.boundary==='dirichlet')for(const i of [0,n-1]){const row=Array.from({length:n},()=>Z);row[i]=O;incidence.push(row);}
 const laplacian=multiply(transpose(incidence),incidence),precision=laplacian.map((row,i)=>row.map((x,j)=>kappa.mul(x).add(i===j?mass:Z))),constrained=input.constraint==='mean-zero',dimension=constrained?n-1:n;
 const augmented=constrained?[...precision.map(row=>[...row,O]),[...Array.from({length:n},()=>O),Z]]:precision;
 const expandedInverse=inverse(augmented),response=expandedInverse.slice(0,n).map(row=>row.slice(0,n));
 const basis=zeros(n,dimension);for(let i=0;i<dimension;i++){basis[i][i]=O;if(constrained)basis[n-1][i]=O.neg();}
 const reduced=multiply(multiply(transpose(basis),precision),basis),lower=zeros(dimension),diagonal:Q[]=[];let previous=O;
 for(let i=0;i<dimension;i++){const leading=reduced.slice(0,i+1).map(row=>row.slice(0,i+1)),det=determinant(leading);diagonal.push(det.div(previous));lower[i][i]=O;for(let j=i+1;j<dimension;j++){const minor=[...reduced.slice(0,i).map(row=>row.slice(0,i+1)),reduced[j].slice(0,i+1)];lower[j][i]=determinant(minor).div(det);}previous=det;}
 if(diagonal.some(v=>v.n<=BigInt(0)))throw Error('Nonpositive restricted precision');
 const projector=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>(i===j?O:Z).sub(constrained?O.div(integer(n)):Z))),mean=response.map(row=>dot(row,source)),covariance=response.map(row=>row.map(x=>temperature.mul(x)));
 const moment=(indices:number[],mu=mean):Q=>{if(!indices.length)return O;const[a,...rest]=indices;return mu[a].mul(moment(rest,mu)).add(sum(rest.map((b,j)=>covariance[a][b].mul(moment(rest.filter((_,k)=>j!==k),mu)))));};
 const rawTwo=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>moment([i,j]))),[i,j,k,l]=input.probes,pairs=[[i,j,k,l],[i,k,j,l],[i,l,j,k]],wick=pairs.map(([a,b,c,d])=>covariance[a][b].mul(covariance[c][d])),centered=moment([...input.probes],Array.from({length:n},()=>Z)),rawFourth=moment([...input.probes]),meanProduct=input.probes.map(a=>mean[a]).reduce((a,b)=>a.mul(b),O);
 const ambient=precision.map((row,i)=>dot(row,mean).sub(source[i])),residualMean=projector.map(row=>dot(row,ambient)),identity=multiply(multiply(projector,precision),response).map((row,i)=>row.map((x,j)=>x.sub(projector[i][j]))),restrictedDet=constrained?determinant(augmented).neg().div(integer(n)):determinant(precision);
 return{input,integrationDimension:dimension,precision:wire(precision),laplacian:wire(laplacian),basis:wire(basis),projector:wire(projector),reducedPrecision:wire(reduced),factorization:{lower:wire(lower),diagonal:diagonal.map(v=>v.text()),reducedDeterminant:determinant(reduced).text()},normalization:{restrictedPrecisionDeterminant:restrictedDet.text(),euclideanMeasureGramDeterminant:constrained?String(n):'1',formula:'Z(0)=(2πT)^(dimension/2)/sqrt(restrictedPrecisionDeterminant)',logZRatio:dot(source,mean).div(integer(2).mul(temperature)).text()},mean:mean.map(v=>v.text()),susceptibility:wire(response),connectedTwo:wire(covariance),rawTwo:wire(rawTwo),sourceDerivatives:{logZGradient:mean.map(v=>v.div(temperature).text()),logZHessian:covariance.map(row=>row.map(v=>v.div(temperature.mul(temperature)).text()))},fourPoint:{indices:input.probes,meanProduct:meanProduct.text(),wickPairProducts:wick.map(v=>v.text()),centered:centered.text(),raw:rawFourth.text(),connectedCumulant:centered.sub(sum(wick)).text(),incorrectRawPairSubtraction:rawFourth.sub(sum(pairs.map(([a,b,c,d])=>rawTwo[a][b].mul(rawTwo[c][d])))).text(),incorrectSubtractionDifference:integer(-2).mul(meanProduct).text()},certificates:{ambientMeanResidual:ambient.map(v=>v.text()),constraintMultiplier:constrained?sum(ambient).neg().div(integer(n)).text():null,projectedMeanResidual:residualMean.map(v=>v.text()),projectedGreenResidual:wire(identity),constraintMeanSum:constrained?sum(mean).text():null,covarianceRowSums:constrained?covariance.map(row=>sum(row).text()):null,precisionPositiveOnIntegrationSpace:true},contract:'Finite classical equilibrium Gaussian field on 3..8 sites. Temperature is positive. Open, periodic and fixed-zero exterior boundaries differ. Mean-zero uses the induced Euclidean measure on sum(phi)=0; it is a different ensemble. No continuum measure, time ordering, interacting Wick rule or fitted correlation length is claimed.'};
}
function same(a:unknown,b:unknown):boolean{
 if(b===null||typeof b!=='object')return a===b;if(!a||typeof a!=='object'||Array.isArray(a)!==Array.isArray(b))return false;
 if(Array.isArray(b))return Array.isArray(a)&&a.length===b.length&&b.every((v,i)=>same(a[i],v));const left=a as Record<string,unknown>,right=b as Record<string,unknown>,keys=Object.keys(right);return Object.keys(left).length===keys.length&&keys.every(k=>Object.hasOwn(left,k)&&same(left[k],right[k]));
}
const cache=new Map<string,unknown>();
export function validateGaussianFieldResult(source:GaussianFieldInput,value:unknown):GaussianFieldResult|null{
 try{const input=readGaussianFieldInput(source);if(!input)return null;const owned=snapshotQubitWire(value),key=JSON.stringify(input);let expected=cache.get(key);if(!expected){expected=snapshotQubitWire(reconstructGaussianField(input));if(cache.size===16)cache.delete(cache.keys().next().value!);cache.set(key,expected);}return same(owned,expected)?owned as GaussianFieldResult:null;}catch{return null;}
}
