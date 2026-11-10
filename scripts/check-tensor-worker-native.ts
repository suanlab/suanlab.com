import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {tensorKinds,snapshotTensorInput,isTensorSuccessForInput,type TensorInput} from '../src/lib/learning-tensor-protocol';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';
import type {PartitionObservation as Q} from '../src/lib/learning-partition-exact-core';
type F={n:bigint;d:bigint};const fraction=(n:bigint,d=BigInt(1)):F=>({n,d}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,eq=(a:F,b:F)=>a.n*b.d===b.n*a.d;
function parse(s:string):F{if(s.includes('/')){const[n,d]=s.split('/');return fraction(BigInt(n),BigInt(d));}const m=/^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(s);assert(m,s);const n=BigInt(m[1]+m[2]+(m[3]||'')),k=Number(m[4]||0)-(m[3]||'').length;return k>=0?fraction(n*BigInt(10)**BigInt(k)):fraction(n,BigInt(10)**BigInt(-k));}
const endpoint=(q:NonNullable<Q['bounds']>['lower'])=>fraction(BigInt(q.numerator.coefficient),BigInt(q.denominator.coefficient));let directedChecks=0,exactChecks=0,maxCoefficientDigits=0;
function check(actual:unknown,expected:string,path:string,exact=false){assert(isMasterObservation(actual),path+' shape');const q=actual as Q;assert(q.bounds,path);const lo=endpoint(q.bounds.lower),hi=endpoint(q.bounds.upper);assert(le(lo,hi),path+' order');maxCoefficientDigits=Math.max(maxCoefficientDigits,...[q.bounds.lower,q.bounds.upper].flatMap(b=>[b.numerator.coefficient.length,b.denominator.coefficient.length]));
 const decimal=expected.startsWith('decimal:'),x=parse(decimal?expected.slice(8):expected),absolute=x.n<0?-x.n:x.n,scale=absolute>x.d?absolute:x.d;
 if(decimal){const eps=BigInt(10)**BigInt(200),lower=fraction(x.n*eps-scale,x.d*eps),upper=fraction(x.n*eps+scale,x.d*eps);assert(le(lo,upper)&&le(lower,hi),path+' reference '+expected);}
 else{assert(le(lo,x)&&le(x,hi),path+' contains '+expected);if(exact){assert(eq(lo,x)&&eq(hi,x),path+' exact');exactChecks++;}}
 assert(le(fraction(hi.n*lo.d-lo.n*hi.d,hi.d*lo.d),fraction(scale,x.d*BigInt(10)**BigInt(125))),path+' width');if(!exact)directedChecks++;
}
function compare(actual:unknown,expected:unknown,path='result'){
 if(typeof expected==='string'&&(/^-?\d/.test(expected)||expected.startsWith('decimal:'))){const exact=!expected.startsWith('decimal:');check(actual,expected,path,exact);return;}
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((v,i)=>compare(actual[i],v,path+'.'+i));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);for(const[k,v]of Object.entries(expected))compare((actual as Record<string,unknown>)[k],v,path+'.'+k);return;}assert.equal(actual,expected,path);
}

async function main(){
 const runtime='.runtime/learning/quantum-tensor-worker';
 const families=tensorKinds.map((kind,i)=>({kind,cases:JSON.parse(readFileSync(`scripts/fixtures/${i===0?'tensor-pure':'mixed-entanglement'}-reference.json`,'utf8')).cases as {label:string;input:TensorInput['parameters'];expected:unknown}[]}));
 const preparations=families.flatMap(f=>f.cases.map(c=>({label:c.label,input:snapshotTensorInput({kind:f.kind,parameters:c.input}as TensorInput),expected:c.expected})));
 const code=await build({entryPoints:['src/lib/learning-tensor.worker.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'iife',target:'es2020'});
 writeFileSync(`${runtime}/native-worker-bundle.js`,code.outputFiles[0].text);
 const reader=await build({entryPoints:['src/components/learning/use-tensor-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'});
 const inputs=Object.keys(reader.metafile!.inputs);assert(!inputs.some(p=>/learning-(tensor-pure|tensor-roots|mixed-entanglement(?:-input|-algebra)?|quantum-evolution-interval|observable-matrix|quantum-hilbert-core)\.ts|entropy-production-math|interval-transcendentals/.test(p)),inputs.join('\n'));
 writeFileSync(`${runtime}/reader-dependency-graph.json`,JSON.stringify({inputs,bytes:reader.outputFiles[0].contents.length,numericalEnginesExcluded:true},null,2)+'\n');
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage();const pageErrors:string[]=[];page.on('pageerror',e=>pageErrors.push(e.message));await page.goto('http://localhost:4175/book/interactive/');
  const run=await page.evaluate(async ({code,preparations})=>{
   const url=URL.createObjectURL(new Blob([code],{type:'text/javascript'})),responses:unknown[]=[],starts:unknown[]=[],times:number[]=[];let created=0,terminated=0;
   try{for(let i=0;i<preparations.length;i++){
    const worker=new Worker(url,{type:'module',name:'tensor-native-verification'});created++;const before=performance.now();
    try{const messages=await new Promise<unknown[]>((resolve,reject)=>{
     const messages:unknown[]=[],timer=setTimeout(()=>reject(Error('Native tensor Worker timeout')),45000);
     worker.onmessage=event=>{messages.push(event.data);if(messages.length===2){clearTimeout(timer);resolve(messages);}};
     worker.onerror=event=>{event.preventDefault();clearTimeout(timer);reject(Error(event.message));};worker.onmessageerror=()=>{clearTimeout(timer);reject(Error('Native structured clone failed'));};
     worker.postMessage({version:1,requestId:i+1,input:preparations[i]});
    });starts.push(messages[0]);responses.push(messages[1]);times.push(performance.now()-before);
    }finally{worker.onmessage=null;worker.onerror=null;worker.onmessageerror=null;worker.terminate();terminated++;}
   }}finally{URL.revokeObjectURL(url);}
   return{responses,starts,created,terminated,times};
  },{code:code.outputFiles[0].text,preparations:preparations.map(p=>p.input)});
  const cases:unknown[]=[];
  for(const[i,p]of preparations.entries()){
   assert.deepEqual(run.starts[i],{version:1,requestId:i+1,phase:'started',kind:p.input.kind});const response=run.responses[i];assert(isTensorSuccessForInput(response,p.input,i+1),p.label);compare(response.result,p.expected,p.label);cases.push({kind:p.input.kind,label:p.label,milliseconds:run.times[i]});
  }
  assert.equal(run.created,120);assert.equal(run.terminated,120);assert.deepEqual(pageErrors,[]);
  const report={workerType:'native browser module Worker from bundled entry via Blob;not a new Next static-export asset',preparations:preparations.length,uniquePreparations:new Set(preparations.map(p=>JSON.stringify(p.input))).size,workersCreated:run.created,workersTerminated:run.terminated,startedFrames:run.starts.length,validatedResults:run.responses.length,exactRationalCertificates:exactChecks,directedObservationChecks:directedChecks,maxCoefficientDigits,maximumRelativeWidthAgainstMaxOneReference:1e-125,pageErrors,readerNumericalEnginesExcluded:true,cases};
  writeFileSync(`${runtime}/native-verification.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({...report,cases:undefined}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});


