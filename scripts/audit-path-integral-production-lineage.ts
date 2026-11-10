import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import path from 'node:path';
const root='.runtime/learning/path-integral-topic-audit',hash=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
const units=[
 {id:'brownian-bridge-time-slicing-measure',stem:'brownian-bridge',candidate:'/tmp/suanlab-brownian-bridge-candidate-20261110',folder:'brownian-bridge-reader-integration'},
 {id:'euclidean-quantum-kernels-and-time-slicing',stem:'euclidean-kernel',candidate:'/tmp/suanlab-euclidean-kernel-candidate-20261110',folder:'euclidean-kernel-reader-integration'},
 {id:'saddle-fluctuation-and-nongaussian-approximation',stem:'saddle-fluctuation',candidate:'/tmp/suanlab-saddle-fluctuation-candidate-20261110',folder:'saddle-fluctuation-reader-integration'},
 {id:'real-time-phase-and-classical-limit',stem:'real-time-phase',candidate:'/tmp/suanlab-real-time-phase-candidate-20261111',folder:'real-time-phase-reader-integration'},
];
async function main(){const rows=[];
 for(const unit of units){const graph=await build({entryPoints:[`src/lib/learning-${unit.stem}.worker.ts`,`src/lib/learning-${unit.stem}-validation.worker.ts`],bundle:true,write:false,outdir:'unused',platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),files=Object.keys(graph.metafile!.inputs).filter(p=>p.startsWith('src/'));const sources=files.map(p=>({path:p,sha256:hash(p),priorSha256:hash(path.join(unit.candidate,p))}));for(const source of sources)assert.equal(source.sha256,source.priorSha256,unit.id+': '+source.path);
  const record=JSON.parse(readFileSync(`.runtime/learning/${unit.folder}/production-assets.json`,'utf8')) as{assets:{url:string;sha256:string}[];rows:unknown[]};assert.equal(record.assets.length,2);const assets=record.assets.map(asset=>{const pathname=new URL(asset.url).pathname;const actual=hash(path.join(unit.candidate,'out',pathname));assert.equal(actual,asset.sha256);return{path:pathname,sha256:actual};});rows.push({unitId:unit.id,unchangedNumericalDependencies:sources.length,priorProductionRequests:record.rows.length,reusedRequestCountNotRerun:true,assets,sources});
 }writeFileSync(root+'/production-lineage.json',JSON.stringify({rows,allNumericalDependenciesUnchanged:true,previousProductionExecutionsReused:true},null,2)+'\n');console.log(rows.map(r=>({id:r.unitId,dependencies:r.unchangedNumericalDependencies,oldRequests:r.priorProductionRequests})));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
