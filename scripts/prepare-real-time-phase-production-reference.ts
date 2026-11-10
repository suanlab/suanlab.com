import {writeFileSync,readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
import {realTimePhaseGuidePreparations} from '../src/data/real-time-phase-teaching';
import {realTimePhaseInputKey} from '../src/lib/learning-real-time-phase-protocol';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
const root='.runtime/learning/real-time-phase-reader-integration';
const sourceFiles=['src/lib/learning-real-time-phase.ts','src/lib/learning-real-time-phase-exact.ts','src/lib/learning-real-time-wave-observations.ts','src/lib/learning-real-time-matrix-observations.ts','src/lib/learning-real-time-airy.ts'];
const producerSourceHashes=Object.fromEntries(sourceFiles.map(p=>[p,createHash('sha256').update(readFileSync(p)).digest('hex')]));
const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
const hash=(v:unknown)=>createHash('sha256').update(stable(v)).digest('hex');
async function main(){const seen=new Set<string>(),cases=[];for(const e of [...realTimePhaseExamples,...realTimePhaseGuidePreparations.map(p=>({id:'guide:'+p.id,input:p.input}))]){const key=realTimePhaseInputKey(e.input);if(seen.has(key))continue;seen.add(key);cases.push(e);}const rows=[];
 for(const c of cases){rows.push({id:c.id,input:c.input,inputHash:hash(c.input),resultHash:hash(calculateRealTimePhase(c.input))});writeFileSync(root+'/production-node-progress.json',JSON.stringify({completed:rows.length,total:cases.length,last:c.id,rows,producerSourceHashes},null,2)+'\n');await new Promise(resolve=>setImmediate(resolve));}
 writeFileSync(root+'/production-node-references.json',JSON.stringify({cases:rows,canonicalStableHash:true,independentMathematicalProof:false,producerSourceHashes},null,2)+'\n');console.log('Production reference inputs:',rows.length);}
main().catch(error=>{console.error(error);process.exitCode=1;});
