/** Measure complete producer responses,without publishing or changing the site. */
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {calculateRealTimePhase} from '../../src/lib/learning-real-time-phase';
import {realTimePhaseExamples} from '../../src/lib/learning-real-time-phase-input';
import {snapshotQubitWire} from '../../src/lib/learning-qubit-wire';
const root='.runtime/learning/real-time-phase-numerics';mkdirSync(root,{recursive:true});
const output=`${root}/profile-node${process.versions.node.split('.')[0]}.json`,foldOnly=process.argv.includes('--fold-only'),regulatedOnly=process.argv.includes('--regulated-only');
const selectedIds=new Set(realTimePhaseExamples.filter(({input})=>regulatedOnly?input.mode==='chain'&&BigInt(input.damping.split('/')[0])!==BigInt(0):foldOnly?input.mode==='fold':true).map(row=>row.id));
const previous=existsSync(output)?JSON.parse(readFileSync(output,'utf8')):null;
const rows:Record<string,unknown>[]=(foldOnly||regulatedOnly)&&previous?previous.rows.filter((row:{id:string})=>!selectedIds.has(row.id)):[];
for(const {id,input}of realTimePhaseExamples) {
  if(!selectedIds.has(id))continue;
  const started=performance.now(),result=calculateRealTimePhase(input),calculationMilliseconds=performance.now()-started;
  const copied=snapshotQubitWire(result);let characters=0,maxString=0,nodes=0;
  const walk=(x:unknown):void=>{nodes++;if(typeof x==='string'){characters+=x.length;maxString=Math.max(maxString,x.length);}else if(Array.isArray(x))x.forEach(walk);else if(x&&typeof x==='object')Object.values(x).forEach(walk);};walk(copied);
  const row={id,kind:result.kind,calculationMilliseconds,bytes:Buffer.byteLength(JSON.stringify(copied)),characters,maxString,nodes};rows.push(row);
  writeFileSync(output,JSON.stringify({fullProducerResponse:true,independentValidatorComplete:false,foldShapeRefreshed:foldOnly||previous?.foldShapeRefreshed===true,regulatedShapeRefreshed:regulatedOnly,rows},null,2)+'\n');
  console.log(JSON.stringify(row));
}
