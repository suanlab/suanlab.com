/** Actual core-only wire measurements; not a complete calculator budget. */
import {mkdirSync,writeFileSync} from 'node:fs';
import {realTimePhaseExamples} from '../../src/lib/learning-real-time-phase-input';
import {buildRealTimeChainExact,buildRealTimePacketExact} from '../../src/lib/learning-real-time-phase-exact';
import {realTimeChainCoreWire,realTimePacketCoreWire} from '../../src/lib/learning-real-time-phase-core-wire';
import {snapshotQubitWire} from '../../src/lib/learning-qubit-wire';
const root='.runtime/learning/real-time-phase-core';mkdirSync(root,{recursive:true});
const rows:Record<string,unknown>[]=[];
for(const {id,input} of realTimePhaseExamples) {
  if(input.mode==='fold')continue;
  const started=performance.now();
  const wire=input.mode==='chain'?realTimeChainCoreWire(buildRealTimeChainExact(input)):realTimePacketCoreWire(buildRealTimePacketExact(input));
  const snapshot=snapshotQubitWire(wire);let nodes=0,characters=0,maxString=0;
  const visit=(x:unknown):void=>{nodes++;if(typeof x==='string'){characters+=x.length;maxString=Math.max(maxString,x.length);}else if(Array.isArray(x))x.forEach(visit);else if(x&&typeof x==='object')Object.values(x).forEach(visit);};visit(snapshot);
  const row={id,milliseconds:performance.now()-started,bytes:Buffer.byteLength(JSON.stringify(snapshot)),characters,maxString,nodes};rows.push(row);
  writeFileSync(`${root}/core-profile-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({coreOnly:true,fullResponseBudgetFinalized:false,rows},null,2)+'\n');
  console.log(JSON.stringify(row));
}
