/** Deterministic draft-input corpus for the forthcoming real Worker smoke test. */
import {mkdirSync,writeFileSync}from'node:fs';
import {quantumInformationPresets}from'../../src/lib/learning-quantum-information-presets';
import {quantumInformationInputKey}from'../../src/lib/learning-quantum-information-input';
const directory='.runtime/learning/quantum-information-input';
const cases=Object.entries(quantumInformationPresets).flatMap(([unit,presets])=>presets.map(p=>({unit,preset:p.id,label:p.label,input:p.input,inputKey:quantumInformationInputKey(p.input)})));
mkdirSync(directory,{recursive:true});
writeFileSync(`${directory}/request-corpus.json`,JSON.stringify({version:1,status:'draft-inputs-not-worker-verified',cases},null,2)+'\n');
console.log(JSON.stringify({units:Object.keys(quantumInformationPresets).length,inputs:cases.length}));
