import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
/** Explicit wording corrections made during the path-integral topic review.
 * Historical fixtures are immutable. Reversing ONLY these replacements must
 * recover their exact hash; every other byte still has to match the baseline.
 */
export const pathIntegralEditorialCorrections:Record<string,readonly(readonly[string,string])[]>={
 'content/learning/physics-math-for-ai/euclidean-quantum-kernels-and-time-slicing.en.md':[
  ['The planned exact calculator accepts','The exact calculator accepts'],
  ['Only the input range through32 intervals is planned for full matrices.','Full matrices are computed for the input range through32 intervals.'],
 ],
 'content/learning/physics-math-for-ai/euclidean-quantum-kernels-and-time-slicing.ko.md':[
  ['계획한 정확분수 계산기는','정확분수 계산기는'],
  ['전체 행렬을 계산할 계획 범위는 입력32구간까지다.','전체 행렬은 입력32구간까지 계산한다.'],
 ],
};
const hash=(raw:string)=>createHash('sha256').update(raw).digest('hex');
export function assertPreservedLearningFile(path:string,expected:string){
 const raw=readFileSync(path,'utf8');if(hash(raw)===expected)return;
 const edits=pathIntegralEditorialCorrections[path];assert(edits,'Unexpected change to preserved file: '+path);let original=raw;
 for(const[before,after]of edits){assert.equal(original.split(after).length-1,1,'Editorial replacement must occur exactly once: '+path);original=original.replace(after,before);}
 assert.equal(hash(original),expected,'Changes exceed the declared editorial correction: '+path);
}
