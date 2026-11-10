import { readFileSync } from 'node:fs';
import { snapshotQubitInput,qubitKinds,type QubitInput } from '../../src/lib/learning-qubit-input';
type Reference={name:string;expected:unknown;[key:string]:unknown};
const circuit=JSON.parse(readFileSync('scripts/fixtures/qubit-circuit-response-independent.json','utf8')) as {cases:Reference[]};
const channel=JSON.parse(readFileSync('scripts/fixtures/qubit-channel-independent.json','utf8')) as Record<string,Reference[]>;
export const qubitWorkerCases:{label:string;input:QubitInput;expected:unknown}[]=[
  ...circuit.cases.map(c=>({label:`circuit/${c.name}`,input:snapshotQubitInput({kind:'circuit',sites:c.sites,gates:c.gates,input:c.input}),expected:c.expected})),
  ...channel.channelCases.map(c=>({label:`channel/${c.name}`,input:snapshotQubitInput({kind:'channel',map:c.map,state:c.state}),expected:c.expected})),
  ...channel.environmentCases.map(c=>({label:`environment/${c.name}`,input:snapshotQubitInput({kind:'environment',parameter:c.parameter,state:c.state}),expected:c.expected})),
  ...channel.instrumentCases.map(c=>({label:`instrument/${c.name}`,input:snapshotQubitInput({kind:'instrument',outcomes:c.outcomes,state:c.state,correctZ:c.correctZ}),expected:c.expected})),
  ...channel.deferredCases.map(c=>({label:`deferred/${c.name}`,input:snapshotQubitInput({kind:'deferred',state:c.state}),expected:c.expected})),
];
export const qubitWorkerRepresentatives=qubitKinds.map(kind=>{
  const preferred={circuit:'bell-control-0',channel:'damping-1/2/plus-i',environment:'1/2/plus-i',instrument:'damping-mixed-record/1/2/plus-i/True',deferred:'plus-i'}[kind];
  return qubitWorkerCases.find(c=>c.input.kind===kind&&c.label.endsWith(preferred))??qubitWorkerCases.find(c=>c.input.kind===kind)!;
});
