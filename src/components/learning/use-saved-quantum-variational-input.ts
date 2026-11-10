'use client';
import{useEffect,useState}from'react';
import{quantumVariationalPresetInput,type QuantumVariationalKind,type QuantumVariationalInput}from'@/lib/learning-quantum-variational';
import{isQuantumVariationalInput,parseQuantumVariationalStoredInput,snapshotQuantumVariationalInput}from'@/lib/learning-quantum-variational-protocol';
export function useSavedQuantumVariationalInput(kind:QuantumVariationalKind){
 const key=`suanlab-quantum-variational-${kind}`,[state,setState]=useState(()=>({kind,input:quantumVariationalPresetInput(kind,0),ready:false,storage:true}));
 useEffect(()=>{let input=quantumVariationalPresetInput(kind,0),storage=true;try{const raw=localStorage.getItem(key),saved=raw?parseQuantumVariationalStoredInput(raw):null;if(saved?.kind===kind)input=saved;}catch{storage=false;}setState({kind,input,ready:true,storage});},[key,kind]);
 useEffect(()=>{if(!state.ready||state.kind!==kind)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotQuantumVariationalInput(state.input)}));}catch{setState(s=>s.storage?{...s,storage:false}:s);}},[key,kind,state.kind,state.ready,state.input]);
 const current=state.kind===kind;return{input:current?state.input:quantumVariationalPresetInput(kind,0),ready:current&&state.ready,storage:state.storage,setInput:(next:QuantumVariationalInput)=>{if(!isQuantumVariationalInput(next)||next.kind!==kind)return;setState(s=>s.kind===kind?{...s,input:snapshotQuantumVariationalInput(next)}:s);}};
}
