'use client';
import {useEffect,useState}from'react';
import {reversalPresetInput,type ReversalKind,type ReversalInput}from'@/lib/learning-cycle-reversal';
import {parseReversalStoredInput,snapshotReversalInput}from'@/lib/learning-cycle-reversal-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedReversalInput(kind:ReversalKind){
 const[value,setValue]=useState<ReversalInput>(()=>reversalPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-reversal-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseReversalStoredInput(raw,kind):null;if(saved)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotReversalInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
