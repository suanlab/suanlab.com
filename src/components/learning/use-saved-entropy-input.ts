'use client';
import {useEffect,useState}from'react';
import {entropyPresetInput,type EntropyKind,type EntropyInput}from'@/lib/learning-entropy-production';
import {parseEntropyStoredInput,snapshotEntropyInput}from'@/lib/learning-entropy-production-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedEntropyInput(kind:EntropyKind){
 const[value,setValue]=useState<EntropyInput>(()=>entropyPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-entropy-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseEntropyStoredInput(raw):null;if(saved)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotEntropyInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
