'use client';
import {useEffect,useState}from'react';
import {fokkerPresetInput,type FokkerKind,type FokkerInput}from'@/lib/learning-fokker-planck';
import {parseFokkerStoredInput,snapshotFokkerInput}from'@/lib/learning-fokker-planck-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedFokkerInput(kind:FokkerKind){
 const[value,setValue]=useState<FokkerInput>(()=>fokkerPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-fokker-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseFokkerStoredInput(raw,kind):null;if(saved)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotFokkerInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
