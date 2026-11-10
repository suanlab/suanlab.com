'use client';
import {useEffect,useState}from'react';
import {fdtPresetInput,type FdtKind,type FdtInput}from'@/lib/learning-fluctuation-dissipation';
import {parseFdtStoredInput,snapshotFdtInput}from'@/lib/learning-fluctuation-dissipation-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedFdtInput(kind:FdtKind){
 const[value,setValue]=useState<FdtInput>(()=>fdtPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-fdt-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseFdtStoredInput(raw,kind):null;if(saved)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotFdtInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
