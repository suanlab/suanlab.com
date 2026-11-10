'use client';
import {useEffect,useState}from'react';
import {masterPresetInput,type MasterKind,type MasterInput}from'@/lib/learning-master-equation';
import {parseMasterStoredInput,snapshotMasterInput}from'@/lib/learning-master-equation-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedMasterInput(kind:MasterKind){
 const[value,setValue]=useState<MasterInput>(()=>masterPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-master-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseMasterStoredInput(raw,kind):null;if(saved)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotMasterInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
