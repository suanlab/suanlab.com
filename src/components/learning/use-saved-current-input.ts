'use client';
import {useEffect,useState}from'react';
import {currentPresetInput,type CurrentKind,type CurrentInput}from'@/lib/learning-probability-current';
import {parseCurrentStoredInput,snapshotCurrentInput}from'@/lib/learning-probability-current-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedCurrentInput(kind:CurrentKind){
 const[value,setValue]=useState<CurrentInput>(()=>currentPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-current-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseCurrentStoredInput(raw,kind):null;if(saved)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotCurrentInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
