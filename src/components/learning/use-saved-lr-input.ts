'use client';
import {useEffect,useState}from'react';
import {lrPresetInput,type LrKind,type LrInput}from'@/lib/learning-linear-response';
import {parseLrStoredInput,snapshotLrInput}from'@/lib/learning-linear-response-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedLrInput(kind:LrKind){
 const[value,setValue]=useState<LrInput>(()=>lrPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-lr-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseLrStoredInput(raw,kind):null;if(saved)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotLrInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
