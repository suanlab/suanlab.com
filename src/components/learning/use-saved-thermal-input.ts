'use client';
import {useEffect,useState}from'react';
import {thermalPresetInput,type ThermalKind,type ThermalInput}from'@/lib/learning-thermal';
import {parseThermalStoredInput,snapshotThermalInput}from'@/lib/learning-thermal-protocol';
/** Read validated original JSON text before writing; model families never share state. */
export function useSavedThermalInput(kind:ThermalKind){
 const[value,setValue]=useState<ThermalInput>(()=>thermalPresetInput(kind,0)),[ready,setReady]=useState(false),[available,setAvailable]=useState(true),key=`suanlab-thermal-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseThermalStoredInput(raw):null;if(saved&&saved.kind===kind)setValue(saved);}catch{setAvailable(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotThermalInput(value)}));}catch{setAvailable(false);}},[key,value,ready]);
 return{input:value,setInput:setValue,ready,storage:available};
}
