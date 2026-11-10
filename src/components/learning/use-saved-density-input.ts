'use client';
import {useEffect,useState} from 'react';
import {densityPresetInput,type DensityKind,type DensityInput} from '@/lib/learning-density';
import {parseDensityStoredInput,snapshotDensityInput} from '@/lib/learning-density-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedDensityInput(kind:DensityKind){
 const [input,setInput]=useState<DensityInput>(()=>densityPresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-density-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseDensityStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotDensityInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
