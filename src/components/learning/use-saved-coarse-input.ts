'use client';
import {useEffect,useState} from 'react';
import {coarsePresetInput,type CoarseKind,type CoarseInput} from '@/lib/learning-coarse';
import {parseCoarseStoredInput,snapshotCoarseInput} from '@/lib/learning-coarse-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedCoarseInput(kind:CoarseKind){
 const [input,setInput]=useState<CoarseInput>(()=>coarsePresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-coarse-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseCoarseStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotCoarseInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
