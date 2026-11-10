'use client';
import {useEffect,useState} from 'react';
import {synchronizationPresetInput,type SynchronizationKind,type SynchronizationInput} from '@/lib/learning-synchronization';
import {parseSynchronizationStoredInput,snapshotSynchronizationInput} from '@/lib/learning-synchronization-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedSynchronizationInput(kind:SynchronizationKind){
 const [input,setInput]=useState<SynchronizationInput>(()=>synchronizationPresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-synchronization-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseSynchronizationStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotSynchronizationInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
