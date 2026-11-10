'use client';
import {useEffect,useState} from 'react';
import {observablePresetInput,type ObservableKind,type ObservableInput} from '@/lib/learning-observable';
import {parseObservableStoredInput,snapshotObservableInput} from '@/lib/learning-observable-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedObservableInput(kind:ObservableKind){
 const [input,setInput]=useState<ObservableInput>(()=>observablePresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-observable-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseObservableStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotObservableInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
