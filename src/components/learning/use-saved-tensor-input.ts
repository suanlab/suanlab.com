'use client';
import {useEffect,useState} from 'react';
import {tensorPresetInput,type TensorKind,type TensorInput} from '@/lib/learning-tensor';
import {parseTensorStoredInput,snapshotTensorInput} from '@/lib/learning-tensor-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedTensorInput(kind:TensorKind){
 const [input,setInput]=useState<TensorInput>(()=>tensorPresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-tensor-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseTensorStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotTensorInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
