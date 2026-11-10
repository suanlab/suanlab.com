'use client';
import {useEffect,useState} from 'react';
import {uncertaintyPresetInput,type UncertaintyKind,type UncertaintyInput} from '@/lib/learning-uncertainty';
import {parseUncertaintyStoredInput,snapshotUncertaintyInput} from '@/lib/learning-uncertainty-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedUncertaintyInput(kind:UncertaintyKind){
 const [input,setInput]=useState<UncertaintyInput>(()=>uncertaintyPresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-uncertainty-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseUncertaintyStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotUncertaintyInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
