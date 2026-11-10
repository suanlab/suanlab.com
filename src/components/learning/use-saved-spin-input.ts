'use client';
import {useEffect,useState} from 'react';
import {spinPresetInput,type SpinKind,type SpinInput} from '@/lib/learning-spin';
import {parseSpinStoredInput,snapshotSpinInput} from '@/lib/learning-spin-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedSpinInput(kind:SpinKind){
 const [input,setInput]=useState<SpinInput>(()=>spinPresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-spin-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseSpinStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotSpinInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
