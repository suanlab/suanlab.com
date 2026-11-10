'use client';
import {useEffect,useState} from 'react';
import {hilbertPresetInput,type HilbertKind,type HilbertInput} from '@/lib/learning-hilbert';
import {parseHilbertStoredInput,snapshotHilbertInput} from '@/lib/learning-hilbert-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedHilbertInput(kind:HilbertKind){
 const [input,setInput]=useState<HilbertInput>(()=>hilbertPresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-hilbert-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseHilbertStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotHilbertInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
