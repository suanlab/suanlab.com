'use client';
import {useEffect,useState} from 'react';
import {unitaryPresetInput,type UnitaryKind,type UnitaryInput} from '@/lib/learning-unitary';
import {parseUnitaryStoredInput,snapshotUnitaryInput} from '@/lib/learning-unitary-protocol';
/** Keyed callers mount one family at a time. Restore before the first write. */
export function useSavedUnitaryInput(kind:UnitaryKind){
 const [input,setInput]=useState<UnitaryInput>(()=>unitaryPresetInput(kind,0)),[ready,setReady]=useState(false),[storage,setStorage]=useState(true),key=`suanlab-unitary-${kind}`;
 useEffect(()=>{try{const raw=localStorage.getItem(key),saved=raw?parseUnitaryStoredInput(raw):null;if(saved?.kind===kind)setInput(saved);}catch{setStorage(false);}setReady(true);},[key,kind]);
 useEffect(()=>{if(!ready)return;try{localStorage.setItem(key,JSON.stringify({version:1,value:snapshotUnitaryInput(input)}));}catch{setStorage(false);}},[key,input,ready]);
 return{input,setInput,ready,storage};
}
