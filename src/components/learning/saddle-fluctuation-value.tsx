import React from 'react';
import type {SaddleRange} from '@/lib/learning-saddle-fluctuation';
import {saddleProjection} from '@/lib/learning-saddle-fluctuation-view';
export const saddleNumber=(x:number)=>x===0?'0':x.toPrecision(6);
export function saddleRangeSummary(value:SaddleRange) {
 const a=saddleProjection(value.lower),b=saddleProjection(value.upper);
 if(value.lower===value.upper&&value.lower==='0')return '0';
 if(a===null||b===null)return '[…]';
 return `[≈${saddleNumber(a)}, ≈${saddleNumber(b)}]`;
}
export function SaddleValue({value,language}:{value:string|null;language:'ko'|'en'}) {
 if(value===null)return <span>{language==='ko'?'해당 없음':'Not applicable'}</span>;
 if(value.length<30)return <code style={{overflowWrap:'anywhere'}}>{value}</code>;
 const approximate=saddleProjection(value);
 return <details><summary>{approximate===null?(language==='ko'?'원시 분수 보기':'Show raw fraction'):`≈${saddleNumber(approximate)} · ${language==='ko'?'원시 분수':'raw fraction'}`}</summary><code style={{overflowWrap:'anywhere',whiteSpace:'normal'}}>{value}</code></details>;
}
export function SaddleInterval({value,language}:{value:SaddleRange;language:'ko'|'en'}) {
 return <details><summary>{saddleRangeSummary(value)} · {language==='ko'?'인증 끝점':'certified endpoints'}</summary><dl><dt>{language==='ko'?'하계':'Lower bound'}</dt><dd style={{overflowWrap:'anywhere'}}><code>{value.lower}</code></dd><dt>{language==='ko'?'상계':'Upper bound'}</dt><dd style={{overflowWrap:'anywhere'}}><code>{value.upper}</code></dd></dl><p>{language==='ko'?'소수 끝점이 같아 보여도 원시 분수 구간의 폭이0이라는 뜻은 아닙니다.':'Identical-looking decimal endpoints do not imply a zero-width raw fractional interval.'}</p></details>;
}
