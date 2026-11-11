'use client';
import {useLanguage} from '@/components/language-provider';
import type {ConvexResult} from '@/lib/learning-convex';
import MeasureFigure from './measure-figure';
export default function ConvexFigure({result,label,axis}:{result:ConvexResult;label:string;axis:string}){
 const {language}=useLanguage(),ko=language==='ko';
 return <><MeasureFigure result={result} label={label} axis={axis} variant="convex" equalAspect={Boolean(result.spatial)}/>{result.numericNote&&<p className="ai-model-note" role="status">{ko?'수학적으로 유한한 중간값이 현재 숫자 표현 범위를 벗어나 해당 수치를 숨겼습니다. 이를 수학적 무한대나 정의역 위반으로 해석하지 마세요.':'A mathematically finite intermediate lies outside the current numeric representation range and its readout is hidden. This is not mathematical infinity or a domain violation.'}</p>}</>;
}
