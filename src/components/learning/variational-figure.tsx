'use client';
import type {VariationalResult} from '@/lib/learning-variational';
import MeasureFigure from './measure-figure';
export default function VariationalFigure({result,label,axis}:{result:VariationalResult;label:string;axis:string}){
 return <MeasureFigure result={result} label={label} axis={axis} variant="variational" equalAspect={Boolean(result.spatial)} wideLabels={Boolean(result.wideLabels)}/>;
}
