import type {Vec2} from '@/lib/learning-simulations';
const fmt=(v:number)=>v!==0&&(Math.abs(v)<.001||Math.abs(v)>10000)?v.toExponential(2):Number(v.toPrecision(4)).toString();
export default function MechanicsPlot({curves,dots=[],dotColors,curveColors,breaks=[],roots=[],bounds,title,xLabel,yLabel,rows,marker,arrows=false,xTicks}:{curves:Vec2[][];dots?:Vec2[];dotColors?:string[];curveColors?:string[];breaks?:number[];roots?:Vec2[];bounds:number[];title:string;xLabel:string;yLabel:string;rows:string[];marker:string;arrows?:boolean;xTicks?:{value:number;label:string}[]}){
 const[x0,x1,y0,y1]=bounds,x=(v:number)=>55+240*(v-x0)/(x1-x0),y=(v:number)=>295-240*(v-y0)/(y1-y0),colors=['blue','orange','green'];
 return <svg data-analytical-mechanics-plot={marker} viewBox="0 0 350 560" role="img" aria-label={title}>
 <text x="175" y="25" textAnchor="middle" fill="currentColor" style={{fontSize:11}}>{title}</text><path d="M55 55V295H295" fill="none" stroke="var(--learning-neutral)"/>
 {y0<0&&y1>0&&<path d={`M55 ${y(0)}H295`} stroke="var(--learning-neutral)" strokeOpacity=".45" strokeDasharray="3 4"/>}
 {breaks.map((v,j)=><path key={`break-${j}`} d={`M${x(v)} 55V295`} stroke="var(--learning-neutral)" strokeOpacity=".5" strokeDasharray="2 5"/>)}
 {curves.map((c,i)=>{const color=`var(--learning-${curveColors?.[i]??colors[i%3]})`,last=c[c.length-1],prev=c.length>1?c[c.length-2]:last,dx=last&&prev?x(last[0])-x(prev[0]):0,dy=last&&prev?y(last[1])-y(prev[1]):0,norm=Math.hypot(dx,dy),ux=norm?dx/norm:0,uy=norm?dy/norm:0;return <g key={i}>{c.length===1?<circle cx={x(c[0][0])} cy={y(c[0][1])} r="3" fill={color}/>:<path d={c.map(([a,b],j)=>`${j?'L':'M'}${x(a)} ${y(b)}`).join(' ')} fill="none" stroke={color} strokeWidth={(curveColors?curveColors[i]!=='blue':i>0)?2:3} strokeDasharray={(curveColors?curveColors[i]!=='blue':i>0)?'5 4':undefined}/>} {arrows&&last&&(norm>0?<polygon points={`${x(last[0])},${y(last[1])} ${x(last[0])-8*ux+3*uy},${y(last[1])-8*uy-3*ux} ${x(last[0])-8*ux-3*uy},${y(last[1])-8*uy+3*ux}`} fill={color}/>:<circle cx={x(last[0])} cy={y(last[1])} r="3" fill={color}/>)}</g>;})}
 {dots.map(([a,b],j)=><circle key={j} cx={x(a)} cy={y(b)} r="4" fill={`var(--learning-${dotColors?.[j]??(j?'green':'orange')})`}/>)}
 {roots.map(([a,b],j)=><circle key={j} cx={x(a)} cy={y(b)} r="6" fill="none" stroke="currentColor" strokeWidth="1.5"/>)}
 {[y0,(y0+y1)/2,y1].map(v=><text key={v} x="48" y={y(v)+3} textAnchor="end" fill="currentColor" style={{fontSize:9}}>{fmt(v)}</text>)}
 {xTicks?xTicks.map((tick,j)=><text key={j} x={x(tick.value)} y="315" textAnchor="middle" fill="currentColor" style={{fontSize:9}}>{tick.label}</text>):<><text x="55" y="315" fill="currentColor" style={{fontSize:9}}>{fmt(x0)}</text><text x="295" y="315" textAnchor="end" fill="currentColor" style={{fontSize:9}}>{fmt(x1)}</text></>}<text x="175" y={xTicks?336:315} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{xLabel}</text><text x="175" y={xTicks?357:337} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{yLabel}</text>
 {rows.map((v,j)=><text key={j} x="175" y={(xTicks?386:365)+26*j} textAnchor="middle" fill="currentColor" style={{fontSize:10}}>{v}</text>)}
 </svg>;
}
