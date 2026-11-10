import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {LabPalette} from './lab-palette';
export type SchmidtSpectrumGeometry={selected:number;groups:{weight:number|null;label:string;multiplicity:number;status:'kept'|'discarded'|'split'|'zero'}[]};
export function schmidtSpectrumCameraDistance(groups:number,aspect:number){const radius=Math.max(3.3,(groups-1)*1.05+1.2),half=Math.atan(Math.tan(42*Math.PI/360)*Math.min(1,aspect));return Math.max(3,radius/Math.max(1,aspect))/Math.sin(half);}
/** One column denotes a distinct weight,not an arbitrarily chosen eigenvector. */
export function createSchmidtSpectrumRenderer(host:HTMLElement,initial:SchmidtSpectrumGeometry,palette:LabPalette,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);const canvas=renderer.domElement;canvas.setAttribute('aria-hidden','true');canvas.style.cssText='width:100%;height:100%;display:block;touch-action:pan-y';
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.01,200),controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;let group=new THREE.Group(),disposed=false,count=initial.groups.length;scene.add(group);
 const render=()=>{if(!disposed)renderer.render(scene,camera);};
 function release(){group.traverse(node=>{const mesh=node as THREE.Mesh;mesh.geometry?.dispose();for(const m of Array.isArray(mesh.material)?mesh.material:mesh.material?[mesh.material]:[]){(m as THREE.SpriteMaterial).map?.dispose();m.dispose();}});scene.remove(group);}
 function fit(){camera.position.normalize().multiplyScalar(schmidtSpectrumCameraDistance(count,camera.aspect));camera.updateProjectionMatrix();}
 function label(text:string,x:number,y:number,color:string,size=1){const c=document.createElement('canvas');c.width=256;c.height=64;const ctx=c.getContext('2d');if(!ctx)return;ctx.font='bold 30px sans-serif';ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(text,128,43);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));sprite.position.set(x,y,0);sprite.scale.set(2*size,.5*size,1);group.add(sprite);}
 function view(mode:'perspective'|'front'|'top'){camera.position.set(...(mode==='front'?[0,.001,1]:mode==='top'?[0,1,.001]:[1.7,1.6,4])as[number,number,number]);fit();controls.target.set(0,0,0);controls.update();render();}
 function update(g:SchmidtSpectrumGeometry,p:LabPalette){if(disposed)return;count=g.groups.length;release();group=new THREE.Group();scene.add(group);fit();g.groups.forEach((v,i)=>{const x=(i-(count-1)/2)*2.1,h=Math.max(0,Math.min(1,v.weight??0))*3.2,color=v.status==='kept'?p.blue:v.status==='split'?p.green:v.status==='zero'?p.neutral:p.orange;
  const base=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x-.6,-1.4,0),new THREE.Vector3(x+.6,-1.4,0)]),new THREE.LineBasicMaterial({color:i===g.selected?p.label:p.neutral}));group.add(base);
  if(h>0){const bar=new THREE.Mesh(new THREE.BoxGeometry(.7,h,.45),new THREE.MeshBasicMaterial({color}));bar.position.set(x,-1.4+h/2,0);group.add(bar);}
  label(v.label,x,2.2,p.label,.95);label(`g${i} x${v.multiplicity}`,x,-1.95,p.label,1.05);
 });render();}
 const resize=new ResizeObserver(()=>{if(!disposed&&host.clientWidth&&host.clientHeight){renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;fit();render();}}),lose=(event:Event)=>{event.preventDefault();if(!disposed)onFailure();};
 function dispose(){if(disposed)return;disposed=true;resize.disconnect();controls.removeEventListener('change',render);controls.dispose();canvas.removeEventListener('webglcontextlost',lose);release();scene.clear();renderer.dispose();renderer.forceContextLoss();canvas.remove();}
 try{host.appendChild(canvas);view('perspective');controls.addEventListener('change',render);canvas.addEventListener('webglcontextlost',lose);resize.observe(host);update(initial,palette);}catch(error){dispose();throw error;}
 return{update,view,dispose};
}
export type SchmidtSpectrumRenderer=ReturnType<typeof createSchmidtSpectrumRenderer>;
