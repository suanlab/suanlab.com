import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { LabPalette } from './lab-palette';
export type SpinGraphGeometry = { sites: { index:number;position:[number,number,number];bloch:[number|null,number|null,number|null] }[]; edges:{from:number;to:number;active:boolean}[]; selected:number };
export function spinGraphCameraDistance(radius:number,aspect:number){const half=Math.atan(Math.tan(42*Math.PI/360)*Math.min(1,aspect));return Math.max(3,radius*1.12/Math.sin(half));}
/** Event-driven camera and local expectation arrows; no physical-time loop. */
export function createSpinGraphRenderer(host:HTMLElement,initial:SpinGraphGeometry,palette:LabPalette,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);
 const canvas=renderer.domElement;canvas.setAttribute('aria-hidden','true');canvas.style.cssText='width:100%;height:100%;display:block;touch-action:pan-y';
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.01,200),controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;
 let group=new THREE.Group(),disposed=false,radius=3;const render=()=>{if(!disposed)renderer.render(scene,camera);};
 function release(){group.traverse(node=>{const mesh=node as THREE.Mesh;mesh.geometry?.dispose();for(const m of Array.isArray(mesh.material)?mesh.material:mesh.material?[mesh.material]:[]){(m as THREE.SpriteMaterial).map?.dispose();m.dispose();}});scene.remove(group);}
 function fit(){camera.position.normalize().multiplyScalar(spinGraphCameraDistance(radius,camera.aspect));camera.far=Math.max(200,camera.position.length()+radius*4);camera.updateProjectionMatrix();}
 function view(mode:'perspective'|'front'|'top'){camera.position.set(...(mode==='front'?[0,.001,1]:mode==='top'?[0,1,.001]:[2,2.3,3]) as [number,number,number]);fit();controls.target.set(0,0,0);controls.update();render();}
 function update(g:SpinGraphGeometry,p:LabPalette){
  if(disposed)return;release();group=new THREE.Group();scene.add(group);radius=Math.max(1.6,...g.sites.map(s=>Math.hypot(...s.position)+1.4));fit();
  for(const e of g.edges){group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...g.sites[e.from].position),new THREE.Vector3(...g.sites[e.to].position)]),new THREE.LineBasicMaterial({color:e.active?p.orange:p.neutral,transparent:true,opacity:e.active?1:.3})));}
  for(const s of g.sites){
   const center=new THREE.Vector3(...s.position),selected=s.index===g.selected;
   const sphere=new THREE.Mesh(new THREE.SphereGeometry(1,20,12),new THREE.MeshBasicMaterial({color:selected?p.orange:p.neutral,wireframe:true,transparent:true,opacity:selected ? 0.35 : 0.15}));sphere.position.copy(center);group.add(sphere);
   const dot=new THREE.Mesh(new THREE.SphereGeometry(.07,10,8),new THREE.MeshBasicMaterial({color:p.label}));dot.position.copy(center);group.add(dot);
   if(s.bloch.every(v=>v!==null)){const v=new THREE.Vector3(...s.bloch as [number,number,number]),length=v.length();if(length>0)group.add(new THREE.ArrowHelper(v.clone().normalize(),center,length,p.blue,Math.min(.18,length*.25),Math.min(.1,length*.15)));}
   const text=document.createElement('canvas');text.width=128;text.height=64;const ctx=text.getContext('2d');if(ctx){ctx.font='bold 32px sans-serif';ctx.textAlign='center';ctx.fillStyle=p.label;ctx.fillText(String(s.index),64,43);const label=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(text),depthTest:false}));label.position.copy(center).add(new THREE.Vector3(0,1.25,0));label.scale.set(.7,.35,1);group.add(label);}
   if(selected)for(let axis=0;axis<3;axis++){const end=center.clone();end.setComponent(axis,end.getComponent(axis)+1.1);group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([center,end]),new THREE.LineBasicMaterial({color:p.neutral})));const axisText=document.createElement('canvas');axisText.width=64;axisText.height=64;const a=axisText.getContext('2d');if(a){a.fillStyle=p.label;a.font='28px sans-serif';a.fillText(['X','Y','Z'][axis],18,40);const label=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(axisText),depthTest:false}));label.position.copy(end);label.scale.set(.35,.35,1);group.add(label);}}
  }
  render();
 }
 const resize=new ResizeObserver(()=>{if(disposed)return;const w=host.clientWidth,h=host.clientHeight;if(w&&h){renderer.setSize(w,h,false);camera.aspect=w/h;fit();render();}});
 const lost=(event:Event)=>{event.preventDefault();if(!disposed)onFailure();};
 function dispose(){if(disposed)return;disposed=true;resize.disconnect();controls.removeEventListener('change',render);controls.dispose();canvas.removeEventListener('webglcontextlost',lost);release();scene.clear();renderer.dispose();renderer.forceContextLoss();canvas.remove();}
 try{host.appendChild(canvas);view('perspective');controls.addEventListener('change',render);canvas.addEventListener('webglcontextlost',lost);resize.observe(host);update(initial,palette);}catch(e){dispose();throw e;}
 return{update,view,dispose};
}
export type SpinGraphRenderer=ReturnType<typeof createSpinGraphRenderer>;
