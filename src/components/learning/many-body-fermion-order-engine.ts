import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {LabPalette} from './lab-palette';
export type FermionOrderGeometry={modes:number;bits:number[]|null;mode:number|null;sign:number};
export function fermionOrderCameraDistance(radius:number,aspect:number){const half=Math.atan(Math.tan(42*Math.PI/360)*Math.min(1,aspect));return Math.max(3,radius*1.12/Math.sin(half));}
/** Event-driven occupation diagram. A zero vector is never drawn as vacuum bits. */
export function createFermionOrderRenderer(host:HTMLElement,initial:FermionOrderGeometry,palette:LabPalette,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);const canvas=renderer.domElement;canvas.setAttribute('aria-hidden','true');canvas.style.cssText='width:100%;height:100%;display:block;touch-action:pan-y';
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.01,200),controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;
 let group=new THREE.Group(),radius=3,disposed=false;const render=()=>{if(!disposed)renderer.render(scene,camera);};
 function release(){group.traverse(node=>{const m=node as THREE.Mesh;m.geometry?.dispose();for(const a of Array.isArray(m.material)?m.material:m.material?[m.material]:[]){(a as THREE.SpriteMaterial).map?.dispose();a.dispose();}});scene.remove(group);}
 function fit(){camera.position.normalize().multiplyScalar(fermionOrderCameraDistance(Math.max(1.8,radius/Math.max(1,camera.aspect)),camera.aspect));camera.far=Math.max(200,camera.position.length()+radius*4);camera.updateProjectionMatrix();}
 function label(text:string,position:THREE.Vector3,color:string,size=1){const c=document.createElement('canvas');c.width=256;c.height=64;const ctx=c.getContext('2d');if(!ctx)return;ctx.font='bold 30px sans-serif';ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(text,128,43);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));sprite.position.copy(position);sprite.scale.set(2*size,.5*size,1);group.add(sprite);}
 function view(mode:'perspective'|'front'|'top'){camera.position.set(...(mode==='front'?[0,.001,1]:mode==='top'?[0,1,.001]:[1.5,1.8,4]) as [number,number,number]);fit();controls.target.set(0,0,0);controls.update();render();}
 function update(g:FermionOrderGeometry,p:LabPalette){if(disposed)return;release();group=new THREE.Group();scene.add(group);radius=Math.max(2,(g.modes-1)*.85+1);fit();
  if(g.bits===null){label('0 vector',new THREE.Vector3(),p.label,1.2);render();return;}
  for(let i=0;i<g.modes;i++){const x=(i-(g.modes-1)/2)*1.7,center=new THREE.Vector3(x,0,0),occupied=g.bits[i]===1;
   const cube=new THREE.Mesh(new THREE.BoxGeometry(.8,.8,.8),new THREE.MeshBasicMaterial({color:occupied?p.blue:p.neutral,wireframe:!occupied,transparent:true,opacity:occupied ? 0.8 : 0.6}));cube.position.copy(center);group.add(cube);
   if(i===g.mode){const frame=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial({color:p.orange}));frame.position.copy(center);group.add(frame);}
   if(i>0)group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x-1.3,0,0),new THREE.Vector3(x-.4,0,0)]),new THREE.LineBasicMaterial({color:g.mode!==null&&i<=g.mode?p.orange:p.neutral})));
   label(String(i),new THREE.Vector3(x,.85,0),p.label,.8);label(String(g.bits[i]),new THREE.Vector3(x,-.8,0),p.label,.8);
  }
  label(g.sign<0?'phase -1':'phase +1',new THREE.Vector3(0,-1.5,0),p.label,.8);render();
 }
 const resize=new ResizeObserver(()=>{if(!disposed&&host.clientWidth&&host.clientHeight){renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;fit();render();}});
 const lose=(e:Event)=>{e.preventDefault();if(!disposed)onFailure();};
 function dispose(){if(disposed)return;disposed=true;resize.disconnect();controls.removeEventListener('change',render);controls.dispose();canvas.removeEventListener('webglcontextlost',lose);release();scene.clear();renderer.dispose();renderer.forceContextLoss();canvas.remove();}
 try{host.appendChild(canvas);view('perspective');controls.addEventListener('change',render);canvas.addEventListener('webglcontextlost',lose);resize.observe(host);update(initial,palette);}catch(e){dispose();throw e;}
 return{update,view,dispose};
}
export type FermionOrderRenderer=ReturnType<typeof createFermionOrderRenderer>;
