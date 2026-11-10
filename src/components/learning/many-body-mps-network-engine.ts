import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {LabPalette} from './lab-palette';
export type MpsNetworkGeometry={sites:number;bonds:readonly number[];step:number;gaugeBond:number|null;gauged:boolean};
export function mpsNetworkCameraDistance(sites:number,aspect:number){const radius=Math.max(2.7,(sites-1)*1.05+1.2),half=Math.atan(Math.tan(42*Math.PI/360)*Math.min(1,aspect));return Math.max(2.6,radius/Math.max(1,aspect))/Math.sin(half);}
/** Geometry explains contractions; it carries no state amplitudes or dynamics. */
export function createMpsNetworkRenderer(host:HTMLElement,initial:MpsNetworkGeometry,palette:LabPalette,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);const canvas=renderer.domElement;canvas.setAttribute('aria-hidden','true');canvas.style.cssText='width:100%;height:100%;display:block;touch-action:pan-y';
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.01,200),controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;let group=new THREE.Group(),disposed=false,sites=initial.sites;scene.add(group);
 const render=()=>{if(!disposed)renderer.render(scene,camera);};
 function release(){group.traverse(node=>{const mesh=node as THREE.Mesh;mesh.geometry?.dispose();for(const m of Array.isArray(mesh.material)?mesh.material:mesh.material?[mesh.material]:[]){(m as THREE.SpriteMaterial).map?.dispose();m.dispose();}});scene.remove(group);}
 function fit(){camera.position.normalize().multiplyScalar(mpsNetworkCameraDistance(sites,camera.aspect));camera.updateProjectionMatrix();}
 function label(text:string,x:number,y:number,color:string,size=1){const c=document.createElement('canvas');c.width=256;c.height=64;const ctx=c.getContext('2d');if(!ctx)return;ctx.font='bold 30px sans-serif';ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(text,128,43);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));sprite.position.set(x,y,0);sprite.scale.set(2*size,.5*size,1);group.add(sprite);}
 function line(x1:number,y1:number,x2:number,y2:number,color:string){group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x1,y1,0),new THREE.Vector3(x2,y2,0)]),new THREE.LineBasicMaterial({color})));}
 function view(mode:'perspective'|'front'|'top'){camera.position.set(...(mode==='front'?[0,.001,1]:mode==='top'?[0,1,.001]:[1.6,1.6,4])as[number,number,number]);fit();controls.target.set(0,0,0);controls.update();render();}
 function update(g:MpsNetworkGeometry,p:LabPalette){if(disposed)return;sites=g.sites;release();group=new THREE.Group();scene.add(group);fit();
  for(let i=0;i<sites;i++){
   const x=(i-(sites-1)/2)*2.1,box=new THREE.Mesh(new THREE.BoxGeometry(.75,.6,.5),new THREE.MeshBasicMaterial({color:i<g.step?p.blue:p.neutral,wireframe:i>=g.step}));box.position.set(x,0,0);group.add(box);label(`A${i}`,x,.6,p.label,1.05);line(x,-.3,x,-1.05,p.neutral);label(`s${i}=0/1`,x,-1.4,p.label,1.1);
   if(i<sites-1){line(x+.4,0,x+1.7,0,g.gaugeBond===i?p.orange:p.neutral);label(`D=${g.bonds[i]}`,x+1.05,-.5,p.label,.9);if(g.gaugeBond===i)label(g.gauged?'G G^-1':'gauge bond',x+1.05,1.25,p.orange,1.05);}
  }render();
 }
 const resize=new ResizeObserver(()=>{if(!disposed&&host.clientWidth&&host.clientHeight){renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;fit();render();}}),lose=(event:Event)=>{event.preventDefault();if(!disposed)onFailure();};
 function dispose(){if(disposed)return;disposed=true;resize.disconnect();controls.removeEventListener('change',render);controls.dispose();canvas.removeEventListener('webglcontextlost',lose);release();scene.clear();renderer.dispose();renderer.forceContextLoss();canvas.remove();}
 try{host.appendChild(canvas);view('perspective');controls.addEventListener('change',render);canvas.addEventListener('webglcontextlost',lose);resize.observe(host);update(initial,palette);}catch(error){dispose();throw error;}
 return{update,view,dispose};
}
export type MpsNetworkRenderer=ReturnType<typeof createMpsNetworkRenderer>;
