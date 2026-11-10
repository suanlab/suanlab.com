import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {LabPalette} from './lab-palette';
export type BosonLadderGeometry={cutoffs:readonly number[];occupations:readonly number[]|null;mode:number|null};
/** Bound includes level labels and the mode captions; no animation loop is needed. */
export function bosonLadderCameraDistance(aspect:number){const half=Math.atan(Math.tan(42*Math.PI/360)*Math.min(1,aspect));return Math.max(2.5,3.2/Math.max(1,aspect))/Math.sin(half);}
export function createBosonLadderRenderer(host:HTMLElement,initial:BosonLadderGeometry,palette:LabPalette,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);
 const canvas=renderer.domElement;canvas.setAttribute('aria-hidden','true');canvas.style.cssText='width:100%;height:100%;display:block;touch-action:pan-y';
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.01,200),controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;
 let group=new THREE.Group(),disposed=false;scene.add(group);
 const render=()=>{if(!disposed)renderer.render(scene,camera);};
 function release(){group.traverse(node=>{const mesh=node as THREE.Mesh;mesh.geometry?.dispose();for(const material of Array.isArray(mesh.material)?mesh.material:mesh.material?[mesh.material]:[]){(material as THREE.SpriteMaterial).map?.dispose();material.dispose();}});scene.remove(group);}
 function fit(){camera.position.normalize().multiplyScalar(bosonLadderCameraDistance(camera.aspect));camera.updateProjectionMatrix();}
 function label(text:string,x:number,y:number,color:string,size=1){const c=document.createElement('canvas');c.width=256;c.height=64;const ctx=c.getContext('2d');if(!ctx)return;ctx.font='bold 30px sans-serif';ctx.fillStyle=color;ctx.textAlign='center';ctx.fillText(text,128,43);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));sprite.position.set(x,y,0);sprite.scale.set(2*size,.5*size,1);group.add(sprite);}
 function view(mode:'perspective'|'front'|'top'){camera.position.set(...(mode==='front'?[0,.001,1]:mode==='top'?[0,1,.001]:[1.5,1.3,4])as[number,number,number]);fit();controls.target.set(0,0,0);controls.update();render();}
 function update(g:BosonLadderGeometry,p:LabPalette){if(disposed)return;release();group=new THREE.Group();scene.add(group);
  for(let mode=0;mode<g.cutoffs.length;mode++){
   const x=(mode-(g.cutoffs.length-1)/2)*2.7;
   for(let n=0;n<=g.cutoffs[mode];n++){
    const y=n*.85-1.2,line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x-.6,y,0),new THREE.Vector3(x+.6,y,0)]),new THREE.LineBasicMaterial({color:n===g.cutoffs[mode]?p.orange:p.neutral}));group.add(line);label(String(n),x-.85,y,p.label,.75);
    if(g.occupations?.[mode]===n){const marker=new THREE.Mesh(new THREE.SphereGeometry(.15,16,12),new THREE.MeshBasicMaterial({color:g.mode===mode?p.orange:p.blue}));marker.position.set(x,y,0);group.add(marker);}
   }
   label(`mode ${mode} K=${g.cutoffs[mode]}`,x,-1.85,p.label,1.1);
  }
  if(g.occupations===null)label('0 vector',0,2,p.label,.9);
  render();
 }
 const resize=new ResizeObserver(()=>{if(!disposed&&host.clientWidth&&host.clientHeight){renderer.setSize(host.clientWidth,host.clientHeight,false);camera.aspect=host.clientWidth/host.clientHeight;fit();render();}});
 const lose=(event:Event)=>{event.preventDefault();if(!disposed)onFailure();};
 function dispose(){if(disposed)return;disposed=true;resize.disconnect();controls.removeEventListener('change',render);controls.dispose();canvas.removeEventListener('webglcontextlost',lose);release();scene.clear();renderer.dispose();renderer.forceContextLoss();canvas.remove();}
 try{host.appendChild(canvas);view('perspective');controls.addEventListener('change',render);canvas.addEventListener('webglcontextlost',lose);resize.observe(host);update(initial,palette);}catch(error){dispose();throw error;}
 return{update,view,dispose};
}
export type BosonLadderRenderer=ReturnType<typeof createBosonLadderRenderer>;
