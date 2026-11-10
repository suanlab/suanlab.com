import * as THREE from 'three';
import {OrbitControls}from'three/addons/controls/OrbitControls.js';
import type{LabPalette}from'./lab-palette';
/** Event-driven camera only; no idle loop or simulated quantum dynamics. */
export function createBlochRenderer(host:HTMLElement,initial:number[],palette:LabPalette,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);renderer.domElement.setAttribute('aria-hidden','true');renderer.domElement.style.touchAction='pan-y';host.appendChild(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.1,30),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;
 let content=new THREE.Group(),disposed=false;scene.add(content);const render=()=>{if(!disposed)renderer.render(scene,camera);};
 function release(group:THREE.Group){group.traverse(node=>{const mesh=node as THREE.Mesh;mesh.geometry?.dispose();const materials=Array.isArray(mesh.material)?mesh.material:mesh.material?[mesh.material]:[];materials.forEach(m=>{(m as THREE.SpriteMaterial).map?.dispose();m.dispose();});});}
 const position=(v:number[])=>new THREE.Vector3(v[0],v[2],-v[1]);
 function update(vector:number[],p:LabPalette){scene.remove(content);release(content);content=new THREE.Group();scene.add(content);content.add(new THREE.Mesh(new THREE.SphereGeometry(1,24,16),new THREE.MeshBasicMaterial({color:p.neutral,wireframe:true,transparent:true,opacity:.25})));
  [[1.3,0,0],[0,1.3,0],[0,0,1.3]].forEach((v,i)=>{const end=position(v);content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([position(v.map(x=>-x)),end]),new THREE.LineBasicMaterial({color:p.neutral})));const canvas=document.createElement('canvas');canvas.width=96;canvas.height=64;const context=canvas.getContext('2d');if(context){context.font='bold 36px monospace';context.textAlign='center';context.fillStyle=p.label;context.fillText(['x','y','z'][i],48,44);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(canvas),depthTest:false}));sprite.position.copy(end.multiplyScalar(1.1));sprite.scale.set(.6,.4,1);content.add(sprite);}});
  const end=position(vector),length=end.length();if(length>0)content.add(new THREE.ArrowHelper(end.clone().normalize(),new THREE.Vector3(),length,p.blue,.12,.06));const point=new THREE.Mesh(new THREE.SphereGeometry(.045,16,12),new THREE.MeshBasicMaterial({color:p.blue}));point.position.copy(end);content.add(point);render();
 }
 function view(mode:'perspective'|'front'|'top'){camera.position.set(...(mode==='front'?[0,0,4.2]:mode==='top'?[0,4.2,.01]:[2.7,2.1,3.2])as[number,number,number]);controls.target.set(0,0,0);controls.update();render();}
 view('perspective');controls.addEventListener('change',render);const resize=new ResizeObserver(()=>{const{width,height}=host.getBoundingClientRect();if(disposed||!width||!height)return;camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);render();});resize.observe(host);
 const lose=(event:Event)=>{event.preventDefault();onFailure();};renderer.domElement.addEventListener('webglcontextlost',lose);update(initial,palette);
 return{update,view,dispose(){if(disposed)return;disposed=true;resize.disconnect();controls.removeEventListener('change',render);controls.dispose();renderer.domElement.removeEventListener('webglcontextlost',lose);release(content);renderer.dispose();renderer.domElement.remove();}};
}
export type BlochRenderer=ReturnType<typeof createBlochRenderer>;
