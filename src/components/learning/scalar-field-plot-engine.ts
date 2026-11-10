import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {LabPalette} from './lab-palette';
import type {scalarFieldView} from '@/lib/learning-scalar-field-view';
type View=NonNullable<ReturnType<typeof scalarFieldView>>;
/** A spatial profile in perspective, with event-driven rendering and no time evolution. */
export function createScalarFieldPlot(host:HTMLElement,initial:View,palette:LabPalette,label:string,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);const canvas=renderer.domElement;canvas.tabIndex=0;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label);canvas.style.touchAction='pan-y';canvas.style.width='100%';canvas.style.height='100%';canvas.style.display='block';host.append(canvas);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.1,60),controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;
 let content=new THREE.Group(),disposed=false;scene.add(content);
 const render=()=>{if(!disposed)renderer.render(scene,camera);};
 const release=(group:THREE.Group)=>group.traverse(node=>{const mesh=node as THREE.Mesh;mesh.geometry?.dispose();(Array.isArray(mesh.material)?mesh.material:mesh.material?[mesh.material]:[]).forEach(m=>m.dispose());});
 function update(model:View,p:LabPalette){scene.remove(content);release(content);content=new THREE.Group();scene.add(content);
  const position=(x:number,y:number)=>new THREE.Vector3(4*(x-.5),1.4*y,0);
  content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([position(0,0),position(1,0)]),new THREE.LineBasicMaterial({color:p.neutral})));
  content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(model.curve.map(v=>position(v.x,v.y))),new THREE.LineBasicMaterial({color:p.blue})));
  model.points.forEach(point=>{content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([position(point.x,0),position(point.x,point.y)]),new THREE.LineBasicMaterial({color:p.orange})));const dot=new THREE.Mesh(new THREE.SphereGeometry(.045,12,8),new THREE.MeshBasicMaterial({color:p.blue}));dot.position.copy(position(point.x,point.y));content.add(dot);});render();
 }
 function view(mode:'perspective'|'front'){camera.position.set(...(mode==='front'?[0,0,9]:[4,2.5,8])as[number,number,number]);controls.target.set(0,0,0);controls.update();render();}
 const key=(event:KeyboardEvent)=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(event.key))return;event.preventDefault();if(event.key==='Home'){view('perspective');return;}const s=new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));s.theta+=(event.key==='ArrowLeft'?1:event.key==='ArrowRight'?-1:0)*Math.PI/18;s.phi=Math.max(.1,Math.min(Math.PI-.1,s.phi+(event.key==='ArrowUp'?-1:event.key==='ArrowDown'?1:0)*Math.PI/18));camera.position.setFromSpherical(s).add(controls.target);controls.update();render();};
 canvas.addEventListener('keydown',key);controls.addEventListener('change',render);view('perspective');
 const resize=new ResizeObserver(()=>{const{width,height}=host.getBoundingClientRect();if(disposed||!width||!height)return;camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);render();});resize.observe(host);
 const lost=(event:Event)=>{event.preventDefault();onFailure();};canvas.addEventListener('webglcontextlost',lost);update(initial,palette);
 return{update,view,dispose(){if(disposed)return;disposed=true;resize.disconnect();canvas.removeEventListener('keydown',key);canvas.removeEventListener('webglcontextlost',lost);controls.removeEventListener('change',render);controls.dispose();release(content);renderer.dispose();renderer.forceContextLoss();canvas.remove();}};
}
export type ScalarFieldPlotRenderer=ReturnType<typeof createScalarFieldPlot>;
