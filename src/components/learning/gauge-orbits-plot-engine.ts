import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {LabPalette} from './lab-palette';
import type {gaugeOrbitView} from '@/lib/learning-gauge-orbits-view';
type View=NonNullable<ReturnType<typeof gaugeOrbitView>>;
/** Event-driven abstract graph and local complex arrows; no physical time evolution. */
export function createGaugeOrbitPlot(host:HTMLElement,initial:View,palette:LabPalette,label:string,onFailure:()=>void){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.75));renderer.setClearColor(0,0);
 const canvas=renderer.domElement;canvas.tabIndex=0;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label);canvas.style.cssText='width:100%;height:100%;display:block;touch-action:pan-y';host.append(canvas);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(42,1,.1,80),controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.enablePan=false;controls.enableZoom=false;let content=new THREE.Group(),disposed=false,labels:THREE.Sprite[]=[],radius=4;scene.add(content);
 const fit=()=>{const half=Math.min(camera.fov*Math.PI/360,Math.atan(Math.tan(camera.fov*Math.PI/360)*camera.aspect)),direction=camera.position.clone().sub(controls.target).normalize();camera.position.copy(controls.target).addScaledVector(direction,1.08*radius/Math.sin(half));controls.update();};
 const render=()=>{if(disposed)return;const height=host.getBoundingClientRect().height||360;for(const sprite of labels){const unit=2*camera.position.distanceTo(sprite.position)*Math.tan(camera.fov*Math.PI/360)/height;sprite.scale.set(51.2*unit,25.6*unit,1);}renderer.render(scene,camera);};
 const release=(group:THREE.Group)=>group.traverse(node=>{const mesh=node as THREE.Mesh;mesh.geometry?.dispose();(Array.isArray(mesh.material)?mesh.material:mesh.material?[mesh.material]:[]).forEach(m=>{(m as THREE.SpriteMaterial).map?.dispose();m.dispose();});});
 function update(model:View,p:LabPalette){
  scene.remove(content);release(content);content=new THREE.Group();labels=[];scene.add(content);radius=Math.max(...model.nodes.map(n=>Math.hypot(n.x,n.z)+.65),...model.edges.flatMap(e=>e.points.map(q=>Math.hypot(q.x,q.z)+.4)))+.3;
  for(const edge of model.edges){const points=edge.points.map(q=>new THREE.Vector3(q.x,q.y,q.z)),color=edge.highlight?p.orange:edge.forest?p.green:p.neutral;content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color})));const at=points[18],next=points[19],dir=next.clone().sub(at).normalize();content.add(new THREE.ArrowHelper(dir,at,.18,color,.12,.08));const origin=points[12],phase=new THREE.Vector3(edge.re!,0,edge.im!);if(phase.length()>0)content.add(new THREE.ArrowHelper(phase.normalize(),origin,.4,p.blue,.1,.06));}
  for(const node of model.nodes){
   const base=new THREE.Vector3(node.x,node.height,node.z),dot=new THREE.Mesh(new THREE.SphereGeometry(.075,12,8),new THREE.MeshBasicMaterial({color:p.blue}));dot.position.copy(base);content.add(dot);
   for(const [enabled,rad,color] of [[node.anchor,.18,p.orange],[node.root,.25,p.green]] as const){if(enabled){const ring=new THREE.Mesh(new THREE.TorusGeometry(rad,.025,6,28),new THREE.MeshBasicMaterial({color}));ring.rotation.x=Math.PI/2;ring.position.copy(base);content.add(ring);}}
   const tip=base.clone().add(new THREE.Vector3(.65*node.re,0,.65*node.im)),delta=tip.clone().sub(base),length=delta.length();if(length>0)content.add(new THREE.ArrowHelper(delta.normalize(),base,length,p.orange,Math.min(.12,length/3),Math.min(.08,length/4)));
   content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(node.x,0,node.z),base]),new THREE.LineBasicMaterial({color:p.neutral})));
   const text=document.createElement('canvas');text.width=128;text.height=64;const ctx=text.getContext('2d');if(ctx){ctx.fillStyle=p.label;ctx.font='40px sans-serif';ctx.textAlign='center';ctx.fillText(`${node.site}${node.root?'r':''}`,64,46);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(text),transparent:true}));sprite.position.copy(base).add(new THREE.Vector3(0,.35,0));labels.push(sprite);content.add(sprite);}
  }fit();render();
 }
 function view(mode:'perspective'|'front'){camera.position.set(...(mode==='front'?[0,7,10]:[7,7,9])as[number,number,number]);controls.target.set(0,.5,0);fit();render();}
 const key=(event:KeyboardEvent)=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(event.key))return;event.preventDefault();if(event.key==='Home'){view('perspective');return;}const s=new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));s.theta+=(event.key==='ArrowLeft'?1:event.key==='ArrowRight'?-1:0)*Math.PI/18;s.phi=Math.max(.1,Math.min(Math.PI-.1,s.phi+(event.key==='ArrowUp'?-1:event.key==='ArrowDown'?1:0)*Math.PI/18));camera.position.setFromSpherical(s).add(controls.target);controls.update();render();};
 canvas.addEventListener('keydown',key);controls.addEventListener('change',render);view('perspective');
 const resize=new ResizeObserver(()=>{const{width,height}=host.getBoundingClientRect();if(disposed||!width||!height)return;camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);fit();render();});resize.observe(host);
 const lost=(event:Event)=>{event.preventDefault();onFailure();};canvas.addEventListener('webglcontextlost',lost);update(initial,palette);
 return{update,view,dispose(){if(disposed)return;disposed=true;resize.disconnect();canvas.removeEventListener('keydown',key);canvas.removeEventListener('webglcontextlost',lost);controls.removeEventListener('change',render);controls.dispose();release(content);renderer.dispose();renderer.forceContextLoss();canvas.remove();}};
}
export type GaugeOrbitPlotRenderer=ReturnType<typeof createGaugeOrbitPlot>;
