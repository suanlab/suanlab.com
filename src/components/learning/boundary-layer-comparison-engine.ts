import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { BoundaryLayerView } from '@/lib/learning-boundary-layer-view';
import type { LabPalette } from './lab-palette';

/** Event-driven comparison ribbons; depth labels solution methods, not space or time. */
export function createBoundaryComparisonPlot(host: HTMLElement, initial: BoundaryLayerView, palette: LabPalette, label: string, onFailure: () => void) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.75)); renderer.setClearColor(0, 0);
  const canvas = renderer.domElement; canvas.tabIndex = 0; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', label);
  canvas.style.cssText = 'display:block;width:100%;height:100%'; host.append(canvas);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(42, 1, .1, 100), controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false; controls.enablePan = false; controls.enableZoom = false; canvas.style.touchAction = 'pan-y';
  let content = new THREE.Group(), disposed = false; scene.add(content);
  const render = () => { if (!disposed) renderer.render(scene, camera); };
  let fitting=false;
  const fit=()=>{
    if(fitting)return;fitting=true;
    const half=Math.min(camera.fov*Math.PI/360,Math.atan(Math.tan(camera.fov*Math.PI/360)*camera.aspect));camera.position.normalize().multiplyScalar(4.8/Math.sin(half));
    const box=new THREE.Box3().setFromObject(content);
    if(!box.isEmpty())for(let i=0;i<5;i++){
      camera.updateMatrixWorld();let extent=0;
      for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z]){const p=new THREE.Vector3(x,y,z).project(camera);extent=Math.max(extent,Math.abs(p.x),Math.abs(p.y));}
      if(!Number.isFinite(extent)||extent===0)break;
      camera.position.multiplyScalar(Math.max(.5,Math.min(2,extent/.82)));
      if(camera.position.length()<4.4)camera.position.setLength(4.4);
    }
    controls.update();fitting=false;
  };
  const release = (group: THREE.Group) => group.traverse(node => {
    const mesh = node as THREE.Mesh; mesh.geometry?.dispose();
    (Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : []).forEach(material => material.dispose());
  });
  function update(model: BoundaryLayerView, p: LabPalette) {
    scene.remove(content); release(content); content = new THREE.Group(); scene.add(content);
    const point=(x:number,value:number,j:number)=>new THREE.Vector3(6*x/model.xMax-3,3*(model.transform(value)-model.yMin)/(model.yMax-model.yMin)-1.5,model.kinds.length===1?0:4*j/(model.kinds.length-1)-2);
    const line=(points:THREE.Vector3[],color:string)=>content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color})));
    const colors={exact:p.blue,outer:p.orange,composite:p.green,inner:p.neutral};
    model.kinds.forEach((kind,j)=>line(model.points.map(v=>point(v.coordinate,v[kind]!,j)),colors[kind]));
    if(model.kinds.length>1)for(let i=0;i<model.points.length;i+=4)line(model.kinds.map((kind,j)=>point(model.points[i].coordinate,model.points[i][kind]!,j)),p.neutral);
    const depth=model.kinds.length===1?0:-2;
    line([new THREE.Vector3(-3,-1.5,depth),new THREE.Vector3(3,-1.5,depth)],p.blue);
    line([new THREE.Vector3(-3,-1.5,depth),new THREE.Vector3(-3,1.5,depth)],p.neutral);
    fit(); render();
  }
  function view(mode: 'perspective' | 'front') { camera.position.set(...(mode === 'front' ? [0, 2, 12] : [7, 6, 9]) as [number, number, number]); controls.target.set(0, 0, 0); fit(); render(); }
  const key = (event: KeyboardEvent) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) return;
    event.preventDefault(); if (event.key === 'Home') { view('perspective'); return; }
    const spherical = new THREE.Spherical().setFromVector3(camera.position);
    spherical.theta += (event.key === 'ArrowLeft' ? 1 : event.key === 'ArrowRight' ? -1 : 0) * Math.PI / 18;
    spherical.phi = Math.max(.1, Math.min(Math.PI - .1, spherical.phi + (event.key === 'ArrowUp' ? -1 : event.key === 'ArrowDown' ? 1 : 0) * Math.PI / 18));
    camera.position.setFromSpherical(spherical); controls.update(); render();
  };
  const resize = new ResizeObserver(() => { const { width, height } = host.getBoundingClientRect(); if (disposed || !width || !height) return; camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height, false); fit(); render(); });
  const changed=()=>{if(!fitting){fit();render();}};
  resize.observe(host); controls.addEventListener('change', changed); canvas.addEventListener('keydown', key);
  const lost = (event: Event) => { event.preventDefault(); onFailure(); }; canvas.addEventListener('webglcontextlost', lost);
  view('perspective'); update(initial, palette);
  return { update, view, dispose() { if (disposed) return; disposed = true; resize.disconnect(); canvas.removeEventListener('keydown', key); canvas.removeEventListener('webglcontextlost', lost); controls.removeEventListener('change', changed); controls.dispose(); release(content); renderer.dispose(); renderer.forceContextLoss(); canvas.remove(); } };
}
export type BoundaryComparisonRenderer = ReturnType<typeof createBoundaryComparisonPlot>;
