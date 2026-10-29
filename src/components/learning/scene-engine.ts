import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { LabScene } from './lab-model';
import type { LabPalette } from './lab-palette';

/** Render only when data, the camera or the viewport changes. No idle animation loop. */
export function createLabRenderer(host: HTMLElement, initial: LabScene, initialPalette: LabPalette, onFailure: () => void) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.setClearColor(0, 0);
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, .1, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false; controls.enablePan = false; controls.enableZoom = false;
  controls.minPolarAngle = .15; controls.maxPolarAngle = Math.PI * .88;
  // Reserve vertical touch gestures for reading the page.
  controls.touches.ONE = THREE.TOUCH.ROTATE;
  renderer.domElement.style.touchAction = 'pan-y';
  let content = new THREE.Group(); scene.add(content);
  let disposed = false;
  const render = () => { if (!disposed) renderer.render(scene, camera); };
  function disposeGroup(group: THREE.Group) {
    group.traverse(object => {
      const item = object as THREE.Mesh;
      item.geometry?.dispose();
      if (Array.isArray(item.material)) item.material.forEach(material => material.dispose());
      else if (item.material) {
        (item.material as THREE.SpriteMaterial).map?.dispose(); item.material.dispose();
      }
    });
  }
  function line(points: number[][], color: string, dashed = false) {
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(...p)));
    const material = dashed ? new THREE.LineDashedMaterial({ color, dashSize: .12, gapSize: .08 }) : new THREE.LineBasicMaterial({ color });
    const object = new THREE.Line(geometry, material); object.computeLineDistances(); content.add(object);
  }
  function update(data: LabScene, palette: LabPalette) {
    scene.remove(content); disposeGroup(content); content = new THREE.Group(); scene.add(content);
    const grid = new THREE.GridHelper(10, 10, palette.neutral, palette.grid);
    if (data.kind === 'inner-products' || data.kind === 'eigenvalues') grid.rotation.x = Math.PI / 2;
    grid.material.transparent = true; grid.material.opacity = .35; content.add(grid);
    [[5, 0, 0], [0, 5, 0], [0, 0, 5]].forEach(end => line([[0, 0, 0], end], palette.neutral));
    (data.axisLabels || ['x', data.kind === 'gradients' ? 'L' : 'y', data.kind === 'gradients' ? 'y' : 'z']).forEach((label, i) => {
      const canvas = document.createElement('canvas'); canvas.width = 96; canvas.height = 64;
      const context = canvas.getContext('2d');
      if (context) {
        context.font = 'bold 36px monospace'; context.fillStyle = palette.label; context.textAlign = 'center'; context.fillText(label, 48, 45);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
        sprite.position.set(...([[5.4, 0, 0], [0, 5.4, 0], [0, 0, 5.4]][i] as [number, number, number]));
        sprite.scale.set(.55, .37, 1); content.add(sprite);
      }
    });
    data.lines.forEach(l => line(l.points, palette[l.color], l.dashed));
    data.arrows.forEach(arrow => {
      const start = new THREE.Vector3(...arrow.start); const delta = new THREE.Vector3(...arrow.end).sub(start); const length = delta.length();
      if (length < 1e-8) return;
      if (arrow.dashed) line([arrow.start, arrow.end], palette[arrow.color], true);
      else content.add(new THREE.ArrowHelper(delta.normalize(), start, length, palette[arrow.color], Math.min(.3, length * .24), Math.min(.16, length * .12)));
    });
    data.dots.forEach(dot => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(dot.radius || .1, 16, 12), new THREE.MeshBasicMaterial({ color: palette[dot.color] }));
      mesh.position.set(...dot.position); content.add(mesh);
    });
    if (data.surface) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.surface.vertices, 3)); geometry.setIndex(data.surface.indices);
      content.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: palette.green, side: THREE.DoubleSide, transparent: true, opacity: .25, depthWrite: false })));
      content.add(new THREE.LineSegments(new THREE.WireframeGeometry(geometry), new THREE.LineBasicMaterial({ color: palette.green, transparent: true, opacity: .22 })));
    }
    render();
  }
  const reset = () => {
    if (initial.kind === 'inner-products' || initial.kind === 'eigenvalues') camera.position.set(0, 1, 15);
    else if (initial.kind === 'least-squares') camera.position.set(10, 6, -9);
    else camera.position.set(8, 5.5, 9);
    controls.target.set(0, 1, 0); controls.update(); render();
  };
  reset();
  controls.addEventListener('change', render);
  const observer = new ResizeObserver(() => {
    const { width, height } = host.getBoundingClientRect();
    if (!width || !height || disposed) return;
    camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height, false); render();
  });
  observer.observe(host);
  const lose = (event: Event) => { event.preventDefault(); onFailure(); };
  renderer.domElement.addEventListener('webglcontextlost', lose);
  update(initial, initialPalette);
  return { update, reset,
    view: (view: 'front' | 'top' | 'perspective') => {
      if (view === 'perspective') return reset();
      camera.position.set(...(view === 'front' ? [0, 1, 14] : [0, 14, .01]) as [number, number, number]); controls.target.set(0, 0, 0); controls.update(); render();
    },
    dispose: () => {
      disposed = true; observer.disconnect(); controls.removeEventListener('change', render); controls.dispose();
      renderer.domElement.removeEventListener('webglcontextlost', lose); disposeGroup(content); renderer.dispose(); renderer.domElement.remove();
    },
  };
}
export type LabRenderer = ReturnType<typeof createLabRenderer>;
