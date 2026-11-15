// three.js view of a MuJoCo model. Geometry, materials and poses come straight from MjModel / MjData:
// every frame copies geom_xpos / geom_xmat into the scene, the way MuJoCo's own renderer does.
import * as THREE from 'three';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';

const G = { PLANE: 0, SPHERE: 2, CAPSULE: 3, ELLIPSOID: 4, CYLINDER: 5, BOX: 6, MESH: 7 };
const meshCache = new Map();

function meshGeometry(m, id) {
  if (meshCache.has(id)) return meshCache.get(id);
  const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
  const pos = m.mesh_vert.slice(3 * va, 3 * (va + vn));
  const idx = m.mesh_face.slice(3 * fa, 3 * (fa + fn));
  let g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
  g = toCreasedNormals(g, Math.PI / 5);        // CAD parts: smooth curves, crisp edges
  meshCache.set(id, g);
  return g;
}

const unit = {
  sphere: new THREE.SphereGeometry(1, 40, 28),
  box: new THREE.BoxGeometry(2, 2, 2),
  cylinder: new THREE.CylinderGeometry(1, 1, 2, 32).rotateX(Math.PI / 2),
};

export function makeTextures(urls) {
  const loader = new THREE.TextureLoader();
  const tex = {};
  for (const [name, url] of Object.entries(urls)) {
    const t = loader.load(url);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    tex[name] = t;
  }
  return tex;
}

function materialFor(m, g, names, tex) {
  const matId = m.geom_matid[g];
  const name = matId >= 0 ? names.mat[matId] : '';
  const rgba = matId >= 0 ? m.mat_rgba.slice(4 * matId, 4 * matId + 4) : m.geom_rgba.slice(4 * g, 4 * g + 4);
  const color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
  if (name === 'table') {
    const t = tex.wood.clone(); t.repeat.set(2.2, 2.2); t.needsUpdate = true;
    return new THREE.MeshStandardMaterial({ map: t, roughness: 0.62, metalness: 0 });
  }
  if (name === 'carton') {
    const t = tex.pulp.clone(); t.repeat.set(1.5, 1.5); t.needsUpdate = true;
    return new THREE.MeshStandardMaterial({ map: t, color: 0xd9cdb8, roughness: 0.95 });
  }
  if (name === 'shell') return new THREE.MeshPhysicalMaterial({ color: 0xf6e7cf, roughness: 0.5, clearcoat: 0.15, sheen: 0.3 });
  if (name === 'yolk') return new THREE.MeshPhysicalMaterial({ color: 0xffa21a, roughness: 0.15, clearcoat: 1 });
  if (name === 'white') return new THREE.MeshPhysicalMaterial({ color: 0xfffff4, roughness: 0.05, transmission: 0.6, transparent: true, opacity: 0.75, thickness: 0.002 });
  if (name === 'basket') return new THREE.MeshStandardMaterial({ color, roughness: 0.8 });
  // SO-101: printed PLA and STS3215 servos
  const dark = rgba[0] < 0.3;
  return new THREE.MeshStandardMaterial({ color, roughness: dark ? 0.45 : 0.55, metalness: dark ? 0.1 : 0 });
}

function readNames(m) {
  const buf = new TextDecoder().decode(m.names);
  const at = (adr) => buf.slice(adr, buf.indexOf('\0', adr));
  return {
    mat: Array.from({ length: m.nmat }, (_, i) => at(m.name_matadr[i])),
    body: Array.from({ length: m.nbody }, (_, i) => at(m.name_bodyadr[i])),
  };
}

export class SceneView {
  constructor(m, tex) {
    this.m = m;
    this.scene = new THREE.Scene();
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;           // MuJoCo is z-up, three.js is y-up
    this.scene.add(this.root);
    this.names = readNames(m);
    this.geoms = [];
    this.collision = [];
    this.pickable = [];
    const collisionMat = new THREE.MeshBasicMaterial({ color: 0xff4d4d, transparent: true, opacity: 0.35, depthWrite: false });
    for (let g = 0; g < m.ngeom; g++) {
      const type = m.geom_type[g], group = m.geom_group[g];
      const colliding = m.geom_contype[g] || m.geom_conaffinity[g];
      const visible = group <= 2;
      if (!visible && !(colliding && group >= 3)) continue;
      let geo, scale = null;
      const s = m.geom_size.slice(3 * g, 3 * g + 3);
      if (type === G.PLANE) {
        geo = new THREE.PlaneGeometry(3, 3);
        scale = [1, 1, 1];
      } else if (type === G.SPHERE) { geo = unit.sphere; scale = [s[0], s[0], s[0]]; }
      else if (type === G.ELLIPSOID) { geo = unit.sphere; scale = 'size'; }
      else if (type === G.BOX) { geo = unit.box; scale = [s[0], s[1], s[2]]; }
      else if (type === G.CYLINDER) { geo = unit.cylinder; scale = [s[0], s[0], s[1]]; }
      else if (type === G.MESH) { geo = meshGeometry(m, m.geom_dataid[g]); scale = [1, 1, 1]; }
      else continue;
      const mat = visible ? materialFor(m, g, this.names, tex) : collisionMat;
      const obj = new THREE.Mesh(geo, mat);
      obj.matrixAutoUpdate = false;
      obj.castShadow = visible && type !== G.PLANE;
      obj.receiveShadow = visible;
      const body = m.geom_bodyid[g];
      obj.userData = { g, body, scale, bodyName: this.names.body[body] };
      this.root.add(obj);
      if (visible) this.geoms.push(obj); else { obj.visible = false; this.collision.push(obj); }
      if (/^(egg|yolk|white|shell)/.test(this.names.body[body])) this.pickable.push(obj);
    }
    // pad touch sensors: the two site boxes, tinted by their reading
    this.pads = [];
    const padMat = () => new THREE.MeshBasicMaterial({ color: 0x18c6d8, transparent: true, opacity: 0.55, depthWrite: false });
    for (let i = 0; i < m.nsite; i++) {
      if (m.site_group[i] !== 4) continue;
      const o = new THREE.Mesh(unit.box, padMat());
      o.matrixAutoUpdate = false; o.visible = false;
      o.userData = { site: i, scale: Array.from(m.site_size.slice(3 * i, 3 * i + 3)) };
      this.root.add(o); this.pads.push(o);
    }
    // contact force arrows
    this.arrows = [];
    this.arrowGroup = new THREE.Group();
    this.root.add(this.arrowGroup);
    // drag spring
    const lg = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    this.spring = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x18c6d8 }));
    this.spring.visible = false; this.spring.frustumCulled = false;
    this.root.add(this.spring);
    this._lights();
  }

  _lights() {
    const hemi = new THREE.HemisphereLight(0xdfe8f5, 0x3a3228, 0.9);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xfff3e2, 2.4);
    key.position.set(0.45, 1.2, 0.35);            // y-up world: high, in front, slightly right
    key.target.position.set(0.18, 0, 0);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const c = key.shadow.camera;
    c.left = -0.45; c.right = 0.45; c.top = 0.45; c.bottom = -0.45; c.near = 0.3; c.far = 2.2;
    key.shadow.bias = -0.0004; key.shadow.normalBias = 0.002; key.shadow.radius = 3;
    this.scene.add(key, key.target);
    const rim = new THREE.DirectionalLight(0xbcd2ff, 0.7);
    rim.position.set(-0.6, 0.5, 0.6);
    this.scene.add(rim);
  }

  _setMatrix(obj, p, R, sx, sy, sz) {
    obj.matrix.set(R[0] * sx, R[1] * sy, R[2] * sz, p[0], R[3] * sx, R[4] * sy, R[5] * sz, p[1],
      R[6] * sx, R[7] * sy, R[8] * sz, p[2], 0, 0, 0, 1);
    obj.matrixWorldNeedsUpdate = true;
  }

  // pose everything from MjData (geom_xpos, geom_xmat, site_xpos, site_xmat)
  update(d, opts = {}) {
    const m = this.m;
    for (const list of [this.geoms, this.collision]) {
      for (const obj of list) {
        if (!obj.visible) continue;
        const { g, scale } = obj.userData;
        const s = scale === 'size' ? m.geom_size.subarray(3 * g, 3 * g + 3) : scale;
        this._setMatrix(obj, d.geom_xpos.subarray(3 * g, 3 * g + 3), d.geom_xmat.subarray(9 * g, 9 * g + 9), s[0], s[1], s[2]);
      }
    }
    for (const [i, o] of this.pads.entries()) {
      o.visible = !!opts.sensors;
      if (!o.visible) continue;
      const { site, scale } = o.userData;
      this._setMatrix(o, d.site_xpos.subarray(3 * site, 3 * site + 3), d.site_xmat.subarray(9 * site, 9 * site + 9), ...scale);
      const f = opts.padForces ? opts.padForces[i] : 0;
      o.material.color.setHSL(0.52 - Math.min(f / 4, 1) * 0.52, 0.85, 0.5);
      o.material.opacity = 0.35 + Math.min(f / 4, 1) * 0.5;
    }
  }

  // collision geometry in red, robot shells see-through
  showCollision(on) {
    for (const o of this.collision) o.visible = on;
    for (const o of this.geoms) {
      if (this.m.geom_group[o.userData.g] !== 2) continue;
      o.material.transparent = on; o.material.opacity = on ? 0.3 : 1; o.material.depthWrite = !on;
    }
  }

  // contacts: [{p:[x,y,z], f:[x,y,z], jaw:bool}] in MuJoCo world coordinates, f = force on the object, N
  setContacts(list, scale = 0.012) {
    while (this.arrows.length < list.length) {
      const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.01, 0xff8a1f, 0.006, 0.004);
      a.line.material.linewidth = 2;
      this.arrowGroup.add(a); this.arrows.push(a);
    }
    this.arrows.forEach((a, i) => {
      const c = list[i];
      if (!c) { a.visible = false; return; }
      const f = new THREE.Vector3(...c.f), n = f.length();
      if (n < 0.05) { a.visible = false; return; }
      a.visible = true;
      // draw the arrow ending at the contact point, pointing along the force on the object
      const len = Math.min(n * scale, 0.07);
      const dir = f.clone().divideScalar(n);
      a.position.set(c.p[0] - dir.x * len, c.p[1] - dir.y * len, c.p[2] - dir.z * len);
      a.setDirection(dir);
      a.setLength(len, Math.min(0.008, len * 0.4), Math.min(0.005, len * 0.3));
      a.setColor(c.jaw ? 0xff8a1f : 0x7fa7ff);
    });
  }

  setSpring(a, b) {
    if (!a) { this.spring.visible = false; return; }
    const p = this.spring.geometry.attributes.position;
    p.array.set([...a, ...b]); p.needsUpdate = true;
    this.spring.visible = true;
  }
}

// contact list for drawing: every active contact that touches an egg or its debris
export function eggContacts(env, bodies) {
  const mj = env.mj, m = env.m, d = env.d, out = [];
  const con = d.contact;
  for (let i = 0; i < d.ncon; i++) {
    const c = con.get(i);
    const b1 = m.geom_bodyid[c.geom1], b2 = m.geom_bodyid[c.geom2];
    const on2 = bodies.has(b2), on1 = bodies.has(b1);
    if (on1 || on2) {
      mj.mj_contactForce(m, d, i, env.f6);
      const f = env.f6.GetView(), fr = c.frame, sign = on2 ? 1 : -1;   // contact force acts on geom2
      const w = [0, 1, 2].map((j) => sign * (f[0] * fr[j] + f[1] * fr[3 + j] + f[2] * fr[6 + j]));
      const jaw = env.jawGeoms.has(on2 ? c.geom1 : c.geom2);
      out.push({ p: Array.from(c.pos), f: w, jaw });
    }
    c.delete();
  }
  con.delete();
  return out;
}
