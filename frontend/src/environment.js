import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

/**
 * Sets up realistic golden-hour industrial lighting, soft PCF shadows,
 * PMREM metallic reflections, atmospheric fog, and a smooth procedural sky dome.
 */
export function setupEnvironment(scene, renderer) {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // Procedural studio/industrial environment reflections for PBR metals
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environment = envTex;
  scene.environmentIntensity = 0.65;
  pmrem.dispose();

  // Atmospheric horizon fog
  const horizonColor = new THREE.Color(0x283546);
  scene.background = horizonColor;
  scene.fog = new THREE.FogExp2(0x283546, 0.0032);

  // Smooth gradient sky dome (twilight zenith -> cool steel sky -> warm golden horizon)
  const skyGeo = new THREE.SphereGeometry(280, 32, 20);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uZenith: { value: new THREE.Color(0x0f1b2e) },
      uMid: { value: new THREE.Color(0x293d56) },
      uHorizon: { value: new THREE.Color(0x526478) },
      uSunGlow: { value: new THREE.Color(0xd99457) },
      uSunDir: { value: new THREE.Vector3(0.62, 0.34, 0.71).normalize() },
    },
    vertexShader: `
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: `
      uniform vec3 uZenith;
      uniform vec3 uMid;
      uniform vec3 uHorizon;
      uniform vec3 uSunGlow;
      uniform vec3 uSunDir;
      varying vec3 vWorldPos;
      void main() {
        vec3 dir = normalize(vWorldPos);
        float h = clamp(dir.y, 0.0, 1.0);
        vec3 col = mix(uHorizon, uMid, smoothstep(0.0, 0.25, h));
        col = mix(col, uZenith, smoothstep(0.22, 0.75, h));
        float sunAmt = max(dot(dir, uSunDir), 0.0);
        col += uSunGlow * pow(sunAmt, 10.0) * 0.45;
        col += vec3(1.0, 0.92, 0.78) * pow(sunAmt, 90.0) * 0.65;
        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  const skyDome = new THREE.Mesh(skyGeo, skyMat);
  scene.add(skyDome);

  // Warm late-afternoon directional sun casting crisp soft shadows
  const sunLight = new THREE.DirectionalLight(0xffe0b8, 2.4);
  sunLight.position.set(78, 58, 72);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(2048, 2048);
  sunLight.shadow.camera.near = 10;
  sunLight.shadow.camera.far = 260;
  const d = 82;
  sunLight.shadow.camera.left = -d;
  sunLight.shadow.camera.right = d;
  sunLight.shadow.camera.top = d;
  sunLight.shadow.camera.bottom = -d;
  sunLight.shadow.bias = -0.0004;
  sunLight.shadow.normalBias = 0.04;
  scene.add(sunLight);

  // Cool sky/ground hemisphere bounce light
  const hemiLight = new THREE.HemisphereLight(0x8bb4dc, 0x262b33, 0.85);
  scene.add(hemiLight);

  // Subtle rim fill light from opposite side so shadow faces stay readable
  const fillLight = new THREE.DirectionalLight(0x6b93c4, 0.65);
  fillLight.position.set(-70, 35, -60);
  scene.add(fillLight);
}
