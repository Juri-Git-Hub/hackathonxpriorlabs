import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

/**
 * Sets up cinematic golden-hour industrial twilight lighting, soft PCF shadows,
 * PMREM metallic reflections, atmospheric haze, and a rich sky dome with subtle
 * warm horizon glow.
 */
export function setupEnvironment(scene, renderer) {
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // Procedural industrial environment reflections for PBR metals
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environment = envTex;
  scene.environmentIntensity = 0.78;
  pmrem.dispose();

  // Warm-cool atmospheric dusk fog matching the sky horizon
  const horizonColor = new THREE.Color(0x2a3648);
  scene.background = horizonColor;
  scene.fog = new THREE.FogExp2(0x2a3648, 0.0026);

  // Rich procedural golden-hour twilight sky dome with subtle horizon cloud & ridge glow
  const skyGeo = new THREE.SphereGeometry(320, 40, 24);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uZenith: { value: new THREE.Color(0x0c1729) },
      uMid: { value: new THREE.Color(0x243a59) },
      uHorizon: { value: new THREE.Color(0x5d6e82) },
      uWarmGlow: { value: new THREE.Color(0xe69856) },
      uSunDir: { value: new THREE.Vector3(0.68, 0.24, 0.69).normalize() },
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
      uniform vec3 uWarmGlow;
      uniform vec3 uSunDir;
      varying vec3 vWorldPos;

      float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
      }
      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = hash(i);
        float b = hash(i + vec2(1.0, 0.0));
        float c = hash(i + vec2(0.0, 1.0));
        float d = hash(i + vec2(1.0, 1.0));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
      }
      float fbm(vec2 p) {
        float v = 0.0;
        float amp = 0.5;
        for (int i = 0; i < 4; i++) {
          v += amp * noise(p);
          p *= 2.05;
          amp *= 0.5;
        }
        return v;
      }

      void main() {
        vec3 dir = normalize(vWorldPos);
        float h = clamp(dir.y, 0.0, 1.0);
        vec3 col = mix(uHorizon, uMid, smoothstep(0.0, 0.24, h));
        col = mix(col, uZenith, smoothstep(0.20, 0.78, h));

        // Silky golden-hour horizon band & sun scatter
        float sunAmt = max(dot(dir, uSunDir), 0.0);
        float horizonBand = exp(-h * 10.0);
        col += uWarmGlow * horizonBand * (0.24 + 0.52 * pow(sunAmt, 3.0));
        col += vec3(1.0, 0.90, 0.72) * pow(sunAmt, 72.0) * 0.65;

        // Smooth FBM twilight clouds (no blocky artifacts)
        vec2 cloudUV = dir.xz / max(h + 0.18, 0.18) * 1.4;
        float cl = fbm(cloudUV + vec2(1.7, 3.1));
        float cloudMask = smoothstep(0.48, 0.82, cl) * smoothstep(0.04, 0.22, h) * (1.0 - smoothstep(0.32, 0.72, h));
        vec3 cloudCol = mix(vec3(0.32, 0.40, 0.52), vec3(0.88, 0.62, 0.42), pow(sunAmt, 2.0));
        col = mix(col, cloudCol, cloudMask * 0.42);

        // Fade below the horizon for a neutral backdrop behind the site boundary.
        col = mix(vec3(0.08, 0.11, 0.15), col, smoothstep(-0.18, 0.0, dir.y));

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));

  // Warm low-angle golden-hour sunlight casting dramatic long shadows
  const sunLight = new THREE.DirectionalLight(0xffdcb0, 2.8);
  sunLight.position.set(85, 52, 76);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(2048, 2048);
  sunLight.shadow.camera.near = 10;
  sunLight.shadow.camera.far = 280;
  const d = 82;
  sunLight.shadow.camera.left = -d;
  sunLight.shadow.camera.right = d;
  sunLight.shadow.camera.top = d;
  sunLight.shadow.camera.bottom = -d;
  sunLight.shadow.bias = -0.00035;
  sunLight.shadow.normalBias = 0.04;
  scene.add(sunLight);

  // Sky / concrete bounce hemisphere light
  const hemiLight = new THREE.HemisphereLight(0x94bce6, 0x222831, 0.9);
  scene.add(hemiLight);

  // Cool blue twilight rim fill light from opposite side for depth & silhouette separation
  const fillLight = new THREE.DirectionalLight(0x5b8bc9, 0.85);
  fillLight.position.set(-75, 38, -68);
  scene.add(fillLight);
}
