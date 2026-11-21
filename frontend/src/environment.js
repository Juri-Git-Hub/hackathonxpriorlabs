import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

const ENV_THEMES = {
  refinery_dusk: {
    zenith: 0x0c1729,
    mid: 0x243a59,
    horizon: 0x5d6e82,
    warmGlow: 0xe69856,
    fogColor: 0x2a3648,
    fogDensity: 0.0024,
    sunColor: 0xffdcb0,
    sunIntensity: 2.8,
    sunPos: [85, 52, 76],
    hemiSky: 0x94bce6,
    hemiGround: 0x222831,
    hemiIntensity: 0.9,
    fillColor: 0x5b8bc9,
    fillIntensity: 0.85,
    exposure: 1.15,
  },
  desert_sun: {
    zenith: 0x1d4b7a,
    mid: 0x5e8db8,
    horizon: 0xd9b382,
    warmGlow: 0xf4a853,
    fogColor: 0x8f7a60,
    fogDensity: 0.0020,
    sunColor: 0xfff0d0,
    sunIntensity: 3.2,
    sunPos: [95, 78, 52],
    hemiSky: 0xd9ecff,
    hemiGround: 0x6b533b,
    hemiIntensity: 1.05,
    fillColor: 0xc2986b,
    fillIntensity: 0.65,
    exposure: 1.12,
  },
  arctic_frost: {
    zenith: 0x1a304d,
    mid: 0x4b7096,
    horizon: 0xa8c2d6,
    warmGlow: 0xdbeeff,
    fogColor: 0x6e8599,
    fogDensity: 0.0023,
    sunColor: 0xe6f2ff,
    sunIntensity: 2.7,
    sunPos: [-78, 62, 74],
    hemiSky: 0xcce4ff,
    hemiGround: 0x556470,
    hemiIntensity: 1.1,
    fillColor: 0x89b4db,
    fillIntensity: 0.85,
    exposure: 1.08,
  },
  alpine_steel: {
    zenith: 0x0f1d33,
    mid: 0x2b4566,
    horizon: 0x677f99,
    warmGlow: 0x9ec5e8,
    fogColor: 0x2d3c4e,
    fogDensity: 0.0022,
    sunColor: 0xeaf3ff,
    sunIntensity: 2.75,
    sunPos: [68, 64, -72],
    hemiSky: 0xa2c4e8,
    hemiGround: 0x222933,
    hemiIntensity: 0.95,
    fillColor: 0x5d82ad,
    fillIntensity: 0.9,
    exposure: 1.14,
  },
  overcast_steam: {
    zenith: 0x243140,
    mid: 0x455669,
    horizon: 0x7a8a99,
    warmGlow: 0xb8c2cc,
    fogColor: 0x4c5a69,
    fogDensity: 0.0031,
    sunColor: 0xe3eaf2,
    sunIntensity: 2.25,
    sunPos: [-55, 86, 65],
    hemiSky: 0xb5c7d9,
    hemiGround: 0x242d36,
    hemiIntensity: 1.15,
    fillColor: 0x738ca6,
    fillIntensity: 0.85,
    exposure: 1.10,
  },
};

/**
 * Sets up atmospheric lighting, soft PCF shadows, PMREM reflections, fog,
 * and a switchable procedural sky dome so each of the 5 worlds has a distinct
 * time-of-day, climate, and lighting mood.
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

  const horizonColor = new THREE.Color(0x2a3648);
  scene.background = horizonColor;
  const fog = new THREE.FogExp2(0x2a3648, 0.0024);
  scene.fog = fog;

  // Switchable procedural sky dome
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

        // Horizon atmospheric scatter & sun glow
        float sunAmt = max(dot(dir, uSunDir), 0.0);
        float horizonBand = exp(-h * 9.0);
        col += uWarmGlow * horizonBand * (0.22 + 0.45 * pow(sunAmt, 3.0));
        col += vec3(1.0, 0.94, 0.82) * pow(sunAmt, 72.0) * 0.58;

        // Smooth FBM atmospheric clouds
        vec2 cloudUV = dir.xz / max(h + 0.18, 0.18) * 1.4;
        float cl = fbm(cloudUV + vec2(1.7, 3.1));
        float cloudMask = smoothstep(0.48, 0.82, cl) * smoothstep(0.04, 0.22, h) * (1.0 - smoothstep(0.32, 0.72, h));
        vec3 cloudCol = mix(uMid * 1.15, uWarmGlow * 1.1, pow(sunAmt, 2.0));
        col = mix(col, cloudCol, cloudMask * 0.38);

        // Fade below the horizon for a neutral backdrop behind the site boundary.
        col = mix(uZenith * 0.55, col, smoothstep(-0.18, 0.0, dir.y));

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));

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

  const hemiLight = new THREE.HemisphereLight(0x94bce6, 0x222831, 0.9);
  scene.add(hemiLight);

  const fillLight = new THREE.DirectionalLight(0x5b8bc9, 0.85);
  fillLight.position.set(-75, 38, -68);
  scene.add(fillLight);

  function setTheme(themeName = "refinery_dusk") {
    const cfg = ENV_THEMES[themeName] ?? ENV_THEMES.refinery_dusk;
    skyMat.uniforms.uZenith.value.setHex(cfg.zenith);
    skyMat.uniforms.uMid.value.setHex(cfg.mid);
    skyMat.uniforms.uHorizon.value.setHex(cfg.horizon);
    skyMat.uniforms.uWarmGlow.value.setHex(cfg.warmGlow);
    skyMat.uniforms.uSunDir.value.set(...cfg.sunPos).normalize();

    scene.background.setHex(cfg.fogColor);
    fog.color.setHex(cfg.fogColor);
    fog.density = cfg.fogDensity;

    sunLight.color.setHex(cfg.sunColor);
    sunLight.intensity = cfg.sunIntensity;
    sunLight.position.set(...cfg.sunPos);

    hemiLight.color.setHex(cfg.hemiSky);
    hemiLight.groundColor.setHex(cfg.hemiGround);
    hemiLight.intensity = cfg.hemiIntensity;

    fillLight.color.setHex(cfg.fillColor);
    fillLight.intensity = cfg.fillIntensity;

    renderer.toneMappingExposure = cfg.exposure;
  }

  setTheme("refinery_dusk");
  return { setTheme };
}
