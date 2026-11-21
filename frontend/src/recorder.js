function formatTimestamp() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`
  );
}

function pickSupportedMimeType() {
  if (typeof MediaRecorder === "undefined") return null;
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) {
      return mime;
    }
  }
  return "";
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/**
 * WebGL canvas video recorder (MediaRecorder + captureStream) and instant PNG snapshotter.
 */
export class CanvasRecorder {
  constructor(canvas, { fps = 60, bitrate = 10_000_000, onStateChange, onError, onSaved } = {}) {
    this.canvas = canvas;
    this.fps = fps;
    this.bitrate = bitrate;
    this.onStateChange = onStateChange;
    this.onError = onError;
    this.onSaved = onSaved;

    this.recording = false;
    this._mediaRecorder = null;
    this._stream = null;
    this._chunks = [];
    this._startTime = 0;
    this._timerInterval = null;
  }

  start() {
    if (this.recording) return true;

    if (typeof this.canvas.captureStream !== "function" || typeof MediaRecorder === "undefined") {
      this.onError?.("Canvas recording is not supported in this browser.");
      return false;
    }

    const mimeType = pickSupportedMimeType();
    if (mimeType === null) {
      this.onError?.("MediaRecorder is not supported in this browser.");
      return false;
    }

    try {
      this._stream = this.canvas.captureStream(this.fps);
      const options = { videoBitsPerSecond: this.bitrate };
      if (mimeType) options.mimeType = mimeType;

      this._chunks = [];
      this._mediaRecorder = new MediaRecorder(this._stream, options);

      this._mediaRecorder.ondataavailable = (evt) => {
        if (evt.data && evt.data.size > 0) {
          this._chunks.push(evt.data);
        }
      };

      this._mediaRecorder.onstop = () => {
        const finalMime = this._mediaRecorder?.mimeType || mimeType || "video/webm";
        const ext = finalMime.includes("mp4") ? "mp4" : "webm";
        const filename = `sim-recording-${formatTimestamp()}.${ext}`;
        const blob = new Blob(this._chunks, { type: finalMime });
        if (blob.size > 0) {
          triggerDownload(blob, filename);
          this.onSaved?.(filename, blob.size);
        }
        this._cleanup();
      };

      this._mediaRecorder.onerror = (evt) => {
        console.error("MediaRecorder error:", evt);
        this.onError?.("Recording failed.");
        this._cleanup();
      };

      this._mediaRecorder.start(250);
      this.recording = true;
      this._startTime = performance.now();
      this._emitState(0);

      this._timerInterval = setInterval(() => {
        if (!this.recording) return;
        const elapsedSec = (performance.now() - this._startTime) / 1000;
        this._emitState(elapsedSec);
      }, 250);

      return true;
    } catch (err) {
      console.error("Failed to start canvas recording:", err);
      this.onError?.("Could not start recording.");
      this._cleanup();
      return false;
    }
  }

  stop() {
    if (!this.recording || !this._mediaRecorder) return;
    try {
      if (this._mediaRecorder.state !== "inactive") {
        this._mediaRecorder.stop();
      }
    } catch (err) {
      console.error("Failed to stop recording:", err);
      this._cleanup();
    }
  }

  toggle() {
    if (this.recording) {
      this.stop();
      return false;
    }
    return this.start();
  }

  captureScreenshot(renderNowFn) {
    try {
      renderNowFn?.();
      const filename = `sim-snapshot-${formatTimestamp()}.png`;
      this.canvas.toBlob((blob) => {
        if (!blob) {
          this.onError?.("Failed to capture screenshot.");
          return;
        }
        triggerDownload(blob, filename);
        this.onSaved?.(filename, blob.size, "screenshot");
      }, "image/png");
      return true;
    } catch (err) {
      console.error("Screenshot capture failed:", err);
      this.onError?.("Screenshot capture failed.");
      return false;
    }
  }

  _emitState(elapsedSec) {
    this.onStateChange?.({
      recording: this.recording,
      elapsedSec,
    });
  }

  _cleanup() {
    this.recording = false;
    if (this._timerInterval) {
      clearInterval(this._timerInterval);
      this._timerInterval = null;
    }
    if (this._stream) {
      for (const track of this._stream.getTracks()) {
        track.stop();
      }
      this._stream = null;
    }
    this._mediaRecorder = null;
    this._chunks = [];
    this._emitState(0);
  }
}
