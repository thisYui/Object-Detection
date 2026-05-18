const imageInput = document.querySelector("#image-input");
const fileName = document.querySelector("#file-name");
const dropzone = document.querySelector(".dropzone");
const uploadPreview = document.querySelector("#upload-preview");
const uploadTitle = document.querySelector("#upload-title");
const thresholdInput = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#threshold-value");
const modeTabs = document.querySelectorAll(".mode-tab");
const sourcePanes = document.querySelectorAll(".source-pane");
const cameraStage = document.querySelector(".camera-stage");
const cameraVideo = document.querySelector("#camera-video");
const cameraCanvas = document.querySelector("#camera-canvas");
const cameraOverlay = document.querySelector("#camera-overlay");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const startCameraButton = document.querySelector("#start-camera");
const stopCameraButton = document.querySelector("#stop-camera");
const captureFrameButton = document.querySelector("#capture-frame");
const detectWebcamButton = document.querySelector("#detect-webcam");
const cameraStatus = document.querySelector("#camera-status");

let cameraStream = null;
let uploadPreviewUrl = null;
let webcamDetectTimer = null;
let webcamDetectInFlight = false;
let webcamAutoDetectEnabled = false;

function showUploadPreview(file) {
  if (!uploadPreview || !uploadTitle || !file) {
    return;
  }

  if (uploadPreviewUrl) {
    URL.revokeObjectURL(uploadPreviewUrl);
  }

  uploadPreviewUrl = URL.createObjectURL(file);
  uploadPreview.src = uploadPreviewUrl;
  uploadPreview.hidden = false;
  uploadTitle.textContent = "Selected image";
  dropzone.classList.add("has-preview");
}

function captureCameraBlob(options = {}) {
  return new Promise((resolve) => {
    if (!cameraStream || !cameraVideo || cameraVideo.videoWidth === 0) {
      resolve(null);
      return;
    }

    const mirror = Boolean(options.mirror);
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    const context = cameraCanvas.getContext("2d");
    context.save();
    if (mirror) {
      context.translate(cameraCanvas.width, 0);
      context.scale(-1, 1);
    }
    context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    context.restore();
    cameraCanvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
  });
}

function clearWebcamOverlay() {
  if (!cameraOverlay) {
    return;
  }
  const context = cameraOverlay.getContext("2d");
  context.clearRect(0, 0, cameraOverlay.width, cameraOverlay.height);
}

function drawWebcamDetections(detections) {
  if (!cameraOverlay || !cameraVideo) {
    return;
  }

  const stage = cameraOverlay.parentElement;
  const stageWidth = stage.clientWidth;
  const stageHeight = stage.clientHeight;
  const videoWidth = cameraVideo.videoWidth;
  const videoHeight = cameraVideo.videoHeight;
  const deviceRatio = window.devicePixelRatio || 1;

  cameraOverlay.width = Math.round(stageWidth * deviceRatio);
  cameraOverlay.height = Math.round(stageHeight * deviceRatio);
  cameraOverlay.style.width = `${stageWidth}px`;
  cameraOverlay.style.height = `${stageHeight}px`;

  const context = cameraOverlay.getContext("2d");
  context.setTransform(deviceRatio, 0, 0, deviceRatio, 0, 0);
  context.clearRect(0, 0, stageWidth, stageHeight);

  if (!videoWidth || !videoHeight) {
    return;
  }

  const scale = Math.min(stageWidth / videoWidth, stageHeight / videoHeight);
  const renderedWidth = videoWidth * scale;
  const renderedHeight = videoHeight * scale;
  const offsetX = (stageWidth - renderedWidth) / 2;
  const offsetY = (stageHeight - renderedHeight) / 2;
  const isMirrored = cameraStage && cameraStage.classList.contains("is-mirrored");

  context.lineWidth = 3;
  context.font = "13px system-ui, sans-serif";
  context.textBaseline = "top";

  detections.forEach((detection) => {
    const [x1, y1, x2, y2] = detection.bbox;
    const left = isMirrored ? offsetX + (videoWidth - x2) * scale : offsetX + x1 * scale;
    const top = offsetY + y1 * scale;
    const width = (x2 - x1) * scale;
    const height = (y2 - y1) * scale;
    const label = `${detection.class_name} ${Number(detection.confidence).toFixed(2)}`;

    context.strokeStyle = "#d7a456";
    context.fillStyle = "rgba(36, 39, 34, 0.82)";
    context.strokeRect(left, top, width, height);

    const textWidth = context.measureText(label).width + 12;
    const textHeight = 22;
    const labelTop = Math.max(0, top - textHeight);
    context.fillRect(left, labelTop, textWidth, textHeight);
    context.fillStyle = "#fffdf8";
    context.fillText(label, left + 6, labelTop + 4);
  });
}

function stopWebcamAutoDetect() {
  webcamAutoDetectEnabled = false;
  if (webcamDetectTimer) {
    window.clearInterval(webcamDetectTimer);
    webcamDetectTimer = null;
  }
  if (detectWebcamButton) {
    detectWebcamButton.textContent = "Start auto detect";
    detectWebcamButton.disabled = !cameraStream;
  }
}

function stopCameraStream() {
  stopWebcamAutoDetect();

  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  if (cameraVideo) {
    cameraVideo.pause();
    cameraVideo.srcObject = null;
    cameraVideo.classList.remove("is-active");
  }

  if (cameraPlaceholder) {
    cameraPlaceholder.hidden = false;
  }

  if (captureFrameButton) {
    captureFrameButton.disabled = true;
  }

  if (detectWebcamButton) {
    detectWebcamButton.disabled = true;
    detectWebcamButton.textContent = "Start auto detect";
  }

  if (stopCameraButton) {
    stopCameraButton.disabled = true;
  }

  if (startCameraButton) {
    startCameraButton.textContent = "Start webcam";
  }

  clearWebcamOverlay();

  if (cameraStatus) {
    cameraStatus.textContent = "Webcam stopped.";
  }
}

async function runWebcamDetection() {
  if (!webcamAutoDetectEnabled || webcamDetectInFlight) {
    return;
  }

  const blob = await captureCameraBlob({ mirror: false });
  if (!blob) {
    cameraStatus.textContent = "Camera is not ready yet.";
    return;
  }

  webcamDetectInFlight = true;
  cameraStatus.textContent = "Auto detecting current frame...";

  const formData = new FormData();
  formData.append("image", blob, `webcam-live-${Date.now()}.jpg`);
  formData.append("threshold", thresholdInput ? thresholdInput.value : "0.5");

  try {
    const response = await fetch("/api/webcam-detect", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "Webcam detection failed.");
    }

    drawWebcamDetections(payload.detections || []);
    cameraStatus.textContent = `${payload.detection_count} objects found. Next update in 2s.`;
  } catch (error) {
    clearWebcamOverlay();
    cameraStatus.textContent = error.message || "Cannot run webcam detection.";
  } finally {
    webcamDetectInFlight = false;
  }
}

function startWebcamAutoDetect() {
  if (!cameraStream) {
    return;
  }

  webcamAutoDetectEnabled = true;
  if (detectWebcamButton) {
    detectWebcamButton.textContent = "Stop auto detect";
    detectWebcamButton.disabled = false;
  }

  if (webcamDetectTimer) {
    window.clearInterval(webcamDetectTimer);
  }

  runWebcamDetection();
  webcamDetectTimer = window.setInterval(runWebcamDetection, 2000);
}

if (imageInput && fileName) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    fileName.textContent = file ? file.name : "JPG, PNG, BMP, WEBP";
    if (file) {
      showUploadPreview(file);
    }
  });
}

if (dropzone) {
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add("is-active"));
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove("is-active"));
  });
}

if (thresholdInput && thresholdValue) {
  thresholdInput.addEventListener("input", () => {
    thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
  });
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const mode = tab.dataset.mode;

    modeTabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    sourcePanes.forEach((pane) => pane.classList.toggle("is-active", pane.dataset.pane === mode));

    if (imageInput) {
      imageInput.required = mode === "upload";
    }
  });
});

if (startCameraButton && cameraVideo) {
  startCameraButton.addEventListener("click", async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("This browser does not support webcam access.");
      }

      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });

      cameraVideo.srcObject = cameraStream;
      await cameraVideo.play();
      cameraVideo.classList.add("is-active");
      cameraPlaceholder.hidden = true;
      captureFrameButton.disabled = false;
      detectWebcamButton.disabled = false;
      stopCameraButton.disabled = false;
      startCameraButton.textContent = "Restart webcam";
      cameraStatus.textContent = "Camera ready. Auto detection starts now.";
      clearWebcamOverlay();
      startWebcamAutoDetect();
    } catch (error) {
      cameraStatus.textContent = error.message || "Cannot start webcam.";
    }
  });
}

if (stopCameraButton) {
  stopCameraButton.addEventListener("click", stopCameraStream);
}

if (captureFrameButton && cameraVideo && cameraCanvas && imageInput) {
  captureFrameButton.addEventListener("click", async () => {
    const blob = await captureCameraBlob({ mirror: true });
    if (!blob) {
      cameraStatus.textContent = "Camera is not ready yet.";
      return;
    }

    const file = new File([blob], `webcam-${Date.now()}.jpg`, { type: "image/jpeg" });
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    imageInput.files = dataTransfer.files;
    imageInput.required = false;
    showUploadPreview(file);

    if (fileName) {
      fileName.textContent = file.name;
    }
    cameraStatus.textContent = "Frame captured. Press Detect objects to open the result page.";
  });
}

if (detectWebcamButton) {
  detectWebcamButton.addEventListener("click", () => {
    if (webcamAutoDetectEnabled) {
      stopWebcamAutoDetect();
      cameraStatus.textContent = "Auto detection paused.";
      return;
    }

    startWebcamAutoDetect();
  });
}

window.addEventListener("beforeunload", stopCameraStream);
