// --- main.js ---
document.addEventListener("DOMContentLoaded", () => {
  console.log("✅ DOM Loaded");

  const genderButtons = document.querySelectorAll(".gender-option");
  const eventButtons = document.querySelectorAll(".event-option");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const eventScreen = document.getElementById("eventOptionsScreen");
  const genderScreen = document.getElementById("genderScreen");
  const hautfit = document.getElementById("hautfit");

  const videoElement = document.getElementById("cameraVideo");
  const poseCanvas = document.getElementById("poseCanvas");
  const ctx = poseCanvas.getContext("2d");
  const cameraSelect = document.getElementById("cameraSelect");

  // --- Logo + Get Started ---
  const logo = document.getElementById("logo");
  const getStartedBtn = document.getElementById("get-started-btn");

  if (logo) {
    logo.addEventListener("click", () => {
      logo.classList.add("fade-out");
      setTimeout(() => {
        logo.style.display = "none";
        document.getElementById("introText").style.display = "block";
      }, 1000);
    });
  }

  if (getStartedBtn) {
    getStartedBtn.addEventListener("click", () => {
      welcomeScreen.style.display = "none";
      eventScreen.style.display = "block";
    });
  }

  let selectedEvent = null;
  let selectedGender = null;

  // --- Event Selection ---
  eventButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      selectedEvent = btn.dataset.event;
      eventScreen.style.display = "none";
      genderScreen.style.display = "block";
    });
  });

  // --- Gender Selection ---
  genderButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      selectedGender = btn.dataset.gender;
      genderScreen.style.display = "none";
      hautfit.style.display = "block";
      setTimeout(() => hautfit.style.opacity = "1", 200);
    });
  });

  // --- Yes Button (start scan) ---
  const cameraYesBtn = document.getElementById("cameraYesBtn");
  const cameraIntroContainer = document.getElementById("cameraIntroContainer");
  const cameraMessage = document.getElementById("cameraMessage");
  const dots = document.getElementById("dots");

  if (cameraYesBtn) {
    cameraYesBtn.addEventListener("click", () => {
      cameraIntroContainer.style.display = "none";
      cameraMessage.style.display = "block";

      let dotCount = 1;
      const dotInterval = setInterval(() => {
        dotCount = (dotCount % 3) + 1;
        dots.textContent = ".".repeat(dotCount);
      }, 500);

      setTimeout(() => {
        clearInterval(dotInterval);
        cameraMessage.textContent = "🔍 Scanning started...";
        initPoseDetection();
      }, 3000);
    });
  }

  // --- MediaPipe Pose Detection ---
  function initPoseDetection() {
    const pose = new Pose({
      locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    const parts = [
      { name: "Head", indices: [0] },
      { name: "Left Shoulder", indices: [11] },
      { name: "Right Shoulder", indices: [12] },
      { name: "Left Arm", indices: [13, 15] },
      { name: "Right Arm", indices: [14, 16] },
      { name: "Waist", indices: [23, 24] },
      { name: "Left Leg", indices: [25, 27] },
      { name: "Right Leg", indices: [26, 28] },
      { name: "Left Foot", indices: [31] },
      { name: "Right Foot", indices: [32] }
    ];

    let scannedParts = [];
    let currentIndex = 0;
    let glow = 0;
    let lastResults = null;

    pose.onResults(results => {
      ctx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
      if (!results.poseLandmarks) return;

      lastResults = results;

      scannedParts.forEach(part => {
        drawBox(part, results, false);
      });

      if (currentIndex < parts.length) {
        drawBox(parts[currentIndex], results, true);
        glow += 0.2;
      }
    });

    function drawBox(part, results, isGlowing) {
      const landmarks = part.indices
        .map(i => results.poseLandmarks[i])
        .filter(l => l);

      if (landmarks.length === 0) return;

      const xs = landmarks.map(l => l.x * poseCanvas.width);
      const ys = landmarks.map(l => l.y * poseCanvas.height);

      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);

      ctx.lineWidth = 3;
      ctx.strokeStyle = isGlowing
        ? `rgba(0, 255, 0, ${0.5 + 0.5 * Math.sin(glow)})`
        : "rgba(0,255,0,0.8)";
      ctx.strokeRect(minX - 15, minY - 15, (maxX - minX) + 30, (maxY - minY) + 30);

      ctx.fillStyle = "lime";
      ctx.font = "16px Arial";
      ctx.fillText(part.name, minX, minY - 20);
    }

    // --- Temporary Skin Tone Detection ---
    function detectSkinTone(results) {
      if (!results.poseLandmarks) return "Unknown";
      const nose = results.poseLandmarks[0];
      if (!nose) return "Unknown";

      const x = Math.floor(nose.x * poseCanvas.width);
      const y = Math.floor(nose.y * poseCanvas.height);

      const regionSize = 20;
      const imageData = ctx.getImageData(
        Math.max(0, x - regionSize),
        Math.max(0, y - regionSize),
        regionSize * 2,
        regionSize * 2
      );

      let r = 0, g = 0, b = 0, count = 0;
      for (let i = 0; i < imageData.data.length; i += 4) {
        r += imageData.data[i];
        g += imageData.data[i + 1];
        b += imageData.data[i + 2];
        count++;
      }

      r = Math.floor(r / count);
      g = Math.floor(g / count);
      b = Math.floor(b / count);

      const brightness = (r + g + b) / 3;
      if (brightness > 180) return "Light";
      if (brightness > 130) return "Mid-light";
      if (brightness > 80) return "Mid-dark";
      return "Dark";
    }

    // --- Body Type Detection ---
    function detectBodyType(results) {
      if (!results.poseLandmarks) return { bodyType: "Unknown", heightCm: 0 };

      const lm = results.poseLandmarks;

      const shoulderWidth = Math.abs(lm[11].x - lm[12].x);
      const hipWidth = Math.abs(lm[23].x - lm[24].x);
      const waistWidth = (shoulderWidth + hipWidth) / 2 * 0.8; 
      const height = Math.abs(lm[0].y - lm[32].y);

      let bodyType = "Rectangle";
      if (shoulderWidth > hipWidth * 1.2) bodyType = "Inverted Triangle";
      else if (hipWidth > shoulderWidth * 1.2) bodyType = "Triangle";
      else if (Math.abs(shoulderWidth - hipWidth) < 0.1 && waistWidth < shoulderWidth * 0.8)
        bodyType = "Hourglass";

      const heightCm = (height * poseCanvas.height / 480) * 170;

      return { bodyType, heightCm: heightCm.toFixed(2) };
    }

    function startScanning() {
      if (currentIndex >= parts.length) {
        cameraMessage.textContent = "✅ Scanning complete!";

        const skinTone = detectSkinTone(lastResults);
        const { bodyType, heightCm } = detectBodyType(lastResults);

        console.log("🎨 Skin Tone:", skinTone);
        console.log("📏 Body Type:", bodyType, "| Height:", heightCm, "cm");

        // --- FIX: match Flask route ---
        fetch("/start-camera", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            status: "active",
            event: selectedEvent,
            gender: selectedGender,
            skin_tone: skinTone,
            body_type: bodyType,
            height_cm: heightCm,
            timestamp: new Date().toISOString()
          })
        })
          .then(res => res.json())
          .then(data => console.log("✅ Saved session to backend:", data))
          .catch(err => console.error("❌ Error saving session:", err));

        return;
      }

      cameraMessage.textContent = `🔍 Scanning ${parts[currentIndex].name}...`;
      setTimeout(() => {
        scannedParts.push(parts[currentIndex]);
        currentIndex++;
        startScanning();
      }, 1500);
    }

    async function startCameraStream(deviceId = null) {
      const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" }
      };

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        videoElement.play();

        const camera = new Camera(videoElement, {
          onFrame: async () => {
            await pose.send({ image: videoElement });
          },
          width: 640,
          height: 480
        });
        camera.start();

        startScanning();
      } catch (err) {
        console.error("❌ Webcam error:", err);
        alert("❌ Please allow camera access in your browser and refresh.");
      }
    }

    async function populateCameraOptions() {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === "videoinput");

      cameraSelect.innerHTML = "";
      videoDevices.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
      });

      let builtInCamera = videoDevices.find(d =>
        d.label.toLowerCase().includes("integrated") ||
        d.label.toLowerCase().includes("built")
      );

      if (builtInCamera) {
        cameraSelect.value = builtInCamera.deviceId;
        startCameraStream(builtInCamera.deviceId);
      } else if (videoDevices.length > 0) {
        cameraSelect.value = videoDevices[0].deviceId;
        startCameraStream(videoDevices[0].deviceId);
      }

      cameraSelect.addEventListener("change", () => {
        startCameraStream(cameraSelect.value);
      });
    }

    poseCanvas.width = 640;
    poseCanvas.height = 480;

    populateCameraOptions();
  }
});
