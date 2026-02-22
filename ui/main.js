document.addEventListener("DOMContentLoaded", () => {
    console.log("✅ DOM Loaded - V18 (Fun Portrait Reset Button & Modal Cleanup)");

    // --- DOM ELEMENTS ---
    const genderButtons = document.querySelectorAll(".gender-option");
    const eventButtons = document.querySelectorAll(".event-option");
    const welcomeScreen = document.getElementById("welcomeScreen");
    const eventScreen = document.getElementById("eventOptionsScreen");
    const genderScreen = document.getElementById("genderScreen");
    const hautfit = document.getElementById("hautfit");
    const logo = document.getElementById("logo");
    
    // Camera & Canvas
    const videoElement = document.getElementById("cameraVideo");
    const poseCanvas = document.getElementById("poseCanvas");
    const ctx = poseCanvas.getContext("2d");

    const cameraYesBtn = document.getElementById("cameraYesBtn");
    const cameraIntroContainer = document.getElementById("cameraIntroContainer");
    const tryOnModeContainer = document.getElementById("tryOnModeContainer"); 
    const btnImageTryOn = document.getElementById("btnImageTryOn"); 
    const btnVirtualTryOn = document.getElementById("btnVirtualTryOn"); 
    const uiPanel = document.getElementById("uiPanel");

    // --- STATE ---
    let selectedEvent = null;
    let selectedGender = null;
    let currentSessionOutfits = []; 
    let isScanningComplete = false; 
    let tryOnMode = null; 
    let activeRealtimeClient = null; 

    // --- 1. SIZE MAPPING RULES (Asian Fit) ---
    const sizeMapping = {
        "male": { "Inverted Triangle": "L", "Rectangle": "M", "Triangle": "S" },
        "female": { "Inverted Triangle": "L", "Rectangle": "M", "Triangle": "S" }
    };

    // ==========================================
    // 🔄 KIOSK SMART RESET LOGIC
    // ==========================================
    // Flushes memory and instantly routes back to the Event screen
    function resetToEventScreen() {
        sessionStorage.setItem('skipToEvent', 'true');
        location.reload();
    }

    // Check if we just refreshed from a "Generate Again" click
    if (sessionStorage.getItem('skipToEvent') === 'true') {
        sessionStorage.removeItem('skipToEvent');
        if (welcomeScreen) welcomeScreen.style.display = "none";
        if (logo) logo.style.display = "none";
        if (eventScreen) eventScreen.style.display = "block";
    }

    // --- 2. NAVIGATION LOGIC ---
    if (logo && sessionStorage.getItem('skipToEvent') !== 'true') {
      logo.addEventListener("click", () => {
        logo.classList.add("fade-out");
        setTimeout(() => { logo.style.display = "none"; document.getElementById("introText").style.display = "block"; }, 1000);
      });
    }
  
    const getStartedBtn = document.getElementById("get-started-btn");
    if (getStartedBtn) {
      getStartedBtn.addEventListener("click", () => { welcomeScreen.style.display = "none"; eventScreen.style.display = "block"; });
    }
  
    eventButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        selectedEvent = btn.dataset.event; 
        eventScreen.style.display = "none";
        genderScreen.style.display = "block";
      });
    });
  
    genderButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        selectedGender = btn.dataset.gender;
        genderScreen.style.display = "none";
        hautfit.style.display = "block";
        setTimeout(() => hautfit.style.opacity = "1", 200);
      });
    });
  
    if (cameraYesBtn) {
      cameraYesBtn.addEventListener("click", () => {
        cameraIntroContainer.style.display = "none";
        if (tryOnModeContainer) tryOnModeContainer.style.display = "block";
      });
    }

    if (btnImageTryOn) {
        btnImageTryOn.onmouseover = () => btnImageTryOn.style.transform = "scale(1.05)";
        btnImageTryOn.onmouseout = () => btnImageTryOn.style.transform = "scale(1)";
        btnImageTryOn.addEventListener("click", () => {
            tryOnMode = "image";
            startCameraCountdown();
        });
    }

    if (btnVirtualTryOn) {
        btnVirtualTryOn.onmouseover = () => btnVirtualTryOn.style.transform = "scale(1.05)";
        btnVirtualTryOn.onmouseout = () => btnVirtualTryOn.style.transform = "scale(1)";
        btnVirtualTryOn.addEventListener("click", () => {
            tryOnMode = "virtual";
            startCameraCountdown();
        });
    }

    function startCameraCountdown() {
        if(tryOnModeContainer) tryOnModeContainer.style.display = "none";
        
        const msg = document.getElementById("cameraMessage");
        if(msg) msg.style.display = "block";
        
        let dotCount = 1;
        const dotInterval = setInterval(() => { 
            dotCount = (dotCount % 3) + 1; 
            const dots = document.getElementById("dots");
            if(dots) dots.textContent = ".".repeat(dotCount); 
        }, 500);
        
        setTimeout(() => {
          clearInterval(dotInterval);
          if(msg) msg.textContent = "🔍 Scanning started...";
          initPoseDetection(); 
        }, 3000);
    }
  
    // --- 3. MEDIAPIPE & SCANNING ---
    function initPoseDetection() {
      const pose = new Pose({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}` });
      pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  
      const parts = [ { name: "Head", indices: [0] }, { name: "Shoulders", indices: [11, 12] }, { name: "Torso", indices: [23, 24] } ];
      let currentIndex = 0; let lastResults = null;
  
      pose.onResults(results => {
        ctx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
        if (!results.poseLandmarks) return;
        lastResults = results;
        
        if (isScanningComplete) {
            parts.forEach(part => drawBox(part, results, true));
        } else {
            if(currentIndex < parts.length) drawBox(parts[currentIndex], results, true);
        }
      });
  
      function drawBox(part, results, isAnimated) {
        const landmarks = part.indices.map(i => results.poseLandmarks[i]).filter(l => l);
        if (landmarks.length === 0) return;
        const width = poseCanvas.width; const height = poseCanvas.height;
        const xs = landmarks.map(l => l.x * width); const ys = landmarks.map(l => l.y * height);
        const pad = 30;
        const minX = Math.min(...xs) - pad; const minY = Math.min(...ys) - pad;
        const boxW = (Math.max(...xs) + pad) - minX; const boxH = (Math.max(...ys) + pad) - minY;
        
        let alpha = 0.8;
        if (isAnimated) alpha = 0.4 + 0.6 * Math.sin(Date.now() / 200); 

        ctx.strokeStyle = `rgba(74, 222, 128, ${alpha})`;
        ctx.lineWidth = 4;
        ctx.strokeRect(minX, minY, boxW, boxH);
        
        ctx.fillStyle = "white"; ctx.font = "bold 14px Arial";
        ctx.shadowColor = "black"; ctx.shadowBlur = 4;
        ctx.fillText(part.name, minX, minY - 10);
        ctx.shadowBlur = 0; 
      }
  
      function startScanning() {
        if (currentIndex >= parts.length) {
          const msg = document.getElementById("cameraMessage");
          if(msg) msg.textContent = "✅ Scan Complete! Processing...";
          isScanningComplete = true; 
          setTimeout(() => finishScanning(lastResults), 500);
          return;
        }
        const msg = document.getElementById("cameraMessage");
        if(msg) msg.textContent = `🔍 Analyzing ${parts[currentIndex].name}...`;
        setTimeout(() => { currentIndex++; startScanning(); }, 1500); 
      }

      function finishScanning(results) {
        const msg = document.getElementById("cameraMessage");
        try {
            const bodyType = detectBodyType(results);
            const recommendedSize = getRecommendedSize(bodyType, selectedGender);
    
            if(msg) msg.textContent = "⏳ AI Analyzing Skin Tone...";
      
            fetch("/start-camera", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ event: selectedEvent, gender: selectedGender, body_type: bodyType, skin_tone: "unknown" })
            })
            .then(res => res.json())
            .then(data => {
                const aiSkinTone = data.data.skin_tone;
                const finalBodyType = data.data.body_type || bodyType; 
                
                if(msg) msg.textContent = "👕 Matching Wardrobe...";
                return fetch('/recommend', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ gender: selectedGender, event: selectedEvent, skin_tone: aiSkinTone })
                })
                .then(res => res.json())
                .then(recData => {
                    displayResults(recData.recommendations, aiSkinTone, finalBodyType, recommendedSize, false);
                });
            })
            .catch(err => { console.error(err); alert("Server Error: " + err.message); location.reload(); });

        } catch (fatalError) { console.error("CRITICAL ERROR:", fatalError); alert("⚠️ App Crash: " + fatalError.message); location.reload(); }
      }

      function detectBodyType(results) {
        if (!results || !results.poseLandmarks) return "Unknown";
        const lm = results.poseLandmarks;
        const leftShoulderX = lm[11].x; const rightShoulderX = lm[12].x;
        const leftHipX = lm[23].x; const rightHipX = lm[24].x;
        const shoulderWidth = Math.abs(leftShoulderX - rightShoulderX);
        const hipWidth = Math.abs(leftHipX - rightHipX);
        const hipVisible = (lm[23].visibility > 0.5 && lm[24].visibility > 0.5);
        const ratio = hipVisible ? (shoulderWidth / (hipWidth || 1)) : 1.0;
        
        if (ratio > 1.2) return "Inverted Triangle"; 
        if (ratio < 0.9) return "Triangle";          
        return "Rectangle";                          
      }

      function getRecommendedSize(bodyType, gender) {
        if (bodyType === "Unknown") return "Unknown";
        const map = sizeMapping[gender] || sizeMapping["male"];
        return map[bodyType] || "M"; 
      }
  
      async function startCameraStream() {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          videoElement.srcObject = stream;
          videoElement.play();
          const camera = new Camera(videoElement, { onFrame: async () => { await pose.send({ image: videoElement }); }, width: 640, height: 480 });
          camera.start();
          startScanning(); 
        } catch (err) { alert("Camera Error: " + err); }
      }
  
      poseCanvas.width = 640; poseCanvas.height = 480;
      startCameraStream();
    }
  
    // =========================================================
    //      5. DISPLAY RESULTS & WARDROBE
    // =========================================================
    function displayResults(outfits, skinTone, bodyType, recommendedSize, showImmediately = false) {
        currentSessionOutfits = outfits; 
        uiPanel.innerHTML = "";

        const colorGroups = {};
        outfits.forEach(item => {
            const color = item.color_category || "Unknown"; 
            if (!colorGroups[color]) colorGroups[color] = [];
            colorGroups[color].push(item);
        });
  
        const container = document.createElement('div');
        container.id = "results-container";
        container.style.cssText = "background: rgba(0,0,0,0.9); padding: 20px; border-radius: 15px; text-align: center; width: 100%; max-height: 100%; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 0 15px rgba(0,0,0,0.5);";
  
        const statsHTML = `
            <div style="flex-shrink: 0;">
                <h1 style="color: #4ade80; margin: 0 0 10px 0; font-size: 1.8rem;">SCAN COMPLETE!</h1>
                <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 10px; margin-bottom: 15px;">
                    <p style="color: #ccc; margin: 5px 0; font-size: 1.1em;">
                        Skin Tone: <b style="color:#facc15">${skinTone}</b>
                    </p>
                    <p style="color: #ccc; margin: 5px 0; font-size: 1.1em;">
                        Body Type: <b style="color:#facc15">${bodyType}</b>
                    </p>
                    <div style="background: #4ade80; color: black; display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: bold; margin-top: 10px; border: 2px solid white; font-size: 1.1em;">
                        Recommended Size: ${recommendedSize}
                    </div>
                </div>
            </div>
        `;
        container.innerHTML = statsHTML;

        const revealBtn = document.createElement('button');
        revealBtn.innerText = "✨ See Suggested Outfits";
        revealBtn.style.cssText = "padding: 12px 25px; font-size: 1.1em; background: #facc15; border: none; border-radius: 30px; cursor: pointer; font-weight: bold; color: black; transition: transform 0.2s; margin-bottom: 10px; align-self: center;";
        revealBtn.onmouseover = () => revealBtn.style.transform = "scale(1.05)";
        revealBtn.onmouseout = () => revealBtn.style.transform = "scale(1)";
        container.appendChild(revealBtn);

        const wardrobeSection = document.createElement('div');
        wardrobeSection.style.cssText = "display: none; opacity: 0; transition: opacity 0.8s ease; flex-grow: 1; overflow-y: auto; min-height: 0; width: 100%; border-top: 1px solid #555; padding-top: 10px;";
        wardrobeSection.innerHTML = `<p style="color: white; font-size: 1em; margin-bottom: 10px;">👇 Based on your <b>${skinTone}</b> skin, we suggest these colors:</p>`;

        const grid = document.createElement('div');
        grid.id = "outfit-grid";
        grid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; width: 100%; box-sizing: border-box; padding-bottom: 15px;";
  
        if (Object.keys(colorGroups).length === 0) {
            grid.innerHTML = "<p style='color:red;'>No matching outfits found.</p>";
        } else {
            Object.keys(colorGroups).forEach(color => {
                const items = colorGroups[color];
                const card = createCard(items[0].front, color.toUpperCase() + " COLLECTION", () => { 
                    chooseFrontDesign(color, items, skinTone, bodyType, recommendedSize); 
                });
                const badge = document.createElement('span');
                badge.innerText = `${items.length}`;
                badge.style.cssText = "position: absolute; top: 5px; right: 5px; background: #facc15; color: black; font-size: 10px; padding: 2px 6px; border-radius: 10px; font-weight: bold;";
                card.appendChild(badge);
                grid.appendChild(card);
            });
        }
  
        wardrobeSection.appendChild(grid);
        
        // Use the smart reset instead of manual reload
        const restartBtn = document.createElement('button');
        restartBtn.innerText = "🔄 Start Over";
        restartBtn.style.cssText = "margin-bottom: 10px; padding: 8px 20px; border-radius: 20px; border: none; background: #ef4444; color: white; font-weight: bold; cursor: pointer; align-self: center;";
        restartBtn.onclick = resetToEventScreen;
        wardrobeSection.appendChild(restartBtn);

        container.appendChild(wardrobeSection);
        uiPanel.appendChild(container);

        const showWardrobe = () => {
            revealBtn.style.display = "none";
            wardrobeSection.style.display = "flex";
            wardrobeSection.style.flexDirection = "column";
            setTimeout(() => { wardrobeSection.style.opacity = "1"; }, 50);
        };

        if (showImmediately) {
            showWardrobe();
        } else {
            revealBtn.onclick = showWardrobe;
        }
    }

    function chooseFrontDesign(colorName, items, skinTone, bodyType, recommendedSize) {
        const grid = document.getElementById('outfit-grid');
        grid.innerHTML = "";
        
        const parentSection = grid.parentElement;
        if(parentSection.firstChild.tagName === 'P') {
             parentSection.firstChild.innerHTML = `<span style="color:white">Color: <b>${colorName.toUpperCase()}</b></span><br>👇 Step 1: Choose FRONT Design`;
        }

        const backCard = document.createElement('div');
        backCard.style.cssText = "height: 120px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #333; border-radius: 10px; cursor: pointer; border: 2px solid #555;";
        backCard.innerHTML = "<div style='color:white; font-size: 24px;'>🔙</div><p style='color:white; font-size:10px;'>Go Back</p>";
        backCard.onclick = () => { displayResults(currentSessionOutfits, skinTone, bodyType, recommendedSize, true); };
        grid.appendChild(backCard);

        items.forEach(item => {
            const card = createCard(item.front, item.name, () => { 
                chooseBackDesign(item, items, skinTone, bodyType, recommendedSize); 
            });
            grid.appendChild(card);
        });
    }

    function chooseBackDesign(selectedFrontItem, allItems, skinTone, bodyType, recommendedSize) {
        const grid = document.getElementById('outfit-grid');
        grid.innerHTML = "";

        const parentSection = grid.parentElement;
        if(parentSection.firstChild.tagName === 'P') {
             parentSection.firstChild.innerHTML = `<span style="color:#4ade80">Front Selected!</span><br>👇 Step 2: Choose BACK Design`;
        }

        const backCard = document.createElement('div');
        backCard.style.cssText = "height: 120px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #333; border-radius: 10px; cursor: pointer; border: 2px solid #555;";
        backCard.innerHTML = "<div style='color:white; font-size: 24px;'>🔙</div><p style='color:white; font-size:10px;'>Change Front</p>";
        backCard.onclick = () => { chooseFrontDesign(selectedFrontItem.color_category, allItems, skinTone, bodyType, recommendedSize); };
        grid.appendChild(backCard);

        allItems.forEach(item => {
            const card = createCard(item.back, "Back: " + item.name, () => {
                
                const resultsContainer = document.getElementById('results-container');
                if(resultsContainer) {
                    resultsContainer.remove(); 
                }

                console.log(`Starting try-on with mode: ${tryOnMode}`);
                if (tryOnMode === "virtual") {
                    startVirtualTryOn(selectedFrontItem.front); 
                } else {
                    triggerDoubleScan(selectedFrontItem.front, item.back); 
                }

            });
            grid.appendChild(card);
        });
    }

    function createCard(imagePath, labelText, onClickHandler) {
        const card = document.createElement('div');
        card.style.cssText = "cursor: pointer; transition: transform 0.2s; position: relative;";
        const img = document.createElement('img');
        if (imagePath) img.src = imagePath; else img.src = "https://placehold.co/100x120?text=No+Image";
        img.style.cssText = "width: 100%; height: 120px; object-fit: cover; border-radius: 10px; border: 2px solid transparent;";
        card.onmouseenter = () => { img.style.borderColor = "#4ade80"; card.style.transform = "scale(1.05)"; };
        card.onmouseleave = () => { img.style.borderColor = "transparent"; card.style.transform = "scale(1)"; };
        card.onclick = onClickHandler;
        const name = document.createElement('p');
        name.innerText = labelText;
        name.style.cssText = "color: #fff; font-size: 10px; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;";
        card.appendChild(img); card.appendChild(name);
        return card;
    }

    // ==========================================
    // 🪞 ENGINE 1: VIRTUAL TRY-ON (DECART AI)
    // ==========================================
    async function startVirtualTryOn(shirtUrl) {
        const msg = document.getElementById("cameraMessage");
        if(msg) {
            msg.style.display = "block";
            msg.textContent = "✨ Connecting to Virtual Try-On AI...";
        }

        try {
            console.log("1. Fetching secure token from backend...");
            const response = await fetch('/api/decart-token', { method: 'POST' });
            if (!response.ok) throw new Error(`Backend Error: ${response.status}`);
            const data = await response.json();
            
            if (!data.apiKey) throw new Error("API Key was not returned from the server.");

            console.log("2. Fetching local shirt image...");
            const shirtResponse = await fetch(shirtUrl);
            const shirtBlob = await shirtResponse.blob();
            const shirtFile = new File([shirtBlob], "shirt.jpg", { type: "image/jpeg" });

            console.log("3. Dynamically loading Decart SDK...");
            const decartSDK = await import('https://esm.sh/@decartai/sdk');
            const createDecartClient = decartSDK.createDecartClient;
            const models = decartSDK.models;

            console.log("4. Initializing Decart Client...");
            const client = createDecartClient({ apiKey: data.apiKey });
            const realtimeModel = models.realtime("lucy_2_rt");

            const stream = videoElement.srcObject;

            console.log("5. Opening WebRTC connection to Decart servers...");
            activeRealtimeClient = await client.realtime.connect(stream, {
                model: realtimeModel,
                onRemoteStream: (transformedStream) => {
                    console.log("6. WebRTC Stream Active! Injecting video...");
                    poseCanvas.style.display = "none";
                    
                    let outputVideo = document.getElementById('smartMirrorVideo');
                    if (!outputVideo) {
                        outputVideo = document.createElement('video');
                        outputVideo.id = 'smartMirrorVideo';
                        outputVideo.style.cssText = "position: absolute; top:0; left:0; width:100%; height:100%; object-fit: cover; z-index: 5; transform: scaleX(-1);";
                        videoElement.parentElement.appendChild(outputVideo);
                    }
                    
                    outputVideo.srcObject = transformedStream;
                    outputVideo.play();
                    
                    if(msg) msg.textContent = "✨ Virtual Try-On Active! Move around.";

                    // 🚨 INJECTING FUN PORTRAIT BUTTON FOR VIRTUAL TRY-ON 🚨
                    const exitBtn = document.createElement('button');
                    exitBtn.innerHTML = "✨ 🔄 Start New Try-On ✨";
                    exitBtn.id = "virtual-reset-btn";
                    // Fun, bouncy, gradient portrait styling
                    exitBtn.style.cssText = `
                        position: fixed;
                        bottom: 8%;
                        left: 50%;
                        transform: translateX(-50%);
                        z-index: 9999;
                        padding: 18px 40px;
                        border-radius: 50px;
                        background: linear-gradient(45deg, #FF512F, #DD2476); /* Vibrant gradient */
                        color: white;
                        font-weight: 800;
                        border: 3px solid rgba(255,255,255,0.8);
                        font-size: 1.3rem;
                        cursor: pointer;
                        box-shadow: 0 15px 35px rgba(221, 36, 118, 0.5);
                        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy transition */
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        letter-spacing: 1px;
                    `;
                    // Bouncy hover effect
                    exitBtn.onmouseover = () => exitBtn.style.transform = "translateX(-50%) scale(1.08) translateY(-5px)";
                    exitBtn.onmouseout = () => exitBtn.style.transform = "translateX(-50%) scale(1) translateY(0)";
                    exitBtn.onclick = resetToEventScreen; 
                    document.body.appendChild(exitBtn);
                }
            });

            console.log("7. Sending shirt image to AI...");
            await activeRealtimeClient.set({
                prompt: "Professional virtual try-on. The person is wearing the exact shirt from the reference image naturally.",
                image: shirtFile,
                enhance: true
            });

        } catch (err) { 
            console.error("Virtual Try-On Error:", err);
            alert("Virtual Try-On Error: " + err.message + "\n\nPress F12 and check the Console for more details."); 
        }
    }

    // ==========================================
    // 📸 ENGINE 2: IMAGE TRY-ON (REPLICATE API)
    // ==========================================
    async function triggerDoubleScan(frontPath, backPath) {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.style.cssText = "position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.9); z-index: 1000; display: flex; flex-direction: column; align-items: center; justify-content: center; color: white; text-align: center;";
        document.body.appendChild(overlay);
        try {
            overlay.innerHTML = `
                <div style="font-size: 60px; margin-bottom: 20px;">📸</div>
                <h1 style="color: #facc15; font-size: 35px; margin-bottom: 10px;">Position Yourself!</h1>
                <h2 style="color: white; font-weight: normal;">You are about to be scanned for Front and Back views.</h2>
            `;
            await new Promise(r => setTimeout(r, 4000)); 

            await runCountdown(overlay, "Front View", "Look at the camera");
            const userFrontImg = captureImage();
            await flashScreen(overlay);

            await runCountdown(overlay, "Back View", "Turn around!");
            const userBackImg = captureImage();
            await flashScreen(overlay);

            overlay.innerHTML = `<div style="font-size: 50px; animation: spin 2s infinite linear;">⚙️</div><h2 style="margin-top: 20px;">Generating Front...</h2><p>Please wait...</p><style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>`;
            const frontResponse = await fetch('/generate-tryon', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_image: userFrontImg, shirt_path: frontPath }) });
            const frontResult = await frontResponse.json();
            if (frontResult.error) throw new Error("Front Error: " + frontResult.error);

            overlay.innerHTML = `<div style="font-size: 50px; animation: spin 2s infinite linear;">⚙️</div><h2 style="margin-top: 20px;">Generating Back...</h2><p>Almost there...</p>`;
            const backResponse = await fetch('/generate-tryon', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_image: userBackImg, shirt_path: backPath }) });
            const backResult = await backResponse.json();
            if (backResult.error) throw new Error("Back Error: " + backResult.error);
            
            showDualResult(frontResult.generated_image, backResult.generated_image);
        } catch (err) { alert("Try-On Failed: " + err.message); } finally { if(document.getElementById('loading-overlay')) document.getElementById('loading-overlay').remove(); }
    }
  
    async function runCountdown(overlay, title, instruction) {
        for (let i = 3; i > 0; i--) {
            overlay.innerHTML = `<h1 style="color: #4ade80; font-size: 40px;">${title}</h1><h2 style="margin-bottom: 20px;">${instruction}</h2><div style="font-size: 100px; font-weight: bold; animation: pop 0.5s ease;">${i}</div>`;
            await new Promise(r => setTimeout(r, 1000));
        }
    }
  
    function captureImage() {
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = videoElement.videoWidth; captureCanvas.height = videoElement.videoHeight;
        const capCtx = captureCanvas.getContext('2d');
        capCtx.translate(captureCanvas.width, 0);
        capCtx.scale(-1, 1);
        capCtx.drawImage(videoElement, 0, 0);
        return captureCanvas.toDataURL('image/jpeg', 0.8);
    }
  
    async function flashScreen(overlay) {
        const oldBg = overlay.style.background; overlay.style.background = "white"; overlay.innerHTML = "";
        await new Promise(r => setTimeout(r, 100)); overlay.style.background = oldBg;
    }
  
    // 🚨 UPDATED MODAL: Cleaned up and injects FUN button on CLOSE 🚨
    function showDualResult(frontUrl, backUrl) {
        const modal = document.createElement('div');
        modal.id = "photo-tryon-modal";
        modal.style.cssText = "position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.95); z-index: 2000; display: flex; flex-direction: column; align-items: center; justify-content: center;";
        
        modal.innerHTML = `
            <h2 style="color: white; margin-bottom: 10px;">✨ Your Look ✨</h2>
            <p style="color: #4ade80; margin-bottom: 20px; font-size: 0.9rem;">(Click an image to view full size)</p>
            
            <div style="display: flex; flex-direction: column; gap: 20px; justify-content: center; align-items: center; overflow-y: auto; width: 100%; max-height: 75vh; padding: 10px;">
                <div style="text-align: center;">
                    <p style="color:white; margin-bottom:5px; font-weight: bold;">Front</p>
                    <img src="${frontUrl}" onclick="window.open('${frontUrl}', '_blank')" style="cursor: pointer; max-height: 30vh; border-radius: 10px; border: 2px solid #4ade80; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                </div>
                <div style="text-align: center;">
                    <p style="color:white; margin-bottom:5px; font-weight: bold;">Back</p>
                    <img src="${backUrl}" onclick="window.open('${backUrl}', '_blank')" style="cursor: pointer; max-height: 30vh; border-radius: 10px; border: 2px solid #4ade80; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                </div>
            </div>
            
            <div style="margin-top: 25px;">
                <button id="close-modal" style="padding: 12px 35px; border-radius: 25px; border: none; font-weight: bold; font-size: 1.1rem; cursor: pointer; background: white; color: black; transition: background 0.2s;" onmouseover="this.style.background='#facc15'" onmouseout="this.style.background='white'">Close</button>
            </div>
        `;
        
        document.body.appendChild(modal); 

        // 🚨 INJECTING FUN PORTRAIT BUTTON ON CLOSE 🚨
        document.getElementById('close-modal').onclick = () => {
            modal.remove();
            
            const exitBtn = document.createElement('button');
            exitBtn.innerHTML = "✨ 🔄 Start New Try-On ✨";
            exitBtn.id = "photo-reset-btn";
            // Fun, bouncy, gradient portrait styling
            exitBtn.style.cssText = `
                position: fixed;
                bottom: 8%;
                left: 50%;
                transform: translateX(-50%);
                z-index: 9999;
                padding: 18px 40px;
                border-radius: 50px;
                background: linear-gradient(45deg, #FF512F, #DD2476); /* Vibrant gradient */
                color: white;
                font-weight: 800;
                border: 3px solid rgba(255,255,255,0.8);
                font-size: 1.3rem;
                cursor: pointer;
                box-shadow: 0 15px 35px rgba(221, 36, 118, 0.5);
                transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Bouncy transition */
                display: flex;
                align-items: center;
                gap: 10px;
                letter-spacing: 1px;
            `;
            // Bouncy hover effect
            exitBtn.onmouseover = () => exitBtn.style.transform = "translateX(-50%) scale(1.08) translateY(-5px)";
            exitBtn.onmouseout = () => exitBtn.style.transform = "translateX(-50%) scale(1) translateY(0)";
            exitBtn.onclick = resetToEventScreen; 
            document.body.appendChild(exitBtn);
        };
    }
});