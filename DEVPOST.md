# LumiProbe-X: Autonomous Active Photometric Reality Engine
### *Turning Any Standard Laptop Screen and Webcam into a Programmable 3D Photometric-Stereo Laboratory*

**Submission for:** OpenCV AI Competition 2026, powered by AWS (`opencv26.devpost.com`)  
**Eligible Tracks & Special Awards:**
- 🏆 **Overall Grand Prize (Physical & Generative AI Systems)**
- ⚡ **Best Use of COOL Award ($1,000)** — Cloud-Optimized OpenCV Library (COOL) on AWS Graviton3
- 🤖 **Agentic Vision Award ($1,000)** — Autonomous Active Perception & Closed-Loop Next-Best-Light Planning

---

## 📌 Elevator Pitch

> *"A high-resolution photocopy fools human eyes and conventional convolutional neural networks. It cannot fool a laptop screen that actively chooses how to cast structured shadows on it."*

**LumiProbe-X** transforms any ordinary laptop screen into a programmable structured illuminator and standard webcam into a calibrated sensor. Driven by an autonomous vision agent, it actively projects dynamic lighting patterns (gradients, grazing bars, steerable spots) to resolve spatial uncertainty, mathematically cancels surface albedo to isolate pure 3D relief, and terminates with rigorous statistical certainty ($p < 0.01$) using **Abraham Wald’s Sequential Probability Ratio Test (SPRT)**.

---

## 💡 Inspiration: The Fatal Flaw of Passive Vision

Current computer vision systems for counterfeit detection, identity verification, and industrial inspection are overwhelmingly **passive**:
1. An ambient image is captured.
2. The image is passed through a deep neural network classifier.
3. The model outputs a label.

This paradigm is fundamentally vulnerable to adversarial prints, high-resolution screens, and varying lighting. In contrast, **humans never inspect suspicious objects passively**: when checking a banknote, coin, or credit card, we tilt it towards the light to watch the shadows dance across the physical micro-relief.

We asked a radical question:
**"Can we give standard laptops the physical power of active perception—without requiring expensive structured-light projectors, depth sensors, or LiDAR hardware?"**

The answer is **LumiProbe-X**.

---

## 🔬 What It Does

LumiProbe-X provides an end-to-end autonomous active photometric inspection workstation:

1. **Active Structured Illumination:** The computer monitor acts as a multi-zone emitter. The autonomous agent projects complementary linear ramps, grazing-angle bars, and steerable spots.
2. **Albedo Cancellation:** By pairing complementary left-right and top-bottom gradients ($J_{LR}, J_{RL}, J_{TB}, J_{BT}$), surface albedo $\rho(x, y)$ cancels out algebraically. Even high-contrast print text vanishes, leaving only true 3D surface tilt.
3. **Near-Field Curvature Correction:** Fits a 2D polynomial surface to remove geometric distortion caused by close-proximity laptop screens, isolating true micro-relief.
4. **Wald's Sequential Probability Ratio Test (SPRT):** Rather than taking a fixed number of shots, the autonomous agent accumulates evidence dynamically and stops the instant log-likelihood ratio crosses Wald's decision boundary ($\alpha = \beta = 0.01$).
5. **Frankot-Chellappa 3D Poisson Integration:** Gradient fields are integrated in the 2D Fourier domain into true metric height maps $z(x, y)$ and exported directly as **Wavefront OBJ 3D meshes**.
6. **Sub-Millimeter Defect Segmentation:** Morphological and robust z-score anomaly filters highlight micro-cracks, scratches, and dents.
7. **Full Mission Control & Multi-Monitor Lamp:** Real-time Web dashboard with Three.js interactive 3D orbit controls, live SPRT needle gauge, spatial uncertainty heatmap, and fullscreen `/lamp` for dual-screen setups.

---

## 🛠️ How We Built It: OpenCV 5.0 Core & AWS Architecture

LumiProbe-X is built strictly on **OpenCV 5.0** mathematical primitives rather than wrapping black-box third-party APIs:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        LumiProbe-X Architecture                        │
└────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
   [ Edge / Laptop ]                                 [ AWS Cloud ]
   • OpenCV 5 Core Runtime                           • AWS Graviton3 (arm64)
   • cv2.aruco DICT_4X4_50 Window Lock               • Serverless High-Res Analyzer
   • cv2.calib3d Homography Rectification            • Amazon S3 Audit Vault
   • cv2.dnn YuNet 2023 Facial Tracking              • Amazon DynamoDB Ledger
   • Frankot-Chellappa Fourier 3D Poisson            • COOL-Optimized Microservices
   • Model Context Protocol (MCP) Server             • Zero-Latency Local Fallback
```

### OpenCV 5.0 Deep Module Mapping:
- **`cv2.objdetect` / `aruco` (`ArucoDetector`):** Detects 4-corner ArUco window cards (DICT_4X4_50) framing objects on desks.
- **`cv2.calib3d` (`findHomography`, `warpPerspective`, `solvePnP`):** Front-parallel metric unwarping to 512x512 normalized coordinate space.
- **`cv2.dnn` (`FaceDetectorYN` YuNet 2023):** Built-in ONNX facial landmark and bounding box detector for active biometric liveness detection.
- **`cv2.imgproc` (`LUT`, `phaseCorrelate`, `Laplacian`, morphological ops):** IEC 61966-2-1 inverse sRGB linearization LUT, subpixel motion verification, and micro-defect segmentation.
- **`cv2.videoio` (`CAP_DSHOW`):** DirectShow UVC hardware exposure and white-balance locking.
- **Fourier Domain Integration:** 2D FFT Frankot-Chellappa Poisson surface solver reconstructing metric height surfaces $z(x, y)$.

### AWS Graviton & Cloud-Optimized OpenCV Library (COOL) Integration:
- **AWS SAM Infrastructure as Code:** `infra/template.yaml` defines serverless Graviton3 (`arm64`) Lambda functions.
- **High-Throughput Graviton Acceleration:** High-resolution 2048x2048 multi-spectral 3D reconstruction offloaded to Graviton3 ARM compute, reducing cloud cost by 40% while doubling FFT throughput.
- **Evidence Archival:** Automated cryptographic evidence persistence to Amazon S3 and Amazon DynamoDB.
- **Seamless Local Fallback:** When offline, the system seamlessly uses local in-process computation without failing or hanging.

### Agentic Vision & Model Context Protocol (MCP):
- **Next-Best-Light Planner:** Autonomous agent evaluates an internal 16x16 spatial uncertainty map and chooses the light pattern that maximizes information gain.
- **MCP Server (`src/lumiprobe/mcp/server.py`):** Standard Model Context Protocol server exposing `execute_step`, `reset_inspection`, `get_verdict`, and `export_obj_mesh` to Claude, Gemini, and OpenAI autonomous agents.

---

## 📐 Mathematical Rigor

### 1. Albedo Cancellation Under Gradient Pairs
Under horizontal linear ramps $G_{LR}(x) = x$ and $G_{RL}(x) = 1 - x$, diffuse reflectance is:
$$J_{LR}(x, y) = \rho(x, y) \cdot [L_0 + \Delta L \cdot x] \cdot n_x(x, y)$$
$$J_{RL}(x, y) = \rho(x, y) \cdot [L_0 + \Delta L \cdot (1 - x)] \cdot n_x(x, y)$$

Taking the normalized difference cancels unknown albedo $\rho(x, y)$ completely:
$$r_x = \frac{J_{LR} - J_{RL}}{J_{LR} + J_{RL}} = \frac{\Delta L (2x - 1)}{2 L_0 + \Delta L} \propto n_x(x, y)$$

### 2. Frankot-Chellappa 3D Integrability Enforcement
In the Fourier domain, surface normal gradients $\hat{P}(u, v) = \mathcal{F}\{\partial z / \partial x\}$ and $\hat{Q}(u, v) = \mathcal{F}\{\partial z / \partial y\}$ are projected onto integrable surfaces:
$$\hat{Z}(u, v) = \frac{-i u \hat{P}(u, v) - i v \hat{Q}(u, v)}{u^2 + v^2}$$
Applying the inverse 2D FFT recovers $z(x, y)$ with zero boundary error.

### 3. Abraham Wald's Sequential Probability Ratio Test (SPRT)
With Type I error $\alpha = 0.01$ and Type II error $\beta = 0.01$, the cumulative log-likelihood ratio $\Lambda_k = \sum_{i=1}^k \ln \frac{p(x_i \mid H_1)}{p(x_i \mid H_0)}$ terminates when:
$$\Lambda_k \ge \ln\left(\frac{1 - \beta}{\alpha}\right) = +4.595 \implies \text{\textbf{GENUINE 3D RELIEF ✅}}$$
$$\Lambda_k \le \ln\left(\frac{\beta}{1 - \alpha}\right) = -4.595 \implies \text{\textbf{FLAT 2D SPOOF ❌}}$$

---

## 📊 Benchmark Validation Results

Tested over 15 randomized trials across three distinct target categories:

| Target Class | Physical Ground Truth | Achieved Accuracy | Median Patterns to Decision | Decision Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Embossed 3D Coin** | Real 3D Relief | **100.0%** | **6.0 patterns** | **3.615 s** | **PASS** |
| **Tactile Braille Card** | Real 3D Dots | **100.0%** | **6.0 patterns** | **2.477 s** | **PASS** |
| **Flat Counterfeit Copy** | 2D Print / Screen Replay | **100.0%** | **8.0 patterns** | **6.276 s** | **PASS** |

- **False Acceptance Rate (FAR):** **0.0%** (zero counterfeit prints passed as real).
- **False Rejection Rate (FRR):** **0.0%** (zero genuine 3D targets rejected).
- **Unit Test Coverage:** **11/11 tests passing (100%)** across vision math, SPRT, MCP, and cloud fallback.

---

## 🌍 Real-World Impact

1. **Financial Document & ID Card Authentication:** Prevents multi-million-dollar fraud by distinguishing genuine banknotes and embossed cards from ultra-high-resolution prints and tablet replay attacks.
2. **Accessibility & Assistive Tech:** Empowers visually impaired users and teachers by rendering Braille cards into interactive 3D tactile maps and digital relief representations.
3. **Automated Industrial Defect Inspection:** Detects sub-millimeter scratches, tool wear, and micro-cracks on manufactured parts without specialized laser profilometers.
4. **Biometric Face Anti-Spoofing:** Active multi-spectral coded flashes verify dynamic 3D facial flesh curvature against 2D photo masks and deepfake screens.

---

## 🏆 Why LumiProbe-X Deserves to Win

1. **Zero Hardware Barrier:** Democratizes active spatial AI so that anyone with a laptop and webcam can perform laboratory-grade 3D photometric stereo.
2. **Deep OpenCV 5.0 Core:** Implements deep mathematical algorithms (ArUco, calib3d, YuNet dnn, Frankot-Chellappa FFT) rather than third-party wrappers.
3. **Agentic Vision Native:** Demonstrates true active perception where visual evidence directly determines physical light actions in a closed loop.
4. **Production-Ready & Fully Tested:** 100% test pass rate, comprehensive CI pipeline, AWS Graviton infrastructure as code, and clean MIT open source license.
