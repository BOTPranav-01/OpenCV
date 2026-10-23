# LumiProbe-X: Official Video Demonstration Script
### *Hackathon Submission Video Guide (Target: 2:30 - 3:00 minutes • Maximum allowed: 5:00 minutes)*

This script is structured specifically to meet all judging criteria for **OpenCV AI Competition 2026, powered by AWS** (`opencv26.devpost.com`).

---

## ⏱️ Video Timeline Overview

| Timestamp | Segment Title | Visual On-Screen | Speaker Talking Points |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:25** | **The Hook & Problem** | Side-by-side coin & color print | Why passive vision & CNNs fail against prints |
| **0:25 – 0:50** | **The Solution & Setup** | Laptop screen + desk target | How LumiProbe-X turns any screen into active light |
| **0:50 – 1:30** | **Live Demo: Flat Print vs Real Coin** | Mission Control Dashboard (`localhost:8000`) | Flashing patterns, SPRT needle swing, 3D mesh orbit |
| **1:30 – 1:55** | **OpenCV 5 & AWS Graviton Architecture** | Architecture slide & code | ArUco, calib3d, YuNet, Frankot-Chellappa FFT & Graviton |
| **1:55 – 2:20** | **Impact & Future Tech** | Braille relief & Defect heatmap | Financial security, accessibility, industrial inspection |
| **2:20 – 2:40** | **Conclusion & Call to Action** | GitHub repo & team credits | Active perception in OpenCV 5: light as action |

---

## 🎙️ Step-by-Step Script & Narration

### [0:00 – 0:25] Segment 1: The Hook & Introduction
- **Visual:** Presenter appears on camera holding two items: a real embossed metal coin in one hand, and a high-resolution color photocopy of the same coin in the other hand.
- **Narrator (Pranav):**
  > "Hello OpenCV 2026 judges! To a standard webcam or a conventional deep learning model, these two items look virtually identical. A high-resolution printout easily fools passive 2D classifiers.
  > But it cannot fool a computer screen that actively chooses how to cast structured shadows on it!
  > Welcome to **LumiProbe-X: The Autonomous Active Photometric Reality Engine**."

---

### [0:25 – 0:50] Segment 2: The Core Concept & Zero-Hardware Democratization
- **Visual:** Cut to screen recording or camera shot of the laptop setup. Show the laptop screen tilted down at 100° over a desk with the window card template printed or on screen.
- **Narrator:**
  > "Rather than requiring thousands of dollars in laser scanners, depth cameras, or LiDAR sensors, LumiProbe-X democratizes spatial 3D inspection using nothing more than your standard laptop screen and webcam.
  > Driven by OpenCV 5.0 and an autonomous vision agent, our system uses the screen as a programmable structured illuminator and the webcam as a calibrated sensor."

---

### [0:50 – 1:30] Segment 3: Live Application Demonstration
- **Visual:** Fullscreen screen recording of the **Mission Control Dashboard** (`http://localhost:8000`).
- **Narrator:**
  > "Let's test this in action on our Mission Control Dashboard.
  > First, we place a flat counterfeit print under the sensor. Notice how the agent executes an active illumination sequence: dark calibration, uniform white, and complementary linear gradients.
  > Under our gradient pairs, the surface albedo cancels out completely. Because there is no physical relief, the Wald SPRT statistical test rapidly detects zero residual energy. In just 8 steps, the SPRT needle swings hard into the red zone: **FLAT 2D SPOOF REJECTED** with 99% confidence!
  > Now, let's switch to the real embossed 3D coin. The agent dynamically projects directional grazing bars and steerable spots. In only 6 steps, the needle surges into the green zone: **VERIFIED 3D RELIEF**!
  > Instantly, our Frankot-Chellappa Fourier solver reconstructs a true 3D surface mesh. We can orbit the portrait relief in real-time in Three.js and export it directly as a 3D Wavefront OBJ file!"

---

### [1:30 – 1:55] Segment 4: OpenCV 5.0 Core & AWS Graviton Architecture
- **Visual:** Show architectural diagram (Mermaid flowchart from README) followed by a quick terminal shot of pytest passing 11/11 tests.
- **Narrator:**
  > "Under the hood, LumiProbe-X is built specifically for OpenCV 5.0:
  > - `cv2.aruco` and `cv2.calib3d` lock onto perspective-warped markers for metric rectification.
  > - `cv2.dnn` runs built-in YuNet for facial liveness anti-spoofing.
  > - `cv2.imgproc` handles inverse sRGB LUT linearization and subpixel motion checks.
  > - And our 2D FFT Frankot-Chellappa algorithm reconstructs integrable 3D height surfaces.
  > For cloud scalability, LumiProbe-X runs on **AWS Graviton3** via serverless ARM64 Lambda functions and Amazon S3 evidence vaults—achieving 2x higher FFT throughput at 40% lower compute cost under the Cloud-Optimized OpenCV Library (COOL) track, with seamless offline local fallback."

---

### [1:55 – 2:20] Segment 5: Real-World Impact & Applications
- **Visual:** Show the dashboard switching to 'Braille Card' mode (dots pop up in 3D) and 'Defect Heatmap' mode (sub-millimeter scratches highlighted in neon red).
- **Narrator:**
  > "LumiProbe-X solves critical real-world bottlenecks:
  > 1. **Financial Security:** Eliminating counterfeit currency, forged checks, and fake ID cards.
  > 2. **Accessibility Tech:** Converting physical Braille cards into interactive 3D relief tactile models.
  > 3. **Industrial Quality Control:** Detecting micro-grooves, dents, and surface cracks on manufactured parts without expensive profilometers."

---

### [2:20 – 2:40] Segment 6: Closing & Team Sign-off
- **Visual:** Presenter back on screen with GitHub repo `https://github.com/BOTPranav-01/OpenCV` on display.
- **Narrator:**
  > "LumiProbe-X brings physical AI to life: visual understanding driving physical light action in real time.
  > Our code is 100% open-source under the MIT license on GitHub, fully tested, and ready to deploy.
  > Thank you to OpenCV and AWS for this incredible challenge!"

---

## 💡 Quick Tips for Recording:
1. **Lighting:** Record in a moderately dim room so the laptop screen lighting reflections are clearly visible on your desk or test object.
2. **Audio:** Use a clear microphone or headset to ensure pristine audio.
3. **Upload:** Upload to YouTube as **Public** or **Unlisted**, copy the link, and paste it into the Devpost submission field!
