import cv2
import mediapipe as mp
import time
import math
import numpy as np

class SmoothedLandmark:
    """Represents a temporally smoothed 3D landmark point."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class GestureDetector:
    """
    Handles webcam frame processing, hand landmark extraction,
    and gesture classification using MediaPipe Hands.
    Stage 4: Highlights-Clamping Gamma LUT, robust coordinate-relative classifiers, and direct RGB output.
    """
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # State variables for multi-hand temporal landmark smoothing
        self.prev_smoothed_left = None
        self.prev_smoothed_right = None
        
        # Rolling FPS estimation variables
        self.prev_time = time.time()
        self.fps = 0.0
        self.fps_timings = []
        self.max_fps_history = 18 # Smooths FPS numbers cleanly
        
        # Cinematic Vignette precomputed mask
        self.vignette_mask_uint8 = None
        
        # Precompute fast combined Scale-Balancing & Highlights-Clamping Gamma correction Lookup Table (LUT)
        # 1. Lens Balance exposure scaling (alpha=0.88, beta=-8)
        # 2. Gamma 0.65 pulls down harsh ceiling lights and exposure reflections.
        # 3. Highlights above 190 are smoothly compressed with 0.18 slope.
        # 4. Peak brightness is clamped at 215 to prevent digital highlight clipping.
        # Computing both in a single pass completely eliminates cv2.convertScaleAbs!
        gamma = 0.65
        inv_gamma = 1.0 / gamma
        lut = []
        for i in range(256):
            # Apply exposure adjustment: i * 0.88 - 8
            adj = i * 0.88 - 8
            adj = max(0, min(255, int(adj)))
            
            # Apply Gamma LUT
            val = ((adj / 255.0) ** inv_gamma) * 255
            # Scale down highlights and clamp peak brightness
            if val > 190:
                val = 190 + (val - 190) * 0.18
            lut.append(min(int(val), 215))
            
        self.combined_lut = np.array(lut).astype("uint8")

    def calculate_fps(self):
        """Calculates a smoothed rolling average of FPS over the last N frames."""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        self.prev_time = current_time
        
        if time_diff > 0:
            self.fps_timings.append(time_diff)
            if len(self.fps_timings) > self.max_fps_history:
                self.fps_timings.pop(0)
                
            avg_diff = sum(self.fps_timings) / len(self.fps_timings)
            self.fps = 1.0 / avg_diff
        return self.fps

    def _precompute_vignette_mask(self, h, w):
        """Generates a 3-channel radial dark vignette mask for ultra-fast C++ SIMD scaling."""
        center_y, center_x = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        max_dist = (h / 2.0)**2 + (w / 2.0)**2
        
        # Soft dark fade toward edges
        mask = 1.0 - (dist_sq / max_dist) * 0.45
        mask = np.clip(mask, 0.45, 1.0)
        
        # Pre-scale mask to 0..255 and stack 3 channels to completely bypass channel broadcasting in C++!
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        self.vignette_mask_uint8 = (mask_3ch * 255).astype(np.uint8)

    @staticmethod
    def get_distance(pt1, pt2):
        """Calculates Euclidean distance between two 3D landmarks."""
        return math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2 + (pt1.z - pt2.z)**2)

    @staticmethod
    def get_distance_2d(pt1, pt2):
        """
        Calculates 2D XY-only Euclidean distance between two landmarks.
        Use for classifier features sensitive to z-axis jitter (tip separations,
        V-spread, palm centroid distances) where depth noise amplifies error.
        """
        return math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)

    def classify_gesture(self, landmarks):
        """
        Classifies the hand gesture based on continuous, weighted feature scoring
        and a probabilistic competitive selection pool with margin thresholding.
        """
        # 1. Wrist reference
        wrist = landmarks[0]
        
        # 2. Scale-invariant hand normalization factor
        # Palm size: 2D XY distance from wrist (0) to middle finger knuckle (9).
        # Using 2D here prevents z-axis jitter from inflating/deflating the normalization baseline.
        palm_size = self.get_distance_2d(landmarks[0], landmarks[9])
        palm_size = max(palm_size, 0.001)
        
        # Palm center centroid (Wrist + Knuckles 5, 9, 13, 17)
        palm_center_x = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5.0
        palm_center_y = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5.0
        palm_center_z = (landmarks[0].z + landmarks[5].z + landmarks[9].z + landmarks[13].z + landmarks[17].z) / 5.0
        
        class Point:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        palm_center = Point(palm_center_x, palm_center_y, palm_center_z)
        
        # Knuckles average y
        knuckle_avg_y = (landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 4.0
        
        # Helper to clamp values
        def clamp(val, min_v=0.0, max_v=1.0):
            return max(min_v, min(max_v, val))
            
        # 3. Perspective-Invariant Local Tip-to-MCP Knuckle Extensions
        # Using 2D XY distances: z-depth from MediaPipe is noisy, and tip extension is
        # fundamentally a 2D vertical projection in camera space. This eliminates
        # z-jitter amplification that was causing Peace Sign and Thumbs Up instability.
        index_tip_knuckle = self.get_distance_2d(landmarks[8], landmarks[5]) / palm_size
        middle_tip_knuckle = self.get_distance_2d(landmarks[12], landmarks[9]) / palm_size
        ring_tip_knuckle = self.get_distance_2d(landmarks[16], landmarks[13]) / palm_size
        pinky_tip_knuckle = self.get_distance_2d(landmarks[20], landmarks[17]) / palm_size

        index_ext = clamp((index_tip_knuckle - 0.35) / (0.75 - 0.35))
        middle_ext = clamp((middle_tip_knuckle - 0.35) / (0.80 - 0.35))
        ring_ext = clamp((ring_tip_knuckle - 0.32) / (0.75 - 0.32))
        pinky_ext = clamp((pinky_tip_knuckle - 0.30) / (0.65 - 0.30))
        
        index_curl = 1.0 - index_ext
        middle_curl = 1.0 - middle_ext
        ring_curl = 1.0 - ring_ext
        pinky_curl = 1.0 - pinky_ext
        
        # Average finger curl and extension
        avg_finger_curl = (index_curl + middle_curl + ring_curl + pinky_curl) / 4.0
        avg_finger_ext = (index_ext + middle_ext + ring_ext + pinky_ext) / 4.0
        
        # Min and max finger extensions
        min_finger_ext = min(index_ext, middle_ext, ring_ext, pinky_ext)
        
        # 4. Compute Thumb features
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        # Thumb extension: 2D tip-to-index-knuckle distance normalized
        # Using 2D avoids z-noise inflating/deflating the extension score.
        thumb_index_dist = self.get_distance_2d(thumb_tip, landmarks[5]) / palm_size
        thumb_ext_score = clamp((thumb_index_dist - 0.85) / (1.20 - 0.85))
        
        # ========================================================
        # INDEPENDENT GESTURE ESTIMATIONS
        # ========================================================
        
        # A. Rock On Score 🤘 (replaces Thumbs Up — mapped to Naruto)
        # Definition: Index + Pinky extended. Middle + Ring curled. Thumb relaxed (not required).
        #
        # Feature breakdown:
        #   30% Index extended
        #   30% Pinky extended
        #   20% Middle curled  (anti-Open Palm gate)
        #   20% Ring curled    (anti-Open Palm gate)
        #
        # Rock On is geometrically maximally distinct from all three other gestures:
        #   vs Fist:      index + pinky are clearly extended (Fist has all curled)
        #   vs Open Palm: middle + ring are clearly curled   (Open Palm has all extended)
        #   vs Peace:     pinky is extended + middle curled  (Peace has middle up, pinky down)
        rock_on_score = (0.30 * index_ext) + (0.30 * pinky_ext) + (0.20 * middle_curl) + (0.20 * ring_curl)
        
        # Hard isolation gates — both must fire to confirm Rock On:
        # 1. Index and Pinky must each be meaningfully extended (> 0.45).
        # 2. Middle and Ring must each be meaningfully curled (> 0.45).
        # These gates collapse the score to near-zero in ambiguous partial poses.
        index_gate  = clamp((index_ext  - 0.45) / 0.25)
        pinky_gate  = clamp((pinky_ext  - 0.45) / 0.25)
        middle_gate = clamp((middle_curl - 0.45) / 0.25)
        ring_gate   = clamp((ring_curl   - 0.45) / 0.25)
        rock_on_score *= index_gate * pinky_gate * middle_gate * ring_gate
        
        # B. Peace Sign Score
        # 25% Index open + 25% Middle open + 15% Ring curled + 15% Pinky curled + 20% V-shape spread
        # v_dist switched to 2D: fingertip separation in XY is what the camera sees; z adds noise not signal.
        v_dist = self.get_distance_2d(landmarks[8], landmarks[12]) / palm_size
        v_sep_score = clamp((v_dist - 0.30) / (0.60 - 0.30))  # Lowered lower bound: 0.35->0.30 for tolerance
        peace_sign_score = (0.25 * index_ext) + (0.25 * middle_ext) + (0.15 * ring_curl) + (0.15 * pinky_curl) + (0.20 * v_sep_score)
        
        # Peace Sign curl gate: ring and pinky must be meaningfully curled (> 0.35) to prevent
        # Open Palm from bleeding into Peace Sign during transitions.
        peace_curl_gate = clamp((ring_curl + pinky_curl) / 2.0 - 0.35) / 0.35
        peace_sign_score *= clamp(peace_curl_gate)
        
        # C. Open Palm Score
        # Finger spread score: 2D distance between adjacent fingertips (z-noise irrelevant for spread).
        dist_8_12 = self.get_distance_2d(landmarks[8], landmarks[12]) / palm_size
        dist_12_16 = self.get_distance_2d(landmarks[12], landmarks[16]) / palm_size
        dist_16_20 = self.get_distance_2d(landmarks[16], landmarks[20]) / palm_size
        avg_spread = (dist_8_12 + dist_12_16 + dist_16_20) / 3.0
        spread_score = clamp((avg_spread - 0.22) / 0.38)
        
        # 45% Average extension + 20% Spread score + 20% Thumb extension + 15% Min extension consistency
        open_palm_score = (0.45 * avg_finger_ext) + (0.20 * spread_score) + (0.20 * thumb_ext_score) + (0.15 * min_finger_ext)
        
        # High-order consistency multiplier: if any finger is curled below 0.38, Open Palm drops sharply to zero
        palm_consistency = clamp((min_finger_ext - 0.38) / 0.38)
        open_palm_score *= palm_consistency
        
        # D. Fist Score
        # 45% Average curl + 35% Palm centroid distance (2D) + 20% Thumb curl
        # Switched centroid distances to 2D: fingertip-to-palm-center in XY space is a stable fist measure.
        avg_palm_dist = sum([
            self.get_distance_2d(landmarks[8], palm_center) / palm_size,
            self.get_distance_2d(landmarks[12], palm_center) / palm_size,
            self.get_distance_2d(landmarks[16], palm_center) / palm_size,
            self.get_distance_2d(landmarks[20], palm_center) / palm_size
        ]) / 4.0
        palm_dist_score = clamp((1.20 - avg_palm_dist) / (1.20 - 0.80))
        thumb_curl_score = 1.0 - thumb_ext_score
        fist_score = (0.45 * avg_finger_curl) + (0.35 * palm_dist_score) + (0.20 * thumb_curl_score)
        
        # E. Gojo Crossed Fingers Score 🤘 (Crossed Index/Middle, Ring/Pinky curled)
        # index and middle tips are extremely close (overlapping or crossed)
        # Using 2D distance to prevent z-noise from causing coordinate instability
        index_middle_tip_dist = self.get_distance_2d(landmarks[8], landmarks[12]) / palm_size
        crossed_closeness = clamp((0.30 - index_middle_tip_dist) / 0.20)  # 1.0 if dist <= 0.10, 0.0 if dist >= 0.30
        
        crossed_fingers_score = (0.30 * index_ext) + (0.30 * middle_ext) + (0.15 * ring_curl) + (0.15 * pinky_curl) + (0.10 * crossed_closeness)
        
        # Hard isolation gate: index and middle tips must be overlapping/crossed (dist < 0.25)
        # and index and middle must be extended (> 0.45), ring and pinky must be curled (> 0.45).
        # This completely separates Crossed Fingers from Peace Sign (wide V) and Fist (unextended)
        crossed_gate = clamp((0.25 - index_middle_tip_dist) / 0.15)
        crossed_ext_gate = clamp((index_ext + middle_ext) / 2.0 - 0.40) / 0.60
        crossed_curl_gate = clamp((ring_curl + pinky_curl) / 2.0 - 0.40) / 0.60
        
        crossed_fingers_score *= crossed_gate * crossed_ext_gate * crossed_curl_gate
        
        # ========================================================
        # PROBABILISTIC SMART SELECTION POOL (Smart Competition)
        # ========================================================
        scores = {
            "Rock On": rock_on_score,
            "Peace Sign": peace_sign_score,
            "Open Palm": open_palm_score,
            "Fist": fist_score,
            "Crossed Fingers": crossed_fingers_score
        }
        
        # Find the best gesture and its score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_gesture, best_score = sorted_scores[0]
        second_gesture, second_score = sorted_scores[1]
        
        # Calculate separation margin (Softmax-like margin)
        margin = best_score - second_score
        
        # Ambiguity threshold and Minimum confidence checks
        # min_score lowered 0.68 -> 0.65: 2D feature distances produce slightly lower absolute scores
        # but are more stable, so we can afford a lower threshold without false positives.
        # margin kept at 0.12 to maintain separation enforcement.
        if best_score >= 0.65 and margin >= 0.12:
            return best_gesture, best_score, scores
            
        return "None", 0.0, scores

    def smooth_landmarks(self, raw_landmarks, side="left"):
        """
        Applies temporal Exponential Moving Average (EMA) to a single hand's landmarks.
        Keeps independent history based on side ('left' or 'right') to prevent crosstalk.
        """
        prev_history = self.prev_smoothed_left if side == "left" else self.prev_smoothed_right
        
        # Calculate scale-invariant palm size
        palm_size = self.get_distance(raw_landmarks[0], raw_landmarks[9])
        palm_size = max(palm_size, 0.001)
        
        if prev_history is not None:
            # Compute average displacement normalized by palm size (scale-invariant speed)
            total_disp = 0.0
            for i in range(21):
                dx = raw_landmarks[i].x - prev_history[i].x
                dy = raw_landmarks[i].y - prev_history[i].y
                dz = raw_landmarks[i].z - prev_history[i].z
                total_disp += math.sqrt(dx*dx + dy*dy + dz*dz)
            avg_disp = total_disp / 21.0
            normalized_speed = avg_disp / palm_size
        else:
            normalized_speed = 0.1  # Fast snap on initial hand entrance
            
        # Determine adaptive standard alpha
        # min_alpha raised from 0.22 -> 0.30: reduces ghost-state lag during gesture transitions
        # while still providing sufficient smoothing at rest.
        # max_alpha raised from 0.55 -> 0.65: allows faster landmark response on intentional movement.
        min_alpha = 0.30
        max_alpha = 0.65
        min_speed = 0.015
        max_speed = 0.10
        
        if prev_history is None:
            base_alpha = 1.0
        else:
            if normalized_speed <= min_speed:
                base_alpha = min_alpha
            elif normalized_speed >= max_speed:
                base_alpha = max_alpha
            else:
                t = (normalized_speed - min_speed) / (max_speed - min_speed)
                base_alpha = min_alpha + t * (max_alpha - min_alpha)
                
        smoothed_list = []
        for i in range(21):
            raw_lm = raw_landmarks[i]
            
            # Fingertip landmarks (4,8,12,16,20) get a mild extra smoothing only when truly stationary.
            # The dampening factor is relaxed (0.88 instead of 0.80) to reduce ghost-state during transitions.
            if i in [0, 4, 8, 12, 16, 20] and prev_history is not None:
                speed_ratio = (base_alpha - min_alpha) / max(max_alpha - min_alpha, 0.001)
                lm_alpha = base_alpha * (0.88 + 0.12 * speed_ratio)
            else:
                lm_alpha = base_alpha
                
            if prev_history is None:
                smoothed_lm = SmoothedLandmark(raw_lm.x, raw_lm.y, raw_lm.z)
            else:
                prev_lm = prev_history[i]
                sm_x = lm_alpha * raw_lm.x + (1.0 - lm_alpha) * prev_lm.x
                sm_y = lm_alpha * raw_lm.y + (1.0 - lm_alpha) * prev_lm.y
                sm_z = lm_alpha * raw_lm.z + (1.0 - lm_alpha) * prev_lm.z
                smoothed_lm = SmoothedLandmark(sm_x, sm_y, sm_z)
                
            smoothed_list.append(smoothed_lm)
            
        if side == "left":
            self.prev_smoothed_left = smoothed_list
        else:
            self.prev_smoothed_right = smoothed_list
            
        return smoothed_list

    # ========================================================
    # ANIME SIGNATURE POSE DETECTORS
    # ========================================================

    def get_hand_peace_shape(self, landmarks, palm_size):
        """Helper to compute Index/Middle extension and Ring/Pinky curl scores for signature poses."""
        index_tip_knuckle = self.get_distance(landmarks[8], landmarks[5]) / palm_size
        middle_tip_knuckle = self.get_distance(landmarks[12], landmarks[9]) / palm_size
        ring_tip_knuckle = self.get_distance(landmarks[16], landmarks[13]) / palm_size
        pinky_tip_knuckle = self.get_distance(landmarks[20], landmarks[17]) / palm_size
        
        index_ext = max(0.0, min(1.0, (index_tip_knuckle - 0.35) / 0.40))
        middle_ext = max(0.0, min(1.0, (middle_tip_knuckle - 0.35) / 0.45))
        ring_curl = 1.0 - max(0.0, min(1.0, (ring_tip_knuckle - 0.32) / 0.43))
        pinky_curl = 1.0 - max(0.0, min(1.0, (pinky_tip_knuckle - 0.30) / 0.35))
        
        return (0.30 * index_ext) + (0.30 * middle_ext) + (0.20 * ring_curl) + (0.20 * pinky_curl)

    def detect_shadow_clone_jutsu(self, left_hand_lms, right_hand_lms):
        """
        Detects Naruto's iconic crossed-fingers Shadow Clone Jutsu pose.
        Requires index & middle extended, ring & pinky curled on both hands,
        and fingers crossed orthogonally in close proximity.
        """
        if left_hand_lms is None or right_hand_lms is None:
            return 0.0
            
        palm_L = self.get_distance(left_hand_lms[0], left_hand_lms[9])
        palm_L = max(palm_L, 0.001)
        palm_R = self.get_distance(right_hand_lms[0], right_hand_lms[9])
        palm_R = max(palm_R, 0.001)
        avg_palm = (palm_L + palm_R) / 2.0
        
        # 1. Individual hand shapes (Index & Middle open, Ring & Pinky curled)
        left_shape = self.get_hand_peace_shape(left_hand_lms, palm_L)
        right_shape = self.get_hand_peace_shape(right_hand_lms, palm_R)
        
        # 2. Perpendicular/Orthogonal index crossing check in 2D
        # Left index vector (MCP 5 to Tip 8)
        vl_x = left_hand_lms[8].x - left_hand_lms[5].x
        vl_y = left_hand_lms[8].y - left_hand_lms[5].y
        len_L = math.sqrt(vl_x*vl_x + vl_y*vl_y) or 0.001
        vl_x /= len_L
        vl_y /= len_L
        
        # Right index vector (MCP 5 to Tip 8)
        vr_x = right_hand_lms[8].x - right_hand_lms[5].x
        vr_y = right_hand_lms[8].y - right_hand_lms[5].y
        len_R = math.sqrt(vr_x*vr_x + vr_y*vr_y) or 0.001
        vr_x /= len_R
        vr_y /= len_R
        
        # Dot product: absolute value is 0 if perpendicular, 1 if parallel
        dot_val = abs(vl_x*vr_x + vl_y*vr_y)
        ortho_score = max(0.0, min(1.0, (0.70 - dot_val) / 0.50)) # Very comfortable threshold
        
        # 3. Fingertip intersection proximity check
        # Check midpoint of left index finger and midpoint of right index finger
        mid_l_x = (left_hand_lms[8].x + left_hand_lms[5].x) / 2.0
        mid_l_y = (left_hand_lms[8].y + left_hand_lms[5].y) / 2.0
        mid_l_z = (left_hand_lms[8].z + left_hand_lms[5].z) / 2.0
        
        mid_r_x = (right_hand_lms[8].x + right_hand_lms[5].x) / 2.0
        mid_r_y = (right_hand_lms[8].y + right_hand_lms[5].y) / 2.0
        mid_r_z = (right_hand_lms[8].z + right_hand_lms[5].z) / 2.0
        
        dist = math.sqrt((mid_l_x - mid_r_x)**2 + (mid_l_y - mid_r_y)**2 + (mid_l_z - mid_r_z)**2) / avg_palm
        proximity_score = max(0.0, min(1.0, (0.90 - dist) / 0.55)) # Highly comfortable distance
        
        # Combine metrics: 25% Left Hand + 25% Right Hand + 25% Orthogonal cross + 25% Proximity
        shadow_clone_score = (0.25 * left_shape) + (0.25 * right_shape) + (0.25 * ortho_score) + (0.25 * proximity_score)
        
        # Gating checks for robust separability
        if left_shape < 0.40 or right_shape < 0.40 or proximity_score < 0.10:
            shadow_clone_score *= 0.0
            
        return shadow_clone_score

    def detect_domain_expansion(self, landmarks):
        """
        Detects Gojo's crossed-fingers Domain Expansion (Infinite Void) single-hand pose.
        Requires index & middle extended, ring & pinky curled, and fingertips overlapping.
        """
        if landmarks is None:
            return 0.0
            
        palm_size = self.get_distance(landmarks[0], landmarks[9])
        palm_size = max(palm_size, 0.001)
        
        index_tip_knuckle = self.get_distance(landmarks[8], landmarks[5]) / palm_size
        middle_tip_knuckle = self.get_distance(landmarks[12], landmarks[9]) / palm_size
        ring_tip_knuckle = self.get_distance(landmarks[16], landmarks[13]) / palm_size
        pinky_tip_knuckle = self.get_distance(landmarks[20], landmarks[17]) / palm_size
        
        index_ext = max(0.0, min(1.0, (index_tip_knuckle - 0.35) / 0.40))
        middle_ext = max(0.0, min(1.0, (middle_tip_knuckle - 0.35) / 0.45))
        ring_curl = 1.0 - max(0.0, min(1.0, (ring_tip_knuckle - 0.32) / 0.43))
        pinky_curl = 1.0 - max(0.0, min(1.0, (pinky_tip_knuckle - 0.30) / 0.35))
        
        index_middle_tip_dist = self.get_distance_2d(landmarks[8], landmarks[12]) / palm_size
        crossed_closeness = max(0.0, min(1.0, (0.30 - index_middle_tip_dist) / 0.20))
        
        domain_expansion_score = (0.30 * index_ext) + (0.30 * middle_ext) + (0.15 * ring_curl) + (0.15 * pinky_curl) + (0.10 * crossed_closeness)
        
        crossed_gate = max(0.0, min(1.0, (0.25 - index_middle_tip_dist) / 0.15))
        crossed_ext_gate = max(0.0, min(1.0, ((index_ext + middle_ext) / 2.0 - 0.40) / 0.60))
        crossed_curl_gate = max(0.0, min(1.0, ((ring_curl + pinky_curl) / 2.0 - 0.40) / 0.60))
        
        domain_expansion_score *= crossed_gate * crossed_ext_gate * crossed_curl_gate
        return domain_expansion_score

    def detect_scout_salute(self, landmarks):
        """
        Detects Eren Yeager's iconic "Dedicate Your Heart" Scout salute (Fist over heart).
        A tight fist is centered vertically/horizontally, and the wrist is aligned
        horizontally across the chest.
        """
        if landmarks is None:
            return 0.0
            
        palm_size = self.get_distance(landmarks[0], landmarks[9])
        palm_size = max(palm_size, 0.001)
        
        # 1. Verify fist shape (index/middle/ring/pinky curled, palm centroid distance is small)
        index_tip_knuckle = self.get_distance(landmarks[8], landmarks[5]) / palm_size
        middle_tip_knuckle = self.get_distance(landmarks[12], landmarks[9]) / palm_size
        ring_tip_knuckle = self.get_distance(landmarks[16], landmarks[13]) / palm_size
        pinky_tip_knuckle = self.get_distance(landmarks[20], landmarks[17]) / palm_size
        
        index_curl = 1.0 - max(0.0, min(1.0, (index_tip_knuckle - 0.35) / 0.40))
        middle_curl = 1.0 - max(0.0, min(1.0, (middle_tip_knuckle - 0.35) / 0.45))
        ring_curl = 1.0 - max(0.0, min(1.0, (ring_tip_knuckle - 0.32) / 0.43))
        pinky_curl = 1.0 - max(0.0, min(1.0, (pinky_tip_knuckle - 0.30) / 0.35))
        avg_finger_curl = (index_curl + middle_curl + ring_curl + pinky_curl) / 4.0
        
        # Fist score
        fist_score = avg_finger_curl
        
        # 2. Horizontal wrist angle check
        # In a scout salute, the forearm lies horizontally across the chest.
        # This means the vector from the wrist (0) to middle knuckle (9) has a strong horizontal component (along x)
        # and wrist.y and knuckle[9].y are relatively close, making the vector perpendicular to vertical.
        # Let's compute normalized vector Wrist (0) to Middle Knuckle (9)
        vy_w_k = landmarks[9].y - landmarks[0].y
        vx_w_k = landmarks[9].x - landmarks[0].x
        len_w_k = math.sqrt(vx_w_k**2 + vy_w_k**2) or 0.001
        vx_norm = abs(vx_w_k / len_w_k)
        
        # A horizontal vector has vx_norm close to 1.0 (forearm across body).
        # We allow a highly tolerant, comfortable horizontal angle:
        alignment_score = max(0.0, min(1.0, (vx_norm - 0.30) / 0.50)) # 1.0 if vx_norm >= 0.80, 0.0 if vx_norm <= 0.30
        
        # Scout Salute score: 60% Fist + 40% Forearm horizontal alignment
        scout_salute_score = (0.60 * fist_score) + (0.40 * alignment_score)
        
        # Gating checks: must be a recognizable fist, and forearm must be horizontally aligned
        if fist_score < 0.55 or alignment_score < 0.15:
            scout_salute_score *= 0.0
            
        return scout_salute_score

    def detect_prayer_focus(self, left_hand_lms, right_hand_lms):
        """
        Detects Demon Slayer's Prayer / concentrated focus pose (palms pressed flat together).
        Both hands must be extended open palms, facing each other, and fingertips pointing upwards.
        """
        if left_hand_lms is None or right_hand_lms is None:
            return 0.0
            
        palm_L = self.get_distance(left_hand_lms[0], left_hand_lms[9])
        palm_L = max(palm_L, 0.001)
        palm_R = self.get_distance(right_hand_lms[0], right_hand_lms[9])
        palm_R = max(palm_R, 0.001)
        avg_palm = (palm_L + palm_R) / 2.0
        
        # 1. Verify both hands are open palms (all fingers extended)
        def get_hand_open_score(landmarks, palm_size):
            index_tip_knuckle = self.get_distance(landmarks[8], landmarks[5]) / palm_size
            middle_tip_knuckle = self.get_distance(landmarks[12], landmarks[9]) / palm_size
            ring_tip_knuckle = self.get_distance(landmarks[16], landmarks[13]) / palm_size
            pinky_tip_knuckle = self.get_distance(landmarks[20], landmarks[17]) / palm_size
            
            index_ext = max(0.0, min(1.0, (index_tip_knuckle - 0.35) / 0.40))
            middle_ext = max(0.0, min(1.0, (middle_tip_knuckle - 0.35) / 0.45))
            ring_ext = max(0.0, min(1.0, (ring_tip_knuckle - 0.32) / 0.43))
            pinky_ext = max(0.0, min(1.0, (pinky_tip_knuckle - 0.30) / 0.35))
            return (index_ext + middle_ext + ring_ext + pinky_ext) / 4.0
            
        left_open = get_hand_open_score(left_hand_lms, palm_L)
        right_open = get_hand_open_score(right_hand_lms, palm_R)
        
        # 2. Closeness check (palms pressed together)
        # Wrists must be close, and index fingertips must be close in 2D plane
        dist_wrists = self.get_distance_2d(left_hand_lms[0], right_hand_lms[0]) / avg_palm
        dist_tips = self.get_distance_2d(left_hand_lms[12], right_hand_lms[12]) / avg_palm
        
        # Closeness score: high if wrists and middle fingertips are in close proximity
        closeness_score = max(0.0, min(1.0, (1.0 - dist_wrists) / 0.60)) * max(0.0, min(1.0, (1.0 - dist_tips) / 0.60))
        
        # 3. Upward vertical pointing check
        # Fingertips (landmark 12) must be significantly higher than wrists (landmark 0) on both hands
        dy_left = left_hand_lms[0].y - left_hand_lms[12].y
        dy_right = right_hand_lms[0].y - right_hand_lms[12].y
        
        upward_score = max(0.0, min(1.0, (dy_left / (palm_L * 1.20) + dy_right / (palm_R * 1.20)) / 2.0))
        
        # Combine metrics: 30% Left Palm + 30% Right Palm + 20% Closeness + 20% Upward vertical pointing
        prayer_score = (0.30 * left_open) + (0.30 * right_open) + (0.20 * closeness_score) + (0.20 * upward_score)
        
        # Gating checks: both hands must be open, and palms must be reasonably close
        if left_open < 0.45 or right_open < 0.45 or closeness_score < 0.10 or upward_score < 0.10:
            prayer_score *= 0.0
            
        return prayer_score

    def process_frame(self, frame):
        """
        Processes an OpenCV BGR frame, applies precomputed combined scale & gamma LUT,
        extracts hand coordinates using lightweight model complexity, and applies fast C++ SIMD vignette.
        """
        # 1. Apply precomputed combined exposure scaling and Gamma correction in a single-pass LUT
        frame = cv2.LUT(frame, self.combined_lut)
        
        # 2. Apply fast radial dark vignette using C++ SIMD cv2.multiply to bypass float conversions
        h, w = frame.shape[:2]
        if self.vignette_mask_uint8 is None or self.vignette_mask_uint8.shape[:2] != (h, w):
            self._precompute_vignette_mask(h, w)
        frame = cv2.multiply(frame, self.vignette_mask_uint8, scale=1.0/255.0)

        # Convert BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture_name = "None"
        confidence = 0.0
        landmarks_list = None

        # Unified scores keyed to anime_controller gesture names
        scores = {
            "Shadow Clone Jutsu": 0.0,  # dual-hand cross (Naruto)
            "Peace Sign": 0.0,           # V fingers (One Piece / Luffy)
            "Scout Salute": 0.0,         # fist horizontal (Attack on Titan / Eren)
            "Prayer Focus": 0.0,         # dual-hand prayer (Demon Slayer / Tanjiro)
            "Domain Expansion": 0.0,     # crossed index+middle (JJK / Gojo)
        }

        # Internal single-hand classifier name → anime_controller key
        SINGLE_HAND_MAP = {
            "Rock On":         None,           # not mapped to any anime card
            "Peace Sign":      "Peace Sign",   # One Piece
            "Open Palm":       None,           # no anime card (absorbed by Prayer Focus)
            "Fist":            None,           # no anime card (absorbed by Scout Salute)
            "Crossed Fingers": "Domain Expansion",  # JJK / Gojo
        }

        left_hand_lms = None
        right_hand_lms = None

        # If hands are detected, extract and smooth landmarks
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                # Spatial sorting along X-axis (immunizes against MediaPipe L/R flips)
                hand_a = results.multi_hand_landmarks[0].landmark
                hand_b = results.multi_hand_landmarks[1].landmark
                if hand_a[0].x < hand_b[0].x:
                    left_hand_lms = self.smooth_landmarks(hand_a, "left")
                    right_hand_lms = self.smooth_landmarks(hand_b, "right")
                else:
                    left_hand_lms = self.smooth_landmarks(hand_b, "left")
                    right_hand_lms = self.smooth_landmarks(hand_a, "right")
                landmarks_list = [left_hand_lms, right_hand_lms]
            else:
                # Single hand detected
                left_hand_lms = self.smooth_landmarks(results.multi_hand_landmarks[0].landmark, "left")
                self.prev_smoothed_right = None  # Clear right hand temporal memory
                landmarks_list = left_hand_lms

            # ── Single-hand gestures via classify_gesture ──────────────────
            # classify_gesture uses left_hand_lms as the primary hand.
            single_gesture, single_score, raw_single_scores = self.classify_gesture(left_hand_lms)

            # Map raw classifier scores to anime keys where applicable
            for clf_name, anime_key in SINGLE_HAND_MAP.items():
                if anime_key is not None and clf_name in raw_single_scores:
                    scores[anime_key] = raw_single_scores[clf_name]

            # Scout Salute uses the dedicated detector (fist + horizontal arm)
            scores["Scout Salute"] = self.detect_scout_salute(left_hand_lms)

            # ── Dual-hand gestures ─────────────────────────────────────────
            if left_hand_lms is not None and right_hand_lms is not None:
                scores["Shadow Clone Jutsu"] = self.detect_shadow_clone_jutsu(left_hand_lms, right_hand_lms)
                scores["Prayer Focus"] = self.detect_prayer_focus(left_hand_lms, right_hand_lms)

            # ── Competitive selection across all anime gestures ────────────
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_gesture, best_score = sorted_scores[0]
            second_gesture, second_score = sorted_scores[1]
            margin = best_score - second_score

            if best_score >= 0.65 and margin >= 0.12:
                gesture_name = best_gesture
                confidence = best_score
            else:
                gesture_name = "None"
                confidence = 0.0
        else:
            # Hands disappeared: clear temporal memories
            self.prev_smoothed_left = None
            self.prev_smoothed_right = None
            
        fps = self.calculate_fps()
        
        # Return rgb_frame directly to completely avoid .rgbSwapped() conversions in the main UI thread!
        return rgb_frame, gesture_name, confidence, landmarks_list, fps, scores

    def release(self):
        """Releases the MediaPipe resources."""
        self.hands.close()
