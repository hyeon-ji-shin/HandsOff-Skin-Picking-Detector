import cv2
import mediapipe as mp
import torch
import numpy as np
from collections import deque
import time
import argparse
from typing import Tuple, Dict


# =========================================================
# 1. Configuration (Hyperparameters)
# =========================================================
PINCH_DISTANCE_THRESHOLD = 0.04          # Normalized distance threshold (MediaPipe uses 0~1 coords)
LOCAL_JITTER_MIN = 0.0035                # Minimum local jitter (std) to qualify as scratching
LOCAL_JITTER_MAX = 0.15                  # Upper bound for valid finger jitter
GLOBAL_WRIST_JITTER_MAX = 0.05           # Maximum wrist movement (std) to consider stable
BUFFER_SIZE = 45                         # Sliding window size (~1.5 seconds)


# =========================================================
# 2. PyTorch-based Action Analyzer
# =========================================================
class ActionAnalyzer:
    """
    Analyze hand landmarks to detect repetitive skin-picking behavior.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

        # Per-hand historical buffers
        self.thumb_history: Dict[str, deque] = {}
        self.target_history: Dict[str, deque] = {}
        self.pinch_history: Dict[str, deque] = {}
        self.wrist_history: Dict[str, deque] = {}

    def _initialize_hand(self, hand_label: str) -> None:
        """Initialize buffers for a new hand label."""
        self.thumb_history[hand_label] = deque(maxlen=BUFFER_SIZE)
        self.target_history[hand_label] = deque(maxlen=BUFFER_SIZE)
        self.pinch_history[hand_label] = deque(maxlen=BUFFER_SIZE)
        self.wrist_history[hand_label] = deque(maxlen=BUFFER_SIZE)

    def _get_buffers(self, hand_label: str):
        if hand_label not in self.thumb_history:
            self._initialize_hand(hand_label)

        return (
            self.thumb_history[hand_label],
            self.target_history[hand_label],
            self.pinch_history[hand_label],
            self.wrist_history[hand_label],
        )

    def update_and_analyze(
        self,
        landmarks,
        hand_label: str
    ) -> Tuple[bool, float, float, torch.Tensor, torch.Tensor, float]:
        """
        Convert MediaPipe landmarks to torch tensors
        and analyze behavioral patterns.
        """

        thumb_q, target_q, pinch_q, wrist_q = self._get_buffers(hand_label)

        # Convert landmarks → torch tensor (21, 3)
        kpts = torch.tensor(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=torch.float32,
            device=self.device
        )

        # Key joints
        thumb = kpts[4, :2]
        index = kpts[8, :2]
        middle = kpts[12, :2]
        wrist = kpts[0, :2]

        # Compute distances
        dist_index = torch.linalg.norm(thumb - index)
        dist_middle = torch.linalg.norm(thumb - middle)

        if dist_index < dist_middle:
            target_finger = index
            min_dist = dist_index
        else:
            target_finger = middle
            min_dist = dist_middle

        # Update buffers
        thumb_q.append(thumb)
        target_q.append(target_finger)
        wrist_q.append(wrist)

        # (A) Pinch Detection
        is_pinching = min_dist < PINCH_DISTANCE_THRESHOLD
        pinch_q.append(is_pinching)

        # Default outputs
        is_picking = False
        local_jitter = 0.0
        wrist_jitter = 0.0

        if len(thumb_q) == BUFFER_SIZE:
            seq_thumb = torch.stack(list(thumb_q))
            seq_target = torch.stack(list(target_q))
            seq_wrist = torch.stack(list(wrist_q))

            # (B) Sustained pinch ratio
            pinch_ratio = sum(pinch_q) / BUFFER_SIZE

            # (C) Local finger jitter
            relative_dist = torch.linalg.norm(seq_thumb - seq_target, dim=1)
            local_jitter = torch.std(relative_dist).item()

            # (D) Wrist global stability
            wrist_std = torch.std(seq_wrist, dim=0)
            wrist_jitter = torch.norm(wrist_std).item()

            # Final decision rule
            is_picking = (
                pinch_ratio > 0.8 and
                LOCAL_JITTER_MIN < local_jitter < LOCAL_JITTER_MAX and
                wrist_jitter < GLOBAL_WRIST_JITTER_MAX
            )

        return (
            is_picking,
            min_dist.item(),
            local_jitter,
            thumb,
            target_finger,
            wrist_jitter,
        )


# =========================================================
# 3. Main Execution
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="HandsOff Skin Picking Detector"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera device index (default: 1)"
    )
    args = parser.parse_args()

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    analyzer = ActionAnalyzer(device=device)

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    print(f"Opening camera index: {args.camera}")
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Failed to open camera index {args.camera}")
        return

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        start_times = {}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            h, w, _ = frame.shape

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                    handedness = results.multi_handedness[idx]
                    hand_label = handedness.classification[0].label
                    hand_label = "Right" if hand_label == "Left" else "Left"

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    (
                        is_picking,
                        dist,
                        jitter,
                        thumb_pt,
                        target_pt,
                        wrist_jitter
                    ) = analyzer.update_and_analyze(
                        hand_landmarks,
                        hand_label
                    )

                    t_x, t_y = int(thumb_pt[0] * w), int(thumb_pt[1] * h)
                    tg_x, tg_y = int(target_pt[0] * w), int(target_pt[1] * h)

                    text_y = 30 + (idx * 30)
                    info_text = (
                        f"[{hand_label}] "
                        f"Dist:{dist:.3f} | "
                        f"Jitter:{jitter:.4f} | "
                        f"Wrist:{wrist_jitter:.4f}"
                    )

                    color = (0, 0, 255) if is_picking else (0, 255, 0)

                    if is_picking:
                        if not start_times.get(hand_label):
                            start_times[hand_label] = time.time()

                        cv2.rectangle(frame, (0, 0), (w, h), color, 10)
                        cv2.putText(
                            frame,
                            "STOP PICKING!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            color,
                            4,
                        )
                        cv2.line(frame, (t_x, t_y), (tg_x, tg_y), color, 5)
                    else:
                        start_times[hand_label] = None
                        cv2.line(frame, (t_x, t_y), (tg_x, tg_y), color, 2)

                    cv2.putText(
                        frame,
                        info_text,
                        (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            cv2.imshow("HandsOff", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

