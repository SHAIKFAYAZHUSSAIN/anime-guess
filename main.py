import sys
import os

# Import local modules
from gesture_detector import GestureDetector
from anime_controller import AnimeController
from ui import run_app

def main():
    """
    Main entry point for the Antigravity 2.0 Gesture Controlled Anime Interaction System.
    """
    print("==========================================================")
    print("  ANTIGRAVITY 2.0 - GESTURE CONTROLLED ANIME INTERACT SYSTEM  ")
    print("  [Phase 1 MVP - Local Tracking Mode Active]              ")
    print("==========================================================")
    
    # 1. Initialize detector with tuned confidence levels
    print("Initializing Hand Landmarks Classifier...")
    detector = GestureDetector(
        max_num_hands=2, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    )
    
    # 2. Initialize local anime controller
    print("Initializing Local State Manager...")
    anime_controller = AnimeController()
    
    # 3. Bootstrap the high-performance GUI
    print("Bootstrapping PySide6 GUI Thread...")
    run_app(detector, anime_controller)

if __name__ == "__main__":
    main()
