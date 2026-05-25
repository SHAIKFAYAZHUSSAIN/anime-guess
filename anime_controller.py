import os

class AnimeController:
    """
    Manages local anime metadata and gesture asset mapping.
    Stage 2: Handles blurred ambient backdrops and premium theme colors.
    """
    def __init__(self, assets_dir=None):
        if assets_dir is None:
            self.assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "anime_images")
        else:
            self.assets_dir = assets_dir
            
        # Cinematic Phase 2 anime metadata and visual color configurations
        self.gesture_mapping = {
            "Shadow Clone Jutsu": {
                "title": "NARUTO",
                "character": "Naruto Uzumaki",
                "quote": "I'm not gonna run away, I never go back on my word!",
                "theme": "Naruto Uzumaki is a ninja who strives to become Hokage, the leader of his village, despite being shunned by everyone.",
                "image_filename": "rock_on.jpg",      # generate_assets.py saves as rock_on.jpg
                "accent_color": "#FF7800"  # Warm Orange
            },
            "Peace Sign": {
                "title": "ONE PIECE",
                "character": "Monkey D. Luffy",
                "quote": "If you don't take risks, you can't create a future!",
                "theme": "Monkey D. Luffy seeks to become the Pirate King by finding the legendary treasure, One Piece, sailing with his crew.",
                "image_filename": "peace_sign.jpg",
                "accent_color": "#FFEF00"  # Sunlight Yellow
            },
            "Scout Salute": {
                "title": "ATTACK ON TITAN",
                "character": "Eren Yeager",
                "quote": "If you win, you live. If you lose, you die. If you don't fight, you can't win!",
                "theme": "Eren Yeager vows to cleanse the earth of all giant humanoid Titans after they destroy his hometown and eat his mother.",
                "image_filename": "fist.jpg",         # generate_assets.py saves as fist.jpg
                "accent_color": "#FF0032"  # Deep Crimson Red
            },
            "Prayer Focus": {
                "title": "DEMON SLAYER",
                "character": "Tanjiro Kamado",
                "quote": "No matter how many people you lose, you have no choice but to go on living.",
                "theme": "Tanjiro Kamado sets out to become a demon slayer to avenge his family and turn his demon sister, Nezuko, back into a human.",
                "image_filename": "open_palm.jpg",    # generate_assets.py saves as open_palm.jpg
                "accent_color": "#00F0FF"  # Soft Aurora Cyan
            },
            "Domain Expansion": {
                "title": "JUJUTSU KAISEN",
                "character": "Gojo Satoru",
                "quote": "Don't worry, I'm the strongest.",
                "theme": "Gojo Satoru is a special-grade jujutsu sorcerer and widely recognized as the strongest in the world.",
                "image_filename": "gojo.jpg",
                "accent_color": "#A020F0"  # Neon Purple
            }
        }

    def get_anime_data(self, gesture_name):
        """
        Returns full anime metadata for a recognized gesture.
        If the gesture is unrecognized or has no mapping, returns None.
        """
        if gesture_name not in self.gesture_mapping:
            return None
            
        data = self.gesture_mapping[gesture_name].copy()
        
        # Verify and build absolute cover poster and blurred ambient path
        image_path = os.path.join(self.assets_dir, data["image_filename"])
        basename, ext = os.path.splitext(data["image_filename"])
        blur_filename = f"{basename}_blurred{ext}"
        blur_path = os.path.join(self.assets_dir, blur_filename)
        
        # Verify file existence on disk
        if os.path.exists(image_path):
            data["image_path"] = image_path
        else:
            data["image_path"] = ""
            
        if os.path.exists(blur_path):
            data["ambient_path"] = blur_path
        else:
            data["ambient_path"] = ""
            
        return data

    def get_all_gestures(self):
        """Returns the list of all supported gestures in Phase 1 & 2."""
        return list(self.gesture_mapping.keys())
