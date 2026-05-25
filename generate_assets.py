import os
import requests
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont

def download_file(url, target_path):
    """Downloads a file via HTTP, returning True if successful."""
    try:
        print(f"Downloading: {url} ...")
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Server returned status {response.status_code} for {url}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def make_ambient_blur(source_path, target_path, radius=80, brightness=0.25):
    """
    Creates a highly blurred and darkened version of the image
    to serve as a hardware-friendly ambient cinematic backdrop.
    """
    try:
        with Image.open(source_path) as img:
            # Blur the image strongly
            blurred = img.filter(ImageFilter.GaussianBlur(radius))
            
            # Darken the blurred image
            enhancer = ImageEnhance.Brightness(blurred)
            darkened = enhancer.enhance(brightness)
            
            # Save as PNG/JPG
            darkened.save(target_path)
            print(f"Created ambient blur: {target_path}")
            return True
    except Exception as e:
        print(f"Error creating ambient blur: {e}")
        return False

def normalize_and_sharpen_poster(img_path):
    """
    Upscales the image to a premium 600x800 resolution using Lanczos,
    applies unsharp mask filtering to restore details, and optimizes contrast
    and color vibrancy to make it look like premium streaming service art.
    """
    try:
        with Image.open(img_path) as img:
            # Determine correct high-quality resampling filter
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.ANTIALIAS
                
            # 1. Upscale to 600x800
            resized = img.resize((600, 800), resample_filter)
            
            # 2. Apply a professional Unsharp Mask to restore micro-textures and sharp lines
            # radius=1.5, percent=120, threshold=2
            sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
            
            # 3. Enhance local contrast and saturation for premium color fidelity
            enhancer_contrast = ImageEnhance.Contrast(sharpened)
            vibrant = enhancer_contrast.enhance(1.06)
            
            enhancer_color = ImageEnhance.Color(vibrant)
            normalized = enhancer_color.enhance(1.05)
            
            # Save the normalized image back
            normalized.save(img_path, "JPEG", quality=95)
            print(f"Normalized and sharpened poster: {img_path}")
            return True
    except Exception as e:
        print(f"Error normalizing poster {img_path}: {e}")
        return False

def create_offline_fallback(title, character, phrase, primary_color, secondary_color, img_path, blur_path):
    """Creates elegant minimalist posters locally if the system is offline during setup."""
    img = Image.new('RGB', (600, 800), color=(10, 11, 16))
    draw = ImageDraw.Draw(img)
    
    # Soft gradients
    for y in range(800):
        factor = y / 800.0
        r = int(10 + (primary_color[0] - 10) * factor * 0.15)
        g = int(11 + (primary_color[1] - 11) * factor * 0.15)
        b = int(16 + (primary_color[2] - 16) * factor * 0.15)
        draw.line([(0, y), (600, y)], fill=(r, g, b))
        
    # Draw simple elegant borders
    draw.rectangle([20, 20, 580, 780], outline=primary_color, width=2)
    
    # Text
    draw.text((300, 200), title, fill=primary_color, anchor="mm")
    draw.text((300, 400), character, fill=(255, 255, 255), anchor="mm")
    draw.text((300, 550), phrase, fill=secondary_color, anchor="mm")
    
    img.save(img_path)
    make_ambient_blur(img_path, blur_path)
    print(f"Offline fallback generated: {img_path}")

def main():
    # Setup directories
    assets_dir = os.path.join("assets", "anime_images")
    fonts_dir = os.path.join("assets", "fonts")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(fonts_dir, exist_ok=True)
    
    # 1. Download Google Font: Inter (Sequential URL fallbacks for maximum resilience)
    font_urls = {
        "Inter-Regular.ttf": [
            "https://raw.githubusercontent.com/shadcn-ui/ui/main/apps/www/public/fonts/Inter-Regular.ttf",
            "https://raw.githubusercontent.com/jakejarvis/inter-font/master/Inter-Regular.ttf",
            "https://raw.githubusercontent.com/rsms/inter/master/docs/font-files/Inter-Regular.ttf"
        ],
        "Inter-Bold.ttf": [
            "https://raw.githubusercontent.com/shadcn-ui/ui/main/apps/www/public/fonts/Inter-Bold.ttf",
            "https://raw.githubusercontent.com/jakejarvis/inter-font/master/Inter-Bold.ttf",
            "https://raw.githubusercontent.com/rsms/inter/master/docs/font-files/Inter-Bold.ttf"
        ]
    }
    
    for name, urls in font_urls.items():
        font_path = os.path.join(fonts_dir, name)
        if not os.path.exists(font_path):
            for url in urls:
                if download_file(url, font_path):
                    print(f"Acquired font {name} from {url}")
                    break
            
    # 2. Key Art details and download URLs
    anime_visuals = {
        "rock_on": {
            "title": "NARUTO",
            "character": "Naruto Uzumaki",
            "phrase": "I'm not gonna run away, I never go back on my word!",
            "primary": (255, 120, 0),
            "secondary": (255, 210, 0),
            "url": "https://cdn.myanimelist.net/images/anime/13/17405l.jpg"
        },
        "peace_sign": {
            "title": "ONE PIECE",
            "character": "Monkey D. Luffy",
            "phrase": "If you don't take risks, you can't create a future!",
            "primary": (255, 210, 0),
            "secondary": (255, 0, 50),
            "url": "https://cdn.myanimelist.net/images/anime/1244/138851l.jpg"
        },
        "fist": {
            "title": "ATTACK ON TITAN",
            "character": "Eren Yeager",
            "phrase": "If you win, you live. If you lose, you die. If you don't fight, you can't win!",
            "primary": (255, 0, 50),
            "secondary": (150, 150, 150),
            "url": "https://cdn.myanimelist.net/images/anime/10/47347l.jpg"
        },
        "open_palm": {
            "title": "DEMON SLAYER",
            "character": "Tanjiro Kamado",
            "phrase": "No matter how many people you lose, you have no choice but to go on living.",
            "primary": (0, 240, 255),
            "secondary": (255, 50, 150),
            "url": "https://cdn.myanimelist.net/images/anime/1286/99889l.jpg"
        },
        "gojo": {
            "title": "JUJUTSU KAISEN",
            "character": "Gojo Satoru",
            "phrase": "Don't worry, I'm the strongest.",
            "primary": (160, 32, 240), # Neon Purple
            "secondary": (255, 255, 255),
            "url": "https://cdn.myanimelist.net/images/anime/1171/109222l.jpg"
        }
    }
    
    for key, info in anime_visuals.items():
        img_path = os.path.join(assets_dir, f"{key}.jpg")
        blur_path = os.path.join(assets_dir, f"{key}_blurred.jpg")
        
        # Download official high-res poster
        success = download_file(info["url"], img_path)
        if success:
            normalize_and_sharpen_poster(img_path)
            make_ambient_blur(img_path, blur_path)
        else:
            # Fall back to high-quality procedural local render if offline
            print(f"Offline detected or network failed for {info['title']}. Initiating elegant local fallback...")
            create_offline_fallback(
                info["title"], 
                info["character"], 
                info["phrase"], 
                info["primary"], 
                info["secondary"], 
                img_path, 
                blur_path
            )
            # Make sure offline fallbacks are also polished
            normalize_and_sharpen_poster(img_path)

    print("\nStage 2 Asset Generation Complete!")

if __name__ == "__main__":
    main()
