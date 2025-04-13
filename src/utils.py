import os
from PIL import Image

def generate_gif(images_dir: str = "./data/analysis/visualizer"):
    """
    Generate a GIF from a list of images.

    Args:
        images_dir (str): Directory containing the images to be converted into a GIF.
    """
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith(".png")]
    )
    image_paths = [os.path.join(images_dir, f) for f in image_files]

    frames = [Image.open(path) for path in image_paths]

    output_path = os.path.join(images_dir, "animation.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # 0.5 seconds per frame (in ms)
        loop=0
    )

if __name__ == "__main__":
	# Example usage
	generate_gif()