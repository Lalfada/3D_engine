from PIL import Image
from math import floor

# thx chat gpt
def load_image(path):
    image = Image.open(path)
    image = image.convert("RGB")

    # Get the width and height of the image
    width, height = image.size
    # Access the pixel data as a list of RGB tuples
    pixel_data = list(image.getdata())
    pixel_matrix = [pixel_data[i * width:(i + 1) * width] for i in range(height)]
    image.close()
    return pixel_matrix, width, height

class Image_Data():
    def __init__(self, path):
        data, width, height = load_image(path)
        self.data = data
        self.width = width
        self.height = height
        # print(f"h: {self.height}; w: {self,width}")

    def pixel_from_uv(self, u, v):
        x = floor(self.width * u) if u != 1 else self.width - 1
        y = floor(self.height * v) if v != 1 else self.height - 1
        if not (0 <= x <= self.width - 1) or not (0 <= y <= self.height - 1):
            print(f"u: {u}; v: {v}; x: {x}; y: {y}")
        return self.data[y][x]