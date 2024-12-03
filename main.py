from src.cropper import crop_characters

if __name__ == "__main__":
    input_image = "./input/BP-50M55_20241128_165039-02.jpg"
    output_directory = "./output/image"
    crop_characters(input_image, output_directory)