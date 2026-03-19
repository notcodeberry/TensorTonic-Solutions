def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Write code here

    gray_image = []

    
    for row in image:
        row_list = []
        
        for pixel in row:
            gray_pixel = 0.299*pixel[0]+0.587*pixel[1]+0.114*pixel[2]

            row_list.append(gray_pixel)

        gray_image.append(row_list)

    return gray_image
    