import numpy as np
import cv2

# Note : Method hayi ke in zir gozashtam baraye image haye sangin runtime toolani daran 
# Code zir baraye resize kardan image hast ta in runtime ro kam konid dar soorat niaz

def resize_image(input_image_path, output_image_path, target_size=(1920, 1080)):
   
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Save the resized image
    cv2.imwrite(output_image_path, resized_image)
    print(f"Resized image saved as {output_image_path}")

# Use
input_image_path = 'Image_Path,jpg'  #Masir image asli ke mikhaid resizesh konid
output_image_path = 'Output_image_path.jpg'
resize_image(input_image_path, output_image_path)



### Replacement Method

def reduce_salt_and_pepper_noise(image_path, neighborhood_size=3):
    
    # image ro load mikonim
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # copy migirim az image
    denoised_image = image.copy()
   
    # rooye har pixel tekrar mikonim
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # mahdoode neighbor ha ro mohasebe mikonim
            row_start = max(0, i - neighborhood_size // 2)
            row_end = min(image.shape[0], i + neighborhood_size // 2 + 1)
            col_start = max(0, j - neighborhood_size // 2)
            col_end = min(image.shape[1], j + neighborhood_size // 2 + 1)
            
            # neighbor window ro peyda mikonim
            neighborhood = image[row_start:row_end, col_start:col_end]
            
            # miangin neighborhood ro hesab mikonim
            average_value = np.mean(neighborhood)
            
            # pixel noisy ro ba average neighborhood replace mikonim
            denoised_image[i, j] = average_value
    
    return denoised_image

input_image_path = 'D:\\Pictures\\Noisy\\inputimage.jpg'
denoised_image = reduce_salt_and_pepper_noise(input_image_path)

# Denoised image ro save mikonim
output_path = 'D:\\Pictures\\Noisy\\replacementttt.jpg'
cv2.imwrite(output_path, denoised_image)
print(f"Denoised image saved as {output_path}")









###Median method
def manual_median_filter(image, neighborhood_size=3):

    # az image copy migirim
    denoised_image = image.copy()
    
    # rooye har pixel tekrar mikonim
    for i in range(neighborhood_size // 2, image.shape[0] - neighborhood_size // 2):
        for j in range(neighborhood_size // 2, image.shape[1] - neighborhood_size // 2):
            #window neighbor ha ro peyda mikonim
            neighborhood = image[i - neighborhood_size // 2 : i + neighborhood_size // 2 + 1,
                                 j - neighborhood_size // 2 : j + neighborhood_size // 2 + 1]
            
            # Median Neighbor ha ro mohasebe mikonim
            median_value = np.median(neighborhood)
            
            # Pixel noisy ro ba median replace mikonim
            denoised_image[i, j] = median_value
    
    return denoised_image


#image Path
input_image_path = 'D:\\Pictures\\Noisy\\inputimage.jpg'
noisy_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

denoised_image = manual_median_filter(noisy_image)

#SAVE
output_path = 'D:\\Pictures\\Noisy\\denoisemedian.jpg'
cv2.imwrite(output_path, denoised_image)
print(f"Denoised image saved as {output_path}")



