import numpy as np
import cv2, os, time
import matplotlib.pyplot as plt

levels      = [ 1.0, 0.5, 1.5 ]
adjustments = [ (b, c, s) for b in levels for c in levels for s in levels ]

def adjust_image(input_image):
    h, w, c = input_image.shape
    images  = np.zeros((3**3, h,w,c))

    for idx, adj in enumerate(adjustments):
        brightness, contrast, saturation = adj
        input_hsv  = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV).astype(float)
        input_hsv *= np.array([ 1.0, saturation, 1.0 ])

        input_hsv[input_hsv > 255] = 255
        input_hsv[input_hsv < 0]   = 0

        input_rgb = cv2.cvtColor(input_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        input_rgb = input_rgb * brightness + ((1-contrast) * 100)

        input_rgb[input_rgb > 255] = 255
        input_rgb[input_rgb < 0]   = 0

        images[idx] = input_rgb.astype(np.uint8)
    return images


# For testing purposes
IMG_DIR     = '../DatasetCleaning/Sortedface_with_mask'
IMAGE_FILES = [ os.path.join(IMG_DIR, x) for x in os.listdir(IMG_DIR) ]

test_image = cv2.cvtColor(cv2.imread(IMAGE_FILES[0]), cv2.COLOR_BGR2RGB)
print(test_image.shape)

beg = time.perf_counter()
adjusted = adjust_image(test_image)
end = time.perf_counter()
print('Finished in {:.2f} seconds'.format(end - beg))
print(adjusted.shape)

fig, ax = plt.subplots(9, 3, figsize=(12,24))
for idx, img in enumerate(adjusted):
    plt_row  = idx // 3
    plt_col  = idx % 3
    
    title = 'S: {}, C: {}, B: {}'.format(levels[plt_col],
                                         levels[(idx // 3) % 3],
                                         levels[(plt_row // 3) % 3])
    ax[plt_row, plt_col].set_title(title)
    ax[plt_row, plt_col].imshow(img.astype(np.uint8))
    ax[plt_row, plt_col].axis('off')

plt.subplots_adjust(wspace=0, hspace=None)
plt.tight_layout()
plt.show()
