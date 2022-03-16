import numpy as np
import cv2, os, time, itertools, threading
import timeit
import matplotlib.pyplot as plt

levels      = [ 1.0, 0.5, 1.5 ]
adjustments = [ adj for adj in itertools.product(levels, levels, levels) ]

def adjust_image(input_image):
    h, w, c = input_image.shape
    images  = np.zeros((3 ** 3, h,w,c), dtype = np.uint8)

    input_image = input_image.astype(np.single)
    for idx, (brightness, contrast, saturation) in enumerate(adjustments):
        img_proc = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
        np.multiply(img_proc, np.array([ 1.0, saturation, 1.0 ], dtype = np.single), out = img_proc)
        
        img_proc[img_proc > 255] = 255
        img_proc[img_proc < 0]   = 0

        cv2.cvtColor(img_proc, cv2.COLOR_HSV2RGB, dst = img_proc)
        np.multiply(img_proc, brightness, out = img_proc)
        np.add(img_proc, ((1-contrast) * 100))

        img_proc[img_proc > 255] = 255
        img_proc[img_proc < 0]   = 0

        images[idx] = img_proc
    return images

def process_images(image_list):
    for idx, img in enumerate(image_list):
        if(img == ''):
            continue

        beg = time.perf_counter()
        input_image = cv2.imread(img)
        input_image = cv2.resize(input_image, (128, 128))
        adjusted = adjust_image(input_image)
        end = time.perf_counter()

        for i, adj in enumerate(adjusted[1:]):
            name = 'Augmented/{:02}-{}'.format(i, os.path.basename(img))
            print(name)
            #input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(name, adj)
        #print('[{}] Finished in {:.2f} seconds'.format(img, end - beg))


# For testing purposes
IMG_DIR     = '../DatasetCleaning/Sortedface_with_mask'
IMAGE_FILES = [ os.path.join(IMG_DIR, x) for x in os.listdir(IMG_DIR) ]

def test_single_image():
    test_image = cv2.cvtColor(cv2.imread(IMAGE_FILES[0]), cv2.COLOR_BGR2RGB)
    print(test_image.shape)

    beg = time.perf_counter()
    adjusted_imgs = adjust_image(test_image)
    end = time.perf_counter()


    fig, ax = plt.subplots(9, 3, figsize=(12,24))
    for idx, img in enumerate(adjusted_imgs):
        plt_row  = idx // 3
        plt_col  = idx % 3
        
        title = 'S: {}, C: {}, B: {}'.format(adjustments[idx][0],
                                             adjustments[idx][1],
                                             adjustments[idx][2])
        ax[plt_row, plt_col].set_title(title)
        ax[plt_row, plt_col].imshow(img)
        ax[plt_row, plt_col].axis('off')

    plt.subplots_adjust(wspace=0, hspace=None)
    plt.tight_layout()
    plt.show()

def adjust_image2(input_image):
    h, w, c = input_image.shape
    images  = np.zeros((3 ** 3, h,w,c))

    for idx, (brightness, contrast, saturation) in enumerate(adjustments):
        input_hsv  = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV).astype(float)
        input_hsv *= np.array([ 1.0, saturation, 1.0 ])
        
        input_hsv[input_hsv > 255] = 255
        input_hsv[input_hsv < 0]   = 0

        input_rgb = cv2.cvtColor(input_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        input_rgb = input_rgb * brightness + ((1-contrast) * 100)

        input_rgb[input_rgb > 255] = 255
        input_rgb[input_rgb < 0]   = 0

        images[idx] = input_rgb
    return images.astype(np.uint8)

test_image = cv2.cvtColor(cv2.imread(IMAGE_FILES[0]), cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (128, 128))

def test():
    adj = adjust_image(test_image)

def test2():
    adj = adjust_image2(test_image)

#test_single_image()

t1 = timeit.Timer("test()",  globals = globals())
t2 = timeit.Timer("test2()", globals = globals())

print("Test 1: {:.2f}".format(t1.timeit(1000)))
print("Test 2: {:.2f}".format(t2.timeit(1000)))

#NUM_THREADS = 6
#padding     = (NUM_THREADS - (len(IMAGE_FILES) % NUM_THREADS)) % NUM_THREADS
#split_files = np.array(IMAGE_FILES + [''] * padding).reshape((NUM_THREADS, -1))
#
#beg = time.perf_counter()
#threads = []
#for i in split_files:
#    x = threading.Thread(target = process_images, args=(i,))
#    x.start()
#    threads.append(x)
#
#for t in threads:
#    t.join()
#end = time.perf_counter()
#print('total time: {:.2f}'.format(end-beg))
