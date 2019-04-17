import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import color
import scipy


#matplotlib.style.use('ggplot')



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def plot_img(img_rgb):
    #ratio = "1680x1050"
    #ratio = "16x10"
    #ratio = "16x5"
    #ratio = "8x5"
    #ratio = "16x10"
    ratio = "32x20"
    #ratio = "64x40"

    #90x50

    scipy.misc.imsave('./head_cam_plots/cam_image_rgb_data_{}.bmp'.format(ratio), img_rgb)
    #np.save("cam_image_rgb_data_{}.out".format(ratio), np.array(img_rgb), fmt='%d')
    # rgb
    plt.imshow(img_rgb)
    plt.savefig('./head_cam_plots/cam_rgb{}.pdf'.format(ratio), bbox_inches='tight')
    plt.clf()

    # r
    img_r = img_rgb[:, :, 0]
    plt.imshow(img_r, cmap='gray')
    plt.savefig('./head_cam_plots/cam_r{}.pdf'.format(ratio), bbox_inches='tight')
    plt.clf()

    # hue, saturation, lightness
    img_hsv = color.rgb2hsv(img_rgb)
    img_s = img_hsv[:, :, 1]
    plt.imshow(img_s, cmap='gray')
    plt.savefig('./head_cam_plots/cam_s{}.pdf'.format(ratio), bbox_inches='tight')
    plt.clf()

    # gray
    img_gray = color.rgb2gray(img_rgb)
    plt.imshow(img_gray, cmap='gray')
    plt.savefig('./head_cam_plots/cam_gray{}.pdf'.format(ratio), bbox_inches='tight')
    plt.clf()

    img_gray = color.rgb2gray(img_rgb)
    img_gray = img_gray[9]
    img_gray = img_gray.reshape(1, 32, 1)
    img_gray = img_gray[:, :, 0]
    #img_gray = np.round(img_gray,2)
    plt.imshow(img_gray, cmap='gray')
    plt.yticks([])

    #cbar= plt.colorbar(orientation='horizontal', ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar = plt.colorbar(orientation='horizontal')
    #cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])

    plt.savefig('./head_cam_plots/cam_gray_row{}.pdf'.format(ratio), bbox_inches='tight')

    plt.clf()


if __name__ == '__main__':
    main()
