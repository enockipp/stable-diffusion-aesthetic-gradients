#!/usr/bin/env python3
#coding:utf-8

import matplotlib.pyplot as plt

def img_out_grid(images=None, titles=None, out_file=None, col_num=1):

    assert images != None and len(images) > 1, (
        "Invalid list of images")
    assert titles != None and len(titles) > 1, (
        "Invalid list of titles")
    assert len(images) == len(titles), ( 
        "The num of image does not equal to the number of titles")

    img_num = len(images)
    row_num = int((img_num - 1) / col_num) + 1

    plt.rcParams['font.size'] = '11'
    plt.figure(figsize=(27 * row_num, 12 * row_num),  layout="tight")
    plot_idx = 1 #the index start from 1
    for idx, img in enumerate(images):
        plt.subplot(row_num, col_num, plot_idx, title= titles[idx], xticks=[], yticks=[])
        plt.imshow(img)

        plot_idx = plot_idx + 1

    if out_file is not None:
        plt.rcParams['savefig.facecolor']='white'    
        plt.savefig(out_file, bbox_inches='tight')
