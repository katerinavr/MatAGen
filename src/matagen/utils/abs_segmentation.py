import mmdet
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import torch
import re
import math
import random
from sklearn import cluster
import numpy as np
import torch, torchvision
import mmdet
import glob
import os
from PIL import Image, ImageDraw
from distinctipy import distinctipy
from scipy.interpolate import interp1d
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.kde import KDE
from sklearn.cluster import AffinityPropagation
from scipy.interpolate import splrep, sproot, splev
from scipy.cluster import vq
import PIL
from .plot_data_extraction.plot_digitizer import PlotDigitizer
from .plot_data_extraction.SpatialEmbeddings.src.utils import transforms as my_transforms
from .plot_data_extraction.evaluation import PlotEvaluator
from .plot_data_extraction.utils import Segmap2Lines, GenerateTestData, dict2class
from .axis_alignment.utils import dict2class, AxisAlignment
from .plot_data_extraction.optical_flow import OpticalFlow
from sklearn import preprocessing
import shutil
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment

axis_align_opt = {
    # region detection
    "config_file": "pneumatic/tools/Plot2Spec_materials_eyes/checkpoints/axis_alignment/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py",
    "checkpoint_file": "pneumatic/tools/Plot2Spec_materials_eyes/checkpoints/axis_alignment/epoch_200.pth",
    "refinement": True,
    # tick detection
    "cuda": False,
    "canvas_size": 1280,
    "mag_ratio": 1.5,
    "poly": False,
    "text_threshold": 0.1,
    "low_text": 0.5,
    "link_threshold": 0.7,
    "show_time": False,
    "refine": True,
    "trained_model": 'pneumatic/tools/Plot2Spec_materials_eyes/checkpoints/axis_alignment/craft_mlt_25k.pth',
    "refiner_model": 'pneumatic/tools/Plot2Spec_materials_eyes/checkpoints/axis_alignment/craft_refiner_CTW1500.pth',
    # tick recognition
    "workers": 0,
    "saved_model": "pneumatic/tools/Plot2Spec_materials_eyes/checkpoints/axis_alignment/TPS-ResNet-BiLSTM-Attn.pth",
    "batch_max_length": 25,
    "imgH": 32,
    "imgW": 100,
    "rgb": False,
    "character": "0123456789abcdefghijklmnopqrstuvwxyz",
    "sensitive": False,
    "PAD": True,
    "Transformation": "TPS",
    "FeatureExtraction": "ResNet",
    "SequenceModeling": "BiLSTM",
    "Prediction": "Attn",
    "num_fiducial": 20,
    "input_channel": 1,
    "output_channel": 512,
    "hidden_size": 256,
}

background_opt = {
    "cuda": False,
    "display": False,
    "save": True,
    "save_dir": "./exp/",
    "root": "pneumatic/tools/Plot2Spec_materials_eyes/data/input_plot_extraction/leftImg8bit/",
    "data_type": "test",
    "mode": "abs_spectra",
    "num_workers": 0,
    "model_file": "pneumatic/tools/Plot2Spec_materials_eyes/checkpoints/plot_data_extraction/lanenet/deeplab_all_checkpoint/0999_checkpoint.ckpt",
    "checkpoint_path": "pneumatic/tools/Plot2Spec_materials_eyes/checkpoints/plot_data_extraction/checkpoint_0999.pth",
    "dataset": {
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': "pneumatic/tools/Plot2Spec_materials_eyes/data/input_plot_extraction",
            'type': 'test',
            'transform': my_transforms.get_transform([
                {
                    "name": "CustomResizePad",
                    "opts": {
                        'keys': ('image', 'instance', 'label'),
                        "is_test": True,
                    },
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
                {
                    'name': 'Normalize',
                    'opts': {
                        'keys': ('image'),
                        'p': -1,
                    }
                },
            ]),
        }
    },
    "model": {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [3, 1],
        }
    }
}

# helper functions

def recognize_text(img_path):
    '''loads an image and recognizes text.'''

    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)

def overlay_ocr_text(img_path):

    '''loads an image, recognizes text, and overlays the text on the image.'''
    points = []
    labels=[]
    # loads image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#BGR2HSV)#)


    # recognize text
    result = recognize_text(img_path)

    # if OCR prob is over 0.5, overlay bounding box and text
    for (bbox, text, prob) in result:
      if len(text)>1:
        if text[0].isalpha() == True:
          if text[:10] != 'Wavelength':
            if text[:2] != 'nm':
              if '00' not in text:
                if prob >= 0.4:
                  labels.append(text)#.split()[0])
                  # get top-left and bottom-right bbox vertices
                  (top_left, top_right, bottom_right, bottom_left) = bbox
                  top_left = (int(top_left[0]), int(top_left[1]))
                  bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                  points.append(bbox)
                  # create a rectangle for bbox display
                  cv2.rectangle(img=img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)

                  # put recognized text
                  cv2.putText(img=img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)
      #print(labels)
    return labels, points

def get_label_colors(image_path, i, labels, points):
    lab = labels[i]
    (top_left, _, bottom_right, _) = points[i]

    # Open the image using PIL
    with Image.open(image_path) as im:
        # Crop the image correctly
        cropped_image = im.crop((top_left[0] - 100, (bottom_right[1] + top_left[1]) / 2 - 2, top_left[0], (bottom_right[1] + top_left[1]) / 2 + 2))

    # Convert the cropped image to RGB and save it
    cropped_image_rgb = cropped_image.convert("RGB")
    cropped_image_rgb.save('cropped_image.png')

    # Read the saved image using OpenCV
    src = cv2.imread('cropped_image.png')
    image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    # Process pixels
    pxls = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = tuple(image[i, j])
            if pixel < (250, 250, 250):
                pxls.append(pixel)

    # Compute the mean of the pixels, if any pixels were added
    if pxls:
        return lab, np.mean(pxls, axis=0)
    else:
        return lab, None

def create_dictionary(image_path):

  labels, points= overlay_ocr_text(image_path)
  labs=[]
  rgb=[]
  for i in range(len(labels)):
    labelaki, color = get_label_colors(image_path, i, labels, points)
    labs.append(labelaki)
    rgb.append(color)
  res = dict(map(lambda i,j : (i,j) , labs, rgb))
  return res

def fig_text_main(path):
  #for filename in os.listdir(folder_path): #make to look only on image files
    #print(im_id)
    #print(filename)
    dict_1=create_dictionary(path)
    dt_keys= dict_1.keys()
    try:
      for i in dt_keys:
        if len(set(dict_1[i])) == 1:
          dict_1.pop(i, None)
    except:
      pass
    print(dict_1)
    #print(res)
    return dict_1

def read_img(path):
  img = cv2.imread(path)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.

def write_img(img, path):
  img = (img * 255).astype(np.uint8)
  print(img.shape, img.dtype)
  Image.fromarray(img).save(path)

def dilate_image(image_path):
  img = read_img(image_path)
  img = np.abs(img - 1)
  print(img.mean())
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  img = cv2.dilate(img, kernel, iterations=1)
  img = np.abs(img - 1)
  write_img(img, image_path)
  return img


def derivative(x_data, y_data):
    N = len(x_data)
    delta_x = [x_data[i+1] - x_data[i] for i in range(N - 1)]
    x_prim = [(x_data[i+1] + x_data[i]) / 2. for i in range(N - 1)]
    y_prim = [(y_data[i+1] - y_data[i]) / delta_x[i] for i in range(N - 1)]
    return x_prim, y_prim



def check_img(i, paletted):
  y, x = np.where(np.asarray(paletted) == i)
  df=pd.concat([pd.DataFrame(x),pd.DataFrame(y)], axis=1)
  if df.shape[0] <  2000: 
    #print(df.shape[0])

    Y = df.values# y.reshape(1, 1)#.values#.reshape(-1, 1)
    clf = KNN(n_neighbors=10, radius=0.3, contamination=0.2) #AffinityPropagation(damping=0.8, max_iter=200, convergence_iter=25)#KDE(contamination=0.02, algorithm='auto')#random_state=0, contamination=0.02)
    clf.fit(Y)
    outliers = clf.predict(Y)
    out = np.where(outliers==1)
    into = np.where(outliers==0)
    if np.mean(clf.decision_scores_)< 7:
      if len(out[0])>0:

        x1 =  x[into]+300  # change to standard np.arrange()np.arange(350, 700, 10) #
        y1 = y[into]
        x_bis, y_bis = derivative(*derivative(x1, y1))
        print('curvature',  np.mean(x_bis) )
        if np.mean(x_bis)>50:
          #print(x1.min())
          #plt.scatter(x1 , y1)
          f = interp1d(x1, y1, fill_value="extrapolate")

          #wavelength = np.arange(450, 650,1)
          #try:
          #  plt.scatter(wavelength , f(wavelength))
          #except:
          wavelength = np.arange(x1.min(), x1.max(),1)  #
          # plt.gca().invert_yaxis()

          # plt.scatter(wavelength , f(wavelength), label='outlier')
          #print(np.count_nonzero(np.isnan(y)))

          #y = y[np.logical_not(np.isnan(y))]
          #print(np.var(y))
          # plt.show()
          return wavelength, f(wavelength) # np.array([wavelength, y])or # np.array([wavelength, y])

def get_abs_dictionary(path, abs_image):
    
    dict_1 = fig_text_main(abs_image) # identifies the names in the images

    axis_alignment = AxisAlignment(axis_align_opt)
    axis_alignment.load_data(path)
    img, plot_bbox, results, results_all = axis_alignment.run(0)
    x_bias, y_bias = list(plot_bbox)[:2]
    x_min, y_min, x_max, y_max = int(list(plot_bbox)[0]), list(plot_bbox)[1], int(list(plot_bbox)[2]), list(plot_bbox)[3]
    x_count = x_max-x_min+1
    norm_ts = np.linspace(0,x_max-x_min, x_count)

    # generate the result for the test image
    
    # print('ticks', [tick[0] for tick in results])

    for file in os.listdir(path):
      shutil.copy(f'{path}/{file}',
                'pneumatic/tools/Plot2Spec_materials_eyes/data/input_plot_extraction/leftImg8bit/test/abs_spectra/')

    # Apply the segmentation map
    plot_digitizer = PlotDigitizer()
    plot_digitizer.load_seg("spatialembedding", background_opt)
    plot_digitizer.predict_from_ins_seg(0, denoise=True)
    # result = Run(plot_digitizer, 0)

    res_map = plot_digitizer.result_dict['visual']
    img_rgb, seg_map, ins_map = res_map['img_rgb'], res_map['seg_map'], res_map['ins_map']
    masked_img = seg_map[..., None] * img_rgb
    y_resize_ratio = (y_max-y_min)/img_rgb.shape[0]
    print("y_resize_ratio", y_resize_ratio)
    centroid, labels = vq.kmeans2(masked_img.reshape((-1, 3)), 5, 50)

    ticks = [tick[0] for tick in results]
    # plt.imshow(masked_img, interpolation='none', extent=[320,1000,400,0])

    masked_img[np.where((masked_img==[0,0,0]).all(axis=2))] = [1,1,1]
    im=Image.fromarray((255*masked_img).astype(np.uint8))
    im.save('spec.jpeg')
    new = dilate_image('spec.jpeg')

    plt.imshow(new, interpolation='none', extent=[320,1000,400,0])#
    plt.savefig('dilated_im.png')
    img = PIL.Image.open('dilated_im.png')
    img.convert('RGB')
    # plt.imshow(img)
    paletted = img.convert('P', palette=PIL.Image.ADAPTIVE, colors=10)
    paletted.getcolors()
    # print(len(paletted.getcolors()))

           
    data = []
    for i in range(10):
      new = check_img(i, paletted)
      # print('new', new)
      # data.append(new)
      try:
        if new != None:
          data.append(new)
      #     print('data', data)
      #     # plot_ts, plot_lines = new
      #     # plot_line_norm = np.interp(norm_ts_pred, plot_ts, plot_lines[:,line_id])
      #     # norm_ts_pred = np.linspace(min(plot_ts), max(plot_ts), x_count)
      #     # norm_ts_pred = np.linspace(min(plot_ts), max(plot_ts), x_count)
      #     # plt.plot(norm_ts+x_bias, plot_line_norm*y_resize_ratio+y_bias, "o", markersize=8)
          
      except Exception as e:
          print('Error:', e)
          # print(data)

    dict_4 = dict(map(lambda i,j : (i,j) , [('plot_%s'%i) for i in range(len(data))],data))
    print('dict_4', dict_4)
    im = Image.open("dilated_im.png")
    im = im.convert('RGB')
    curve_colors = []
    plot_index = []
    for k in range(len(data)):
        try:
            x = data[k][0][1:]
            y = data[k][1][1:]
        except (IndexError, TypeError) as e:
            print(f"Skipping index {k} due to error: {e}")
            continue

        ls = [(int(i), int(j)) for i, j in zip(x, y)]
        pxls = []
        for idx, (xi, yi) in enumerate(ls):
            # Ensure coordinates are within image bounds
            if 0 <= xi < im.width and 0 <= yi < im.height:
                pixel = im.getpixel((xi, yi))
                # Check if pixel values are between 0 and 250 (exclusive)
                if all(0 < val < 250 for val in pixel):
                    pxls.append(pixel)
                    plot_index.append(k)
            else:
                print(f"Pixel coordinates ({xi}, {yi}) are out of bounds.")

        if pxls:
            mean_color = np.mean(pxls, axis=0)
            curve_colors.append(mean_color)
        else:
            print(f"No valid pixels found for data index {k}")
            curve_colors.append(np.array([np.nan, np.nan, np.nan]))
      
    # im = Image.open("dilated_im.png")
    # im.convert('RGB')
    # curve_colors=[]
    # plot_index = []
    # for k in range(len(data)):
    #   x,y = data[k][0][1:], data[k][1][1:]
    #   ls = [(int(i), int(j)) for i,j in zip(x, y)]
    #   pxls=[]
    #   for i,j in enumerate(ls):
    #     #print(i,j)
    #     if im.getpixel((int(ls[i][0]), int(ls[i][1])  ))[:-1] < (250, 250, 250) :
    #       if im.getpixel((int(ls[i][0]), int(ls[i][1])  ))[:-1] > (0, 0, 0) :
    #         pxls.append(im.getpixel((int(ls[i][0]), int(ls[i][1])))[:-1])
    #         plot_index.append(k)
    #   curve_colors.append(np.mean(pxls, axis=0))
      #print(np.mean(pxls, axis=0))
    cleanedList = [x for x in curve_colors if str(x) != 'nan']
    dict_2 = dict(map(lambda i,j : (i,j) , [list(dict_4.keys())[i] for i in set(plot_index)],cleanedList))

    colors = {}
    for key in ['plot_0', 'plot_1', 'plot_2', 'plot_3']:
        if key in dict_2:
            colors[key] = dict_2[key]
        else:
            print(f"Key '{key}' not found in dict_2.")

    # Normalize the RGB values to the range [0, 1] for matplotlib
    normalized_colors = {key: value / 255 for key, value in colors.items()}
    dict_2_norm = normalized_colors

    # Given color data
    colors = {
        list(dict_1.keys())[0]: dict_1[list(dict_1.keys())[0]],
        list(dict_1.keys())[1]: dict_1[list(dict_1.keys())[1]],
        list(dict_1.keys())[2]: dict_1[list(dict_1.keys())[2]],
        list(dict_1.keys())[3]: dict_1[list(dict_1.keys())[3]]
    }

    # Normalize the RGB values to the range [0, 1] for matplotlib
    normalized_colors = {key: value / 255 for key, value in colors.items()}
    dict_1_norm = normalized_colors


    
    def color_distance(color1, color2):
        return np.linalg.norm(color1 - color2)

    # Calculate the distance matrix between all color pairs
    distance_matrix = np.zeros((len(dict_1_norm ), len(dict_2_norm )))

    dict1_keys = list(dict_1_norm.keys())
    dict2_keys = list(dict_2_norm.keys())

    for i, color1 in enumerate(dict_1_norm .values()):
        for j, color2 in enumerate(dict_2_norm.values()):
            distance_matrix[i, j] = color_distance(color1, color2)

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Creating the dictionary for the optimal pairing
    optimal_pairing = {dict1_keys[i]: dict2_keys[j] for i, j in zip(row_ind, col_ind)}
    min_max_scaler = preprocessing.MinMaxScaler()
    new=[]
    for i in optimal_pairing.keys():
      #print(dict_4[dict_3[i]])
      new.append(dict_4[optimal_pairing[i]])
    new_dict = dict(map(lambda i,j : (i,j) , optimal_pairing.keys(),new))

    return new_dict

# Main execution
image_path = f"multimodal_data_folder_demo/abs_spectra/nmat2272_fig1_b.jpg"
#"conjugated_ECPs/uv_vis_automated/Polym. Chem., 2018, 9, 5262-5267/abs_spectra/Screen Shot 2022-11-21 at 1.54.54 PM.png" #

image = Image.open(image_path)
path = f"multimodal_data_folder_demo/abs_spectra/"#conjugated_ECPs/uv_vis_automated/Polym. Chem., 2018, 9, 5262-5267/abs_spectra/" 
final_dict = get_abs_dictionary(path, image_path)
print('final_dict', final_dict)

# plt.show()
def plot_segmented_data(dictionary):
    for label, (x,y) in dictionary.items():
      plt.scatter(x,y)
    plt.savefig('all_data.png')
    # plt.show()

plot_segmented_data(final_dict)