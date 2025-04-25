import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr
import torch
from scipy.interpolate import interp1d
from pyod.models.knn import KNN
from scipy.cluster import vq
from PIL import Image
from custom_tools.Plot2Spec_materials_eyes.src.plot_data_extraction.plot_digitizer import PlotDigitizer
from custom_tools.Plot2Spec_materials_eyes.src.plot_data_extraction.SpatialEmbeddings.src.utils import transforms as my_transforms
from custom_tools.Plot2Spec_materials_eyes.src.axis_alignment.utils import AxisAlignment
from sklearn import preprocessing
import shutil
import os
import pandas as pd
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
import PIL
# Configuration dictionaries
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

# Helper functions
def recognize_text(img_path):
    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path)

def overlay_ocr_text(img_path):
    points = []
    labels = []
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = recognize_text(img_path)
    
    for (bbox, text, prob) in result:
        if len(text) > 1 and text[0].isalpha() and text[:10] != 'Wavelength' and text[:2] != 'nm' and '00' not in text and prob >= 0.4:
            labels.append(text)
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            points.append(bbox)
            cv2.rectangle(img, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=10)
            cv2.putText(img, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=8)
    return labels, points

def get_label_colors(image_path,i, labels, points):
    """Extract the color of a specific label from the legend of the image."""
    lab = labels[i]
    top_left, bottom_right = points[i][0], points[i][2]
    with Image.open(image_path) as im:
        cropped_image = im.crop((top_left[0] - 100, (bottom_right[1] + top_left[1]) / 2 - 2, top_left[0], (bottom_right[1] + top_left[1]) / 2 + 2))
    cropped_image_rgb = np.array(cropped_image.convert("RGB"))    
    mask = np.all(cropped_image_rgb < [250, 250, 250], axis=2)
    pxls = cropped_image_rgb[mask]
    return lab, np.mean(pxls, axis=0) if pxls.size > 0 else None

def create_dictionary(image_path):
    labels, points = overlay_ocr_text(image_path)
    labs = []
    rgb = []
    for i in range(len(labels)):
        labelaki, color = get_label_colors(image_path, i, labels, points)
        labs.append(labelaki)
        rgb.append(color)
    return dict(zip(labs, rgb))

def fig_text_main(path):
    dict_1 = create_dictionary(path)
    dt_keys = dict_1.keys()
    try:
        for i in dt_keys:
            if len(set(dict_1[i])) == 1:
                dict_1.pop(i, None)
    except:
        pass
    return dict_1

def read_img(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.

def write_img(img, path):
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def derivative(x_data, y_data):
    N = len(x_data)
    delta_x = [x_data[i+1] - x_data[i] for i in range(N - 1)]
    x_prim = [(x_data[i+1] + x_data[i]) / 2. for i in range(N - 1)]
    y_prim = [(y_data[i+1] - y_data[i]) / delta_x[i] for i in range(N - 1)]
    return x_prim, y_prim

def check_img(i, paletted):
    y, x = np.where(np.asarray(paletted) == i)
    df = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
    if df.shape[0] < 7000:
        Y = df.values
        n_neighbors = min(5, df.shape[0])
        clf = KNN(n_neighbors=n_neighbors, radius=0.3, contamination=0.2)
        clf.fit(Y)
        outliers = clf.predict(Y)
        into = np.where(outliers == 0)
        if np.mean(clf.decision_scores_) < 7 and len(np.where(outliers == 1)[0]) > 0:
            x1 = x[into] + 300
            y1 = y[into]
            x_bis, y_bis = derivative(*derivative(x1, y1))
            if np.mean(x_bis) > 150:
                f = interp1d(x1, y1, fill_value="extrapolate")
                wavelength = np.arange(x1.min(), x1.max(), 1)
                #plt.gca().invert_yaxis()
                #plt.scatter(wavelength, f(wavelength))
                return wavelength, f(wavelength)

def dilate_image(image_path):
  img = read_img(image_path)
  img = np.abs(img - 1)
  print(img.mean())
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  img = cv2.dilate(img, kernel, iterations=1)
  img = np.abs(img - 1)
  write_img(img, image_path)
  return img

# def dilate_image(img):
#     """Dilate the input image to enhance features."""
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     dilated_img = cv2.dilate(img, kernel, iterations=1)
#     return np.abs(dilated_img - 1)

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
    #plt.imshow(masked_img, interpolation='none', extent=[320,1000,400,0])

    masked_img[np.where((masked_img==[0,0,0]).all(axis=2))] = [1,1,1]
    im=Image.fromarray((255*masked_img).astype(np.uint8))
    im.save('spec.jpeg')
    new = dilate_image('spec.jpeg')

    plt.imshow(new, interpolation='none', extent=[320,1000,400,0])#
    plt.savefig('dilated_im.png')
    img = PIL.Image.open('dilated_im.png')
    img.convert('RGB')
    paletted = img.convert('P', palette=PIL.Image.ADAPTIVE, colors=10)
    paletted.getcolors()
    print(len(paletted.getcolors()))           
    # import io
    # # Create an in-memory buffer to store the image
    # buffer = io.BytesIO()

    # # Display the image with no interpolation and specific extents
    # plt.imshow(new, interpolation='none', extent=[320, 1000, 400, 0])

    # # Save the image to the in-memory buffer as a PNG file
    # plt.savefig(buffer, format='png')

    # # Move the buffer's position to the start
    # buffer.seek(0)

    # # Open the image from the in-memory buffer and convert it to RGB format
    # img = Image.open(buffer).convert('RGB')

    # # Convert the RGB image to a paletted image with an adaptive palette of 10 colors
    # paletted = img.convert('P', palette=Image.ADAPTIVE, colors=10)

    # Get the colors from the paletted image
    colors = paletted.getcolors()

    # Close the buffer
    # buffer.close()

    data = [check_img(i, paletted) for i in range(10) if check_img(i, paletted) is not None]

    dict_4 = dict(map(lambda i,j : (i,j) , [('plot_%s'%i) for i in range(len(data))],data))      
    
    # im = Image.open("dilated_im.png")
    # im.convert('RGB')
    curve_colors=[]
    plot_index = []
    for k in range(len(data)):
      x,y = data[k][0][1:]-350, data[k][1][1:]
      ls = [(int(i), int(j)) for i,j in zip(x, y)]
      pxls=[]
      for i,j in enumerate(ls):
        #print(i,j)
        if im.getpixel((int(ls[i][0]), int(ls[i][1])  ))[:-1] < (250, 250, 250) :
          if im.getpixel((int(ls[i][0]), int(ls[i][1])  ))[:-1] > (0, 0, 0) :
            pxls.append(im.getpixel((int(ls[i][0]), int(ls[i][1])))[:-1])
            plot_index.append(k)
      curve_colors.append(np.mean(pxls, axis=0))
      #print(np.mean(pxls, axis=0))
    cleanedList = [x for x in curve_colors if str(x) != 'nan']
    dict_2 = dict(map(lambda i,j : (i,j) , [list(dict_4.keys())[i] for i in set(plot_index)],cleanedList))

    # Given color data
    colors = {
        'plot_0': dict_2['plot_0'],
        'plot_1': dict_2['plot_1'],
        'plot_2': dict_2['plot_2'],
        'plot_3': dict_2['plot_3']
    }

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


    from scipy.optimize import linear_sum_assignment
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


def plot_segmented_data(dictionary):
    for label, (x,y) in dictionary.items():
      plt.scatter(x,y)
    plt.show()

# Main execution
image_path = "multimodal_data_folder_demo/abs_spectra/nmat2272_fig1_b.jpg"
image = Image.open(image_path)
path = "multimodal_data_folder_demo/abs_spectra/"
final_dict = get_abs_dictionary(path, image_path)
print('final_dict', final_dict)
plot_segmented_data(final_dict)