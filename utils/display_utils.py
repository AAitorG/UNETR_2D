from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# ### Functions to display
def plot_loss_and_metric(eval_metric, history, figsize=(14,14), save_fig_path=''): #multitask
    plt.figure(figsize=figsize)
    plt.tight_layout()

    f = np.ceil(len(eval_metric)/2)
    f = f if len(eval_metric)%2 == 0 else f + 1
    f = int(f)

    # summarize history for metrics
    for i, metric_name in enumerate(eval_metric):
        plt.subplot(f, 2, i+1)
        plt.plot(history.history[metric_name])
        plt.plot(history.history['val_'+metric_name])
        plt.title('model ' + metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

    if save_fig_path:
        plt.savefig(save_fig_path)
    else:
        plt.show()

def view_layers(model):
    for i, layer in enumerate(model.layers):
        print("{:^3} {:<20} {}".format(i, layer.name, layer.trainable))

def display(display_list, title = None, custom_size=False, save_fig_path='', mode = 'gray'):
  if not custom_size:
    plt.figure(figsize=(5*len(display_list), 5))

  if not title:
    title = ['Input Image', 'True Mask', 'Predicted Mask', 'Raw Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i], mode)
    #plt.axis('off')
  if save_fig_path:
    plt.savefig(save_fig_path)
  else:
    plt.show()

def save_only_image(img, filename):
    im = Image.fromarray(img)
    im = im.convert("L") # for grayscale images
    im.save( filename, quality=100, subsampling=0)
