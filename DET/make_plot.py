import numpy as np
from matplotlib import pyplot as plt
import torch
# %matplotlib inline


from dle.inference import load_image, rescale, crop_center, normalize

from examples.SSD300_inference import load_checkpoint, build_predictor

from apex.fp16_utils import network_to_half

from src.utils import dboxes300_coco, Encoder
import matplotlib.patches as patches
import json





# img = load_image('http://images.cocodataset.org/val2017/000000397133.jpg')
# plt.imshow(img)

import glob

all_img_names = glob.glob("/coco/val2017/*.jpg")

all_box = {}
all_class = {}
all_score = {}

for K in all_img_names[:100]:

	img = load_image(K)

	img = rescale(img, 300, 300)
	img = crop_center(img, 300, 300)
	img = normalize(img)

	# plt.imshow(img)

	out = img/2+0.5
	# plt.imshow(out)
	# img.shape

	ssd300 = build_predictor('./logs/SSD_Lap_0.5.pth.tar')
	ssd300 = ssd300.cuda()
	ssd300 = network_to_half(ssd300.cuda())
	ssd300 = ssd300.eval()


	# change the shape
	HWC = img
	CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
	# make a batch of 1 image
	batch = np.expand_dims(CHW, axis=0)
	# turn input into tensor
	tensor = torch.from_numpy(batch)
	tensor = tensor.cuda()
	tensor = tensor.half()
	# tensor.shape


	prediction = ssd300(tensor)

	dboxes = dboxes300_coco()
	encoder = Encoder(dboxes)
	ploc, plabel = [val.float() for val in prediction]
	encoded = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)

	bboxes, classes, confidences = [x.detach().cpu().numpy() for x in encoded[0]]


	# best = np.argwhere(confidences > 0.1).squeeze()

	all_box[K] = bboxes
	all_class[K] = classes
	all_score[K] = confidences

np.save('SSD_Lap_0.5_box', all_box)
np.save('SSD_Lap_0.5_class', all_class)
np.save('SSD_Lap_0.5_score', all_score)

# json_file = '/coco/annotations/instances_val2017.json'
# with open(json_file,'r') as COCO:
#     js = json.loads(COCO.read())
# class_names = [ category['name'] for category in js['categories'] ]

# fig,ax = plt.subplots(1)
# ax.imshow(out)
# for idx in best:
#     left, top, right, bottom = bboxes[idx]
#     x, y, w, h = [val*300 for val in [left, top, right-left, bottom-top]]
#     rect = patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
#     ax.text(x, y, class_names[classes[idx]-1], bbox=dict(facecolor='white', alpha=0.5))
# plt.show()










