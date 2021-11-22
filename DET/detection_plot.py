import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

from dle.inference import load_image, rescale, crop_center, normalize

import matplotlib.patches as patches
import json





# img = load_image('http://images.cocodataset.org/val2017/000000397133.jpg')
# plt.imshow(img)

all_box = np.load('SSD_Lap_0.5_box.npy').item()
all_class = np.load('SSD_Lap_0.5_class.npy').item()
all_score = np.load('SSD_Lap_0.5_score.npy').item()

for J in all_box.keys():
	K = J[14:]

	img = load_image('./val2017/'+K)

	img = rescale(img, 300, 300)
	img = crop_center(img, 300, 300)
	img = normalize(img)

	# plt.imshow(img)

	out = img/2+0.5
	# plt.imshow(out)
	# img.shape

	bboxes = all_box.get(J)
	classes = all_class.get(J)
	confidences = all_score.get(J)

	best = np.argwhere(confidences > 0.3).squeeze()
	if K == '000000560011.jpg' or K == '000000223188.jpg' or K == '000000045070.jpg' or K == '000000157213.jpg' or K == '000000131386.jpg' or K =='000000364636.jpg':
		continue
	print(K)

	json_file = './annotations/instances_val2017.json'
	with open(json_file,'r') as COCO:
	    js = json.loads(COCO.read())
	class_names = [ category['name'] for category in js['categories'] ]

	fig,ax = plt.subplots(1)
	ax.imshow(out)
	for idx in best:
	    left, top, right, bottom = bboxes[idx]
	    x, y, w, h = [val*300 for val in [left, top, right-left, bottom-top]]
	    rect = patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='r',facecolor='none')
	    ax.add_patch(rect)
	    ax.text(x, y, class_names[classes[idx]-1], bbox=dict(facecolor='white', alpha=0.5))
	# plt.show()
	plt.savefig(K)










