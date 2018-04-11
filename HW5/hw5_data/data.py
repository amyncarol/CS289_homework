from skimage.io import imread, imsave
from skimage.transform import resize
import os

for i in os.listdir('data_raw/'):
	name = i.split('.')[0]
	print(name)
	im = imread(os.path.join('data_raw', name+'.jpg'))
	n = im.shape[0]
	m = im.shape[1]
	n_new = 1000
	m_new = 1000*m//n
	im_resized = resize(im, (n_new, m_new))
	print(im_resized.dtype)
	imsave(os.path.join('data', name+'.png'), im_resized)