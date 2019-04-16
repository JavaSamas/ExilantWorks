#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:12:41 2019

@author: samas
"""

import os
import re
from PIL import Image
from pdf2image import convert_from_path

path = "/Users/samas/Desktop/Test"

Image.MAX_IMAGE_PIXELS = 788325156

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)
#Image.errors.simplefilter('error', PDFPageCountError)


fname = []
for root,d_names,f_names in os.walk(path):
	for f in f_names:
		fname.append(os.path.join(root, f))

#print("fname = %s" %fname)
for fnam in fname :
    print(fnam)
    if fnam.endswith('.pdf') or fnam.endswith('.PDF') :
        pages = convert_from_path(fnam, 500)
        print(len(pages))
		# print re.sub('\.pdf$', '.jpg', fnam)
        #pages=pages[0:3]
        jpgnam = re.sub('\.pdf$', '.jpg', fnam,flags= re.IGNORECASE)
        for page in pages:
            pages[0].save(jpgnam, 'JPEG')
            pages[1].save(jpgnam, 'JPEG')
            pages[2].save(jpgnam, 'JPEG')
            
            
    

