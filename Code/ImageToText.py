#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:00:48 2018

@author: samas
"""

import pytesseract
from PIL import Image
#pytesseract.tesseract_cmd = "/Users/samas/Desktop/Anaconda/anaconda3/bin/tesseract"
img=Image.open('/Users/samas/Desktop/BIRF-ZZ-BYECS-510-0003-page-002.jpg')

result=pytesseract.image_to_string(img)
print(result)