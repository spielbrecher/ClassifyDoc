# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 13:16:55 2020

@author: Andy
"""
import numpy as np
import pandas as pd
import pytesseract
import json
import os

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO
from PIL import Image
from PyPDF2 import PdfFileReader, generic
import zlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
 
 


def get_color_mode(obj):

    try:
        cspace = obj['/ColorSpace']
    except KeyError:
        return None

    if cspace == '/DeviceRGB':
        return "RGB"
    elif cspace == '/DeviceCMYK':
        return "CMYK"
    elif cspace == '/DeviceGray':
        return "P"

    if isinstance(cspace, generic.ArrayObject) and cspace[0] == '/ICCBased':
        color_map = obj['/ColorSpace'][1].getObject()['/N']
        if color_map == 1:
            return "P"
        elif color_map == 3:
            return "RGB"
        elif color_map == 4:
            return "CMYK"


def get_object_images(x_obj):
    images = []
    for obj_name in x_obj:
        sub_obj = x_obj[obj_name]

        if '/Resources' in sub_obj and '/XObject' in sub_obj['/Resources']:
            images += get_object_images(sub_obj['/Resources']['/XObject'].getObject())

        elif sub_obj['/Subtype'] == '/Image':
            zlib_compressed = '/FlateDecode' in sub_obj.get('/Filter', '')
            if zlib_compressed:
               sub_obj._data = zlib.decompress(sub_obj._data)

            images.append((
                get_color_mode(sub_obj),
                (sub_obj['/Width'], sub_obj['/Height']),
                sub_obj._data
            ))

    return images


def get_pdf_images(pdf_fp):
    images = []
    try:
        pdf_in = PdfFileReader(open(pdf_fp, "rb"))
    except:
        return images

    for p_n in range(pdf_in.numPages):

        page = pdf_in.getPage(p_n)

        try:
            page_x_obj = page['/Resources']['/XObject'].getObject()
        except KeyError:
            continue

        images += get_object_images(page_x_obj)

    return images


if __name__ == "__main__":
    # set path to tesseract.exe if not in PATH environment
    pytesseract.pytesseract.tesseract_cmd = r'h:/soft/tesseract/tesseract.exe'
       
    k=0
    textz = []
    text_labels = []
    df = pd.DataFrame()
    #Ограничились одним каталогом из-за длительности обработки
    names = os.listdir(os.getcwd()+'/Dataset/Rent') 
    for pdf_fp in names:
        k+=1
        if k==10:
            break
        print(pdf_fp)
        # Перебираем картинки в PDF-файле
        fullname = os.path.join(os.getcwd()+'/Dataset/Rent/', pdf_fp)
        for image in get_pdf_images(fullname):
            (mode, size, data) = image
            try:
                img = Image.open(StringIO(data))            
                # Распознаем с помощью Tesseract
                text = pytesseract.image_to_string(img.convert("RGB"), lang="rus")
                textz.append(text)
                text_labels.append(1) # Номер класса
                # Можно просто по ключевому слову идентифицировать
                if(str.lower(text).find('договор аренды')!=-1):
                    print('Recognized')
            except Exception as e:
                print ("Failed to read image with PIL: {}".format(e))
                continue
    
    # Вариант классификатора по текстам Random Forest с TF-IDF   
     
    text_clf = Pipeline([
                         ('tfidf', TfidfVectorizer()),
                         ('clf', RandomForestClassifier())
                         ])
     
    text_clf.fit(textz[:-1], text_labels[:-1])
     
    res = text_clf.predict([text[-1]])
    print(res)
    input()

    