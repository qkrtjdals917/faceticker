import cv2
import os
import time
import glob

import torch
from torch.utils.data import Dataset,DataLoader

import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets,transforms,models

import face_recognition as fr

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image # 이미지 크기 조절

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 얼굴 나이 및 성별 인식 모델
face_model = models.resnet34(pretrained=True)

for param in face_model.parameters():
    param.requires_grad=False
    
n_inputs = face_model.fc.in_features
face_model.fc = nn.Linear(n_inputs, 20)

# 마스크 사용 여부 인식 모델

mask_model = models.resnet34(pretrained=True)
for param in mask_model.parameters():
    param.requires_grad=False
    
n_inputs = mask_model.fc.in_features
mask_model.fc = nn.Linear(n_inputs, 3)


face_model.load_state_dict(torch.load("data/model/model_Ver5.pt", map_location=torch.device('cpu')))
face_model.eval()

mask_model.load_state_dict(torch.load("data/model/model_mask_1209.pt", map_location=torch.device('cpu')))
mask_model.eval()

transform = transforms.Compose([
    transforms.Resize((80, 80)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
])

result_age = [
    'man_0',
    'man_10',
    'man_20',
    'man_30',
    'man_40',
    'man_50',
    'man_60',
    'man_70',
    'man_80',
    'man_90',
    'woman_0',
    'woman_10',
    'woman_20',
    'woman_30',
    'woman_40',
    'woman_50',
    'woman_60',
    'woman_70',
    'woman_80',
    'woman_90'
]

result_mask = [
    'with_mask',
    "without_mask",
    "mask_weared_incorrect"
]


aniface_name = [
    'baby.png',
    'one1.png',
    'ten1.png',
    'three1.png',
    'four1.png',
    'five1.png',
    'six1.png',
    'seven1.png',
    'seven1.png',
    'seven1.png',
    'baby.png',
    'one2.png',
    'ten2.png',
    'three2.png',
    'four2.png',
    'five2.png',
    'six2.png',
    'seven2.png',
    'seven2.png',
    'seven2.png'
]
def check_face (image_path) :
    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image)
    
    Image.open(image_path).save("static\origin.jpg",'jpeg')

    font =  cv2.FONT_HERSHEY_PLAIN
    count = 1
    for (top, right, bottom, left) in face_locations:
        # (그릴 곳, 시작점, 끝점, 색, 두께)
        cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 3)
        cv2.putText(image, str(count), (left+3, bottom-3), font, 4, (255,255,0), 3, cv2.LINE_AA)
        count += 1
    pil_image=Image.fromarray(image)
    return pil_image , count-1


def two_predictions (image_path, face_model, mask_model, batch_size = 16) : 
    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image)
    
    actor_faces = []
    coordinate_list = []
    face_predictions = []
    mask_predictions = []
    
    for (top, right, bottom, left) in face_locations:
        tmp_image = image[:]
        face_image = tmp_image[top:bottom, left:right]
        actor_faces.append(face_image)
        coordinate_list.append([ (left, top), right-left, bottom-top] )  
        
    for i in range(    round(((len(actor_faces)-1) / batch_size) + 0.5)   ):
        batch = actor_faces[i*batch_size:(i+1)*batch_size]
        inputs = []
        for j, face in enumerate(batch):
            img = transform(Image.fromarray(face)) 
            inputs.append(img)
        inputs = torch.stack(inputs).to(device)
        face_preds = face_model(inputs)
        face_result = torch.argmax(torch.softmax(face_preds, dim=1) , dim=1)

        mask_preds = mask_model(inputs)
        mask_result = torch.argmax(torch.softmax(mask_preds, dim=1) , dim=1)
        
        face_predictions.append(face_result) 
        mask_predictions.append(mask_result)     
    
    return face_predictions , mask_predictions , actor_faces, coordinate_list


def change_img (image_path, number_list, face_model, mask_model, batch_size = 16) : 

    face_predictions , mask_predictions , actor_faces, coordinate_list =  \
        two_predictions (image_path, face_model, mask_model, batch_size = 16)

    count = 0
    back = Image.open(image_path)

    for coordinate in coordinate_list:
        batch_count =  count // batch_size
        realCount = count - (batch_count*batch_size)
        if count not in number_list :

            start_xy = coordinate[0]
            width = coordinate[1]
            height = coordinate[2]

            aniface_path = 'data/new/'

            if mask_predictions[batch_count][realCount] == 0 :
                aniface_path = 'data/mask/'
                

            aniface = Image.open(aniface_path + aniface_name[face_predictions[batch_count][realCount]] )
            aniface = aniface.resize((width, height))
            back.paste(aniface, start_xy, aniface)

        count +=1 
    return back


