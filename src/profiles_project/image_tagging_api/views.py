from django.shortcuts import render

from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework import filters
from rest_framework.authtoken.serializers import AuthTokenSerializer
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.permissions import IsAuthenticated

from PIL import Image

from . import serializers
from . import models

from io import BytesIO
import base64
import subprocess
import json
import os
import sys

# Create your views here.

class UploadImageViewSet(viewsets.ViewSet):
    """Test API ViewSet."""

    self.IMAGE_DIR = '/home/ubuntu/data/ImageNet_mini/'
    self.PROJECT_DIR = '/usr/local/apps/profiles-rest-api'
    #serializer_class = serializers.HelloSerializer

    def list(self, request):
        """Return a hello message"""

        tag_id = self.request.query_params.get('tag_id')
        num = int(self.request.query_params.get('num'))
        count = int(self.request.query_params.get('count'))
        width = int(self.request.query_params.get('width'))

        # no such category in server
        if not os.path.exists(self.IMAGE_DIR + tag_id):
            return Response({'message': 'no such id', 'image': ''})

        if count == 1:
            ls = os.listdir(self.IMAGE_DIR + tag_id)
            # discard '.DS_Store' file
            if num >= len(ls):
                return Response({'message': 'fail', 'image': ''})
            path = self.IMAGE_DIR + tag_id + '/' +ls[num]
            with open(path, 'rb') as imageFile:
                image_str = base64.b64encode(imageFile.read())
            return Response({'message': 'success', 'image': image_str})

        if count > 1:
            ls = os.listdir(self.IMAGE_DIR + tag_id)
            image_arr = []
            has_more = True
            for i in range(num, num + count):
                if i >= len(ls):
                    has_more = False
                    break
                path = self.IMAGE_DIR + tag_id + '/' +ls[i]

                img = Image.open('path')
                w, h = img.size
                new_img = img.resize((width, int(h * width / w)))
                buffered = BytesIO()
                new_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue())
                tuple = {'filename': ls[i], 'image': image_str}
                image_arr.append(tuple)

            return Response({'message': 'success', 'image': image_arr, 'has_more': has_more})

        return Response({'message': 'fail', 'status': '400', 'image': ''})




    def create(self, request):
        """upload an image to recognize"""

        #serializer = serializers.HelloSerializer(data=request.data)
        fh = open(self.PROJECT_DIR + "/imageToSave.JPEG", "wb")
        encoded = base64.b64decode(request.data['image'])
        fh.write(encoded)
        cmd = 'PYTHONPATH=' + self.PROJECT_DIR + '/android-model403/ python -m scripts.label_image  --image=' + self.PROJECT_DIR + '/imageToSave.jpeg'
        result = subprocess.check_output(cmd, shell=True)

        string = result.decode(encoding='UTF-8')
        data = json.loads(string)
        return Response({'message': 'success', 'data': data})

        # if serializer.is_valid():
        #     name = serializer.data.get('name')
        #     message = 'Hello {0}'.format(name)
        #     return Response({'message': message})
        # else:
        #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
