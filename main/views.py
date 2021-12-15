from django.db import models
from django.shortcuts import render

from .__init__ import change_img, check_face, face_model, mask_model
from .models import Post



# Create your views here.

def upload(request):

    if request.method == "POST":
    #    filename = request.POST['filename']
        faceimg = request.FILES['faceimg']
        print(type(faceimg))
        # DB에 저장
        post = Post(
        #    uploadFile = filename,
            uploadImg = faceimg
        )
        post.save()

        posts = Post.objects.last()
        
        check_face_img , number = check_face( posts.uploadImg.url[1:])


        check_face_img.save("static\check_face.jpg",'jpeg')
        return render(request, 'main/check.html', {'posts' : posts, 'number' : range(1, number+1)})

    else:

        # 기존 이미지 삭제?
        return render(request, 'main/index.html')

def download(request):

    if request.method == "POST":
        str_check_list = request.POST.getlist('check[]')
        check_list = []
        for i in str_check_list :
            check_list.append(int(i)-1)
        
        change_img("static\origin.jpg",check_list ,face_model, mask_model, batch_size = 16).save("static\Result.jpg",'jpeg')
        return render(request, 'main/download.html')
    else:

        # 기존 이미지 삭제?
        return render(request, 'main/index.html')
