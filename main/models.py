from django.db import models

# Create your models here.

class Post(models.Model):
#    uploadFile = models.CharField(max_length=50)
    # 게시글 Post에 이미지 추가
    uploadImg = models.ImageField(blank=True, null=True)
    #print(len(uploadImg))
    # 여기서 가져오는 이미지 개수 확인 및 마지막번째만 가져오기? 
    # 제목(postname)이 Post object 대신하기
    def __str__(self):
        return self.fileName
