from django.urls import path
from main.views import *

# 이미지 업로드 하자
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('upload/', upload, name="upload"),
    path('download/', download, name="download")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)