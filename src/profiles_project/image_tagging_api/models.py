from django.db import models

# Create your models here.

def scramble_uploaded_filename(instance, filename):
    """
    Scramble / uglify the filename of the uploaded file, but keep the files extension (e.g., .jpg or .png)
    :param instance:
    :param filename:
    :return:
    """
    extension = filename.split(".")[-1]
    return "{}.{}".format(uuid.uuid4(), extension)

# class UploadedImage(models.Model):
#     """recieve uploaded images."""
#     created = models.DateTimeField(auto_now_add=True)
#     #owner = models.ForeignKey(User, to_field='id')
#     image_data = models.ImageField(upload_to='uploaded_images/' + scramble_uploaded_filename, blank=True)
#     # score = models.FloatField()
#     # tag = models.CharField(max_length=255)
#     # tag_id = models.CharField(max_length=20)
#
# class ImageTaggingResult(models.Model):
#     """recieve uploaded images."""
#     created = models.DateTimeField(auto_now_add=True)
#     #owner = models.ForeignKey(User, to_field='id')
#     image_data = models.ImageField(upload_to='uploaded_images/' + scramble_uploaded_filename, blank=True)
#     # score = models.FloatField()
#     # tag = models.CharField(max_length=255)
#     # tag_id = models.CharField(max_length=20)
