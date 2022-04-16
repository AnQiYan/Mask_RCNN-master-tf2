import os
from PIL import Image
import random
import shutil

class ImageProcess():
    def __init__(self):
        self.files = [f for f in os.listdir('.') if '.JPG' in f or '.PNG' in f or '.jpg' in f or '.jpeg' in f]
        self.file_name_path = [os.getcwd() + '/' + files_name for files_name in self.files]
        if not os.path.exists('processed'):
            os.mkdir('processed')
            print('Director created')

    def resize_image(self, w = 600, h = 450):
        for image in self.files:
            output_path = os.getcwd() + '/' + 'processed/' + 'PRO{}.jpg'.format(random.randint(10000,99999))
            cur_img_path = os.getcwd() + '/' + image
            img = Image.open(image)
            if img.height == h and img.width == 600:
                shutil.copy(cur_img_path, output_path)
                print('image moved to processed')
            else:
                out = img.resize((w, h), Image.ANTIALIAS)
                out.save(output_path)
            print('[{}] processed'.format(image))

processor = ImageProcess()
processor.resize_image()
