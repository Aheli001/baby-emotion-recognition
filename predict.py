import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from tensorflow.keras.models import load_model
# %matplotlib inline

label = ['angry','disgust','fear','happy','neutral','sad','surprise']


def ef(image):
    img = load_img(image,color_mode="grayscale")
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

model = load_model('BabyEmotion.h5')
image="images/train/happy/7.jpg"
img = ef(image)
pred = model.predict (img)
p_label = label[pred.argmax()]
print("This is a image of ",p_label)
plt.imshow(img.reshape(48,48),cmap = 'gray')
