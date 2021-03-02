import torch
import torch.nn.functional
import torchvision
import numpy as np

from pytorch_wavelets import DTCWTForward, DTCWTInverse
from PIL import Image

import clip
import random

dtype = torch.float
device = torch.device("cuda:0")

clip_model, _ = clip.load("ViT-B/32", device=device) # RN50 or ViT-B/32

# Init tensor from image
im = Image.open('mona.png')
img_size = 224 * 5
im = im.resize((img_size, img_size), Image.ANTIALIAS)

im2arr = np.array(im)
im2arr = 1.0 * im2arr / 255
im2arr = np.transpose(im2arr, (2, 0, 1))

X = torch.tensor([im2arr],  device=device, dtype=dtype, requires_grad=True)

xfm = DTCWTForward(J=3).to(device=device)
ifm = DTCWTInverse().to(device=device)

Yl, Yh = xfm(X)
print(Yl.shape)
print(Yh[0].shape)
print(Yh[1].shape)
print(Yh[2].shape)

Yl = Yl.clone().detach().requires_grad_(True)
Yh[0] = Yh[0].clone().detach().requires_grad_(True)
Yh[1] = Yh[1].clone().detach().requires_grad_(True)
Yh[2] = Yh[2].clone().detach().requires_grad_(True)

learning_rate = .5
weight_decay = 1e-5

optimizer = torch.optim.SGD([Yl, Yh[0], Yh[1], Yh[2]], lr=learning_rate)#, weight_decay=weight_decay)

text = clip.tokenize(["mona lisa sad face"]).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text)

for t in range(100000):
    outputs = []
    optimizer.zero_grad()

    Y = ifm((Yl, Yh))#.clip(0, 1)
    for s in range(100):
        cropped_image = torchvision.transforms.functional.affine(Y, angle=random.uniform(-15, 15), translate=(random.uniform(-30, 30), random.uniform(-30, 30)), scale=random.uniform(0.9, 1.1), shear=1.0)

        cropped_size = int(random.uniform(img_size / 2, img_size))
        cropped_image = torchvision.transforms.functional.crop(cropped_image, int(random.uniform(0, img_size-cropped_size)), int(random.uniform(0, img_size-cropped_size)), cropped_size, cropped_size)
        cropped_image = torch.nn.functional.interpolate(cropped_image, (224, 224), mode='bilinear') # , mode='area'

        cropped_image = torchvision.transforms.functional.normalize(cropped_image, (0.48145466, 0.4578275, 0.40821073),  (0.26862954, 0.26130258, 0.27577711))
        outputs.append(cropped_image)


    image_features = clip_model.encode_image(torch.cat(outputs, dim=0))
    loss_multiplier = 5000

    loss = -loss_multiplier * torch.cosine_similarity(text_features, image_features, dim=-1).mean()
    loss.backward()

    optimizer.step()


    print(t, loss.item() / loss_multiplier)

    if t % 5 == 0 and t != 0:
        with torch.no_grad():
            Y = ifm((Yl, Yh)).clip(0, 1)
            imgs = torch.transpose(Y.detach().cpu(), 1, 3)
            imgs = torch.transpose(imgs, 1, 2)

            im = Image.fromarray((imgs[0].numpy() * 255).astype(np.uint8))
            im.save("res.png")
