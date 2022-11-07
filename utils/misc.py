import numpy as np

def tensor_denormalize_to_numpy(img_tensor, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    img_tensor = img_tensor[0]
    img = img_tensor.cpu().detach().numpy()
    img = np.float32(img)

    for i in range(len(mean)):
        img[i] = img[i] * std[i] + mean[i]

    img = img.transpose(1, 2, 0) * 255

    return np.uint8(img)

