import cv2
from PIL import Image
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet50
from IBA import per_sample_IBA
from ImageNetDataloader import data_loader
from test import test
from utils.misc import *
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs')
writer_example = SummaryWriter('runs/example')

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str, size=(224, 224), device="cpu", num_cls=1000, mean=None, std=None) -> None:
        self.fmap_gradupdate = None
        self.model = model
        self.model.eval()

        getattr(self.model, target_layer).register_forward_hook(self.__forward_hook)
        getattr(self.model, target_layer).register_backward_hook(self.__backward_hook)

        self.size = size
        self.origin_size = None
        self.num_cls = num_cls
        self.device = device

        if mean and std:
            self.mean, self.std = mean, std
        else:
            self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.grads = [None]
        self.fmaps = [None]



    def forward(self, img, label, IBA_heatmap_shape, classes_name, show=True, write=False):
        self.origin_size = (img.shape[2], img.shape[3])
        self.img_input = img
        self.label = label

        self.grads.clear()

        output = self.model(self.img_input)
        idx = np.argmax(output.cpu().data.numpy())

        output_softmax = torch.nn.functional.softmax(output, dim=1).data.squeeze()
        print("prediced:"+str(idx)+'('+ str(output_softmax[idx].item()) +')')

        self.prediced_idx = str(idx)
        self.prediced_idx_probability = str(round(output_softmax[idx].item(), 3))

        self.label_idx = str(self.label[0].item())
        self.label_idx_probability = str(round(output_softmax[self.label[0].item()].item(), 3))

        self.model.zero_grad()
        loss = self.__compute_loss_label(output, label)
        loss.backward(retain_graph=True)

        self.model.zero_grad()
        loss = self.__compute_loss_predict(output, int(self.prediced_idx))
        loss.backward()

        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        grads_val_predict = self.grads[1].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        self.grad, self.grad_predict, self.fmap = grads_val, grads_val_predict, fmap

        cam = self.__compute_cam(fmap, grads_val)
        cam_predict = self.__compute_cam(fmap, grads_val_predict)

        cam_show = cv2.resize(cam, self.origin_size)
        cam_predict_show = cv2.resize(cam_predict, self.origin_size)
        img_show = tensor_denormalize_to_numpy(img).astype(np.float32) / 255
        self.__show_cam_on_image(img_show, cam_show, cam_predict_show,IBA_heatmap_shape, classes_name, if_show=show, if_write=write)

        return cam_show

    def saliency_loss(self, saliency_Grad, saliency_IB, saliency_Grad_predict):

        w = 100
        b = saliency_IB.max() * 0.55
        mask = 1 / ( 1 + torch.exp( - w * (saliency_Grad - b)))

        loss1 = torch.sum(torch.min(saliency_Grad, saliency_Grad_predict) * mask) \
               / torch.sum(saliency_Grad + saliency_Grad_predict)

        criterion = torch.nn.CrossEntropyLoss()
        loss2 = criterion(saliency_Grad, saliency_IB)

        loss = loss1 + loss2

        return loss

    def compute_IBA_GradCAM_grad(self, IBA_heatmap: torch.Tensor, lamb, lr = 0.00001):

        self.model.zero_grad()
        self.img_input.requires_grad = True

        output_reduce_fc = self.model(self.img_input, layer4_stop = True)

        cam_grad = torch.from_numpy(self.grad).to(self.device) * output_reduce_fc
        cam_grad = torch.sum(torch.sum(cam_grad, 0), 0)
        cam_grad = cam_grad - cam_grad.min()
        cam_grad = cam_grad / cam_grad.max()

        cam_grad_predict = torch.from_numpy(self.grad_predict).to(self.device) * output_reduce_fc
        cam_grad_predict = torch.sum(torch.sum(cam_grad_predict, 0), 0)
        cam_grad_predict = cam_grad_predict - cam_grad_predict.min()
        cam_grad_predict = cam_grad_predict / cam_grad_predict.max()

        loss2 = self.saliency_loss(cam_grad, IBA_heatmap, cam_grad_predict)

        output = self.model(self.img_input)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr)
        loss1 = criterion(output, self.label)
        loss = loss1 + lamb * loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def __img_transform(self, img_arr: np.ndarray, transform: torchvision.transforms) -> torch.Tensor:
        img = img_arr.copy()
        img = Image.fromarray(np.uint8(img))
        img = transform(img).unsqueeze(0)
        return img

    def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
        self.origin_size = (img_in.shape[1], img_in.shape[0])
        img = img_in.copy()
        img = cv2.resize(img, self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        img_tensor = self.__img_transform(img, transform)
        return img_tensor

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0])

    def __forward_hook(self, module, input, output):
        self.fmaps[0] = output

    def __compute_loss_label(self, logit, index=None):
        if not index:
            index = np.argmax(logit.cpu().data.numpy())
        else:
            index = np.array(index.cpu())

        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        loss = logit[0, index]
        return loss

    def __compute_loss_predict(self, logit, index=None):
        index = np.array(index)

        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        loss = logit[0, index]
        return loss

    def __compute_cam(self, feature_map, grads):
        cam = grads * feature_map
        cam = np.sum(cam, axis=0)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.size)
        cam = (cam - np.min(cam)) / np.max(cam)
        return cam

    def read_json(self, file_name):
        import  json
        with open(file_name,'r',encoding='UTF-8') as f:
            self.json = json.loads(f.read())

    def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, cam_predict_show,IBA_heatmap_shape, classes_name, if_show=True, if_write=False):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)

        heatmap_predict = cv2.applyColorMap(np.uint8(255 * cam_predict_show), cv2.COLORMAP_JET)
        heatmap_predict = np.float32(heatmap_predict) / 255
        cam_predict = heatmap_predict + np.float32(img)
        cam_predict = cam_predict / np.max(cam_predict)
        cam_predict = np.uint8(255 * cam_predict)

        IBA_heatmap_shape = cv2.applyColorMap(np.uint8(255 * IBA_heatmap_shape), cv2.COLORMAP_JET)
        IBA_heatmap_shape = np.float32(IBA_heatmap_shape) / 255
        IBA_cam = IBA_heatmap_shape + np.float32(img)
        IBA_cam = IBA_cam / np.max(IBA_cam)
        IBA_cam = np.uint8(255 * IBA_cam)

        self.read_json('data/imagenet_class_index.json')

        plt.figure(figsize=(12,4))

        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Input"+"("+self.json[self.label_idx][1]+")"+"\n"+"  label :"+self.label_idx+"  probability:"+self.label_idx_probability\
                  +"\n"+"predict:"+self.prediced_idx+"  probability:"+self.prediced_idx_probability)

        plt.subplot(1, 4, 2)
        plt.imshow(cam[:, :, ::-1])
        plt.title("CAM Heatmap(label)")

        plt.subplot(1, 4, 3)
        plt.imshow(cam_predict[:, :, ::-1])
        plt.title("CAM Heatmap(predict)")

        plt.subplot(1, 4, 4)
        plt.imshow(IBA_cam[:, :, ::-1])
        plt.title("IBA Heatmap")

        if if_write:
            plt.savefig('outputs/255/'+ str(classes_name) +'.png')

        if if_show:
            plt.show()

        plt.close()

def train():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    path = '../dataset/ImageNet'
    model = resnet50(pretrained=True).to(device)
    train_loader, val_loader, train_dataset, val_dataset = data_loader(path, batch_size=1)
    _, val_loader, _, _ = data_loader(path, batch_size=256)
    layer = 'layer4'

    grad_cam = GradCAM(model, layer, (224, 224), device, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    for i, data in enumerate(train_loader):

        img, label = data
        img, label = img.to(device), label.to(device)
        classes_name = val_dataset.imgs[i][0].split('/')[-1].split('.')[0]
        IBA_heatmap_shape, IBA_heatmap = per_sample_IBA(img, model, layer, label)
        grad_cam.forward(img, label, IBA_heatmap_shape, classes_name, show=False, write=False)
        loss = grad_cam.compute_IBA_GradCAM_grad(torch.from_numpy(IBA_heatmap).to(device), lamb=1, lr=0.00001)
        writer.add_scalar('loss', loss, global_step=i)

        if i % 1000 == 999:
            accuracy = test(model, val_loader, device)
            writer.add_scalar('accuracy', accuracy, global_step=i)
            torch.save(model.state_dict(), "saved_model/lr00001_number"+str(i)+"_accuracy"+str(accuracy)+".pth")

if __name__ == '__main__':
    train()
