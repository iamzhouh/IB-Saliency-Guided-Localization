import torch
import numpy as np
from ..utils.misc import to_np_img
from ..utils.transforms import Compose, CropPercentile, SetInterval
from ..utils.misc import to_np
from ..attribution.base import AttributionMethod


class ModifiedBackpropMethod(AttributionMethod):

    def __init__(self, model):
        self.hooks = []
        self.model = model

    def _hook_forward(self, module):
        return None

    def _hook_backward(self, module):
        return None

    def _transform_gradient(self, gradient: np.ndarray):
        raise NotImplementedError

    def __restore_model(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __prepare_model(self):

        def hook_layer(module):
            forward = self._hook_forward(module)
            if forward is not None:
                self.hooks.append(module.register_forward_hook(forward))
            backward = self._hook_backward(module)
            if forward is not None:
                self.hooks.append(module.register_backward_hook(backward))

        self.model.apply(hook_layer)

    def heatmap(self, input_t, target_t):

        self.model.eval()
        self.__prepare_model()
        grad_t = self._calc_gradient(input_t=input_t, target_t=target_t).detach()
        self.__restore_model()

        assert isinstance(grad_t, torch.Tensor)
        assert len(grad_t.shape) == 4
        assert grad_t.shape == tuple(input_t.shape), f"Backprop shape mismatch: {grad_t.shape} != {input_t.shape}"

        grad = to_np_img(grad_t)
        heatmap = self._transform_gradient(grad)

        return heatmap

    def _calc_gradient(self, input_t: torch.Tensor, target_t: torch.Tensor):

        self.model.zero_grad()  
        img_var = torch.autograd.Variable(input_t, requires_grad=True) 
        logits = self.model(img_var) 

        target_idx = target_t.item() if isinstance(target_t, torch.Tensor) else target_t
        grad_eval_point = torch.zeros(device=input_t.device, size=logits.shape)
        grad_eval_point[0][target_idx] = 1.0 
        logits.backward(gradient=grad_eval_point)

        return img_var.grad


class Gradient(ModifiedBackpropMethod):
    def _transform_gradient(self, gradient: np.ndarray):
        return Compose(
            lambda x: x.mean(axis=-1),
            CropPercentile(0.5, 99.5),
            SetInterval(1),
        )(gradient)


class Saliency(ModifiedBackpropMethod):
    def _transform_gradient(self, gradient: np.ndarray):
        return Compose(
            lambda x: np.abs(x).max(axis=-1),
            CropPercentile(0.5, 99.5),
            SetInterval(1),
        )(gradient)


class GradientTimesInput(ModifiedBackpropMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_input = None

    """ Gradient * Input """
    def heatmap(self, input_t, target_t):
        # Remember input for future usage
        self.last_input = to_np(input_t)
        return super().heatmap(input_t, target_t)

    def _transform_gradient(self, gradient: np.ndarray):
        return Compose(
            lambda x: x.mean(axis=-1),
            CropPercentile(0.5, 99.5),
            SetInterval(1),
        )(self.last_input * gradient)

