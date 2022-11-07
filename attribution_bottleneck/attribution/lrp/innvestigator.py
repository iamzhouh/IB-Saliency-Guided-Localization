import torch
import warnings
import numpy as np

from attribution_bottleneck.attribution.lrp.inverter_util import RelevancePropagator
from attribution_bottleneck.attribution.lrp.utils import pprint, Flatten


class InnvestigateModel(torch.nn.Module):
    def __init__(self, the_model, lrp_exponent=1, beta=.5, epsilon=1e-6,
                 method="e-rule", device=torch.device('cpu')):
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        self.device = device
        self.prediction = None
        self.r_values_per_layer = None
        self.only_max_score = None
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            beta=beta, method=method, epsilon=epsilon,
                                            device=self.device)
        self.hook_handles = []

        self.register_hooks(self.model)
        if method == "b-rule" and float(beta) in (-1., 0):
            which = "positive" if beta == -1 else "negative"
            which_opp = "negative" if beta == -1 else "positive"
            warnings.warn("WARNING: With the chosen beta value, "
                  "only " + which + " contributions "
                  "will be taken into account.\nHence, "
                  "if in any layer only " + which_opp +
                  " contributions exist, the "
                  "overall relevance will not be conserved.\n")

    def to(self, dev):
        self.device = dev
        self.inverter.device = dev
        super().to(dev)

    def register_hooks(self, parent_module):
        for mod in parent_module.children():
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            self.hook_handles.append(mod.register_forward_hook(
                self.inverter.get_layer_fwd_hook(mod)))
            if isinstance(mod, torch.nn.ReLU):
                self.hook_handles.append(mod.register_backward_hook(self.relu_hook_function))

    def __del__(self):
        for hook in self.hook_handles:
            hook.remove()

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):
        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor):
        self.inverter.reset_module_list()
        self.prediction = self.model(in_tensor)
        return self.prediction

    def get_r_values_per_layer(self):
        if self.r_values_per_layer is None:
            pprint("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
        return self.r_values_per_layer

    def innvestigate(self, in_tensor=None, rel_for_class=None):
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            if in_tensor is not None:
                self.evaluate(in_tensor)

            if rel_for_class is None:
                org_shape = self.prediction.size()

                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)

            else:
                org_shape = self.prediction.size()
                self.prediction = self.prediction.view(org_shape[0], -1)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)

            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            r_values_per_layer = [relevance]
            for layer in rev_model:
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cpu())

            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return self.prediction, r_values_per_layer[-1]

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        return self.model.extra_repr()
