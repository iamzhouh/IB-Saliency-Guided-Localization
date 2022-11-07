from attribution_bottleneck.bottleneck.estimator import ReluEstimator
from attribution_bottleneck.attribution.per_sample_bottleneck import PerSampleBottleneckReader

def per_sample_IBA(img, model, layer, ex_target):
    ex_image = img
    ex_input = ex_image

    estim = ReluEstimator(eval('model.' + layer))
    estim.load("weights/estimator_resnet50_"+layer[-1]+".torch")

    reader = PerSampleBottleneckReader(model, estim, progbar=True)

    heatmap_shape, heatmap = reader.heatmap(ex_input, ex_target)
    result = model(ex_input).argmax().item()
    print("Result class is", result,",Target class is", ex_target.item(), ",which is", (ex_target==result).item(),end='.')

    return heatmap_shape, heatmap