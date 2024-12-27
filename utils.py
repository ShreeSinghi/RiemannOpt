import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from PIL import Image
from torchvision import models, transforms
import timm
import gc
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from torch.utils.data import Dataset, Sampler

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name = "inceptionv3"):
    model = None
    if(name == "resnet50v2"):
        model = models.resnet50(weights='IMAGENET1K_V2').to(device)
    if(name == "resnet101v2"):
        model = models.resnet101(weights='IMAGENET1K_V2').to(device)
    if(name == "resnet152v2"):
        model = models.resnet152(weights='IMAGENET1K_V2').to(device)
    if(name == "inceptionv3"):
        model = models.inception_v3(weights='IMAGENET1K_V1', init_weights=False).to(device)
    if(name == "mobilenetv2"):
        model = models.mobilenet_v2(weights='IMAGENET1K_V2').to(device)
    if(name == "vgg16"):
        model = models.vgg16(weights='IMAGENET1K_V1').to(device)
    if(name == "vgg19"):
        model = models.vgg19(weights='IMAGENET1K_V1').to(device)
    if(name == "densenet121"):
        model = models.densenet121(weights='IMAGENET1K_V1').to(device)
    if(name == "densenet169"):
        model = models.densenet169(weights='IMAGENET1K_V1').to(device)
    if(name == "densenet201"):
        model = models.densenet201(weights='IMAGENET1K_V1').to(device)
    if(name == "xception"):
        model = timm.create_model('xception', pretrained=True).to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.zero_grad()
    return model


def load_transform(name = "inceptionv3"):
    if(name == "resnet50v2" or name == "resnet101v2" or name == "resnet152v2" or name == "mobilenetv2"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    if(name == "inceptionv3"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    if(name == "vgg16" or name == "vgg19" or name == "densenet121" or name == "densenet169" or name == "densenet201"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    if(name == "xception"):
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(342),
            transforms.CenterCrop(299),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    return transformer


@torch.no_grad()
def compute_outputs(model, input_tensor, indices, batchSize=64):
    original_shape = input_tensor.shape
    input_tensor = input_tensor.view(-1, *input_tensor.shape[-3:])

    outputs = torch.zeros(input_tensor.shape[0])
    corrects = torch.zeros(input_tensor.shape[0])

    for i in range(0, input_tensor.shape[0], batchSize):
        # Get the current batch
        batch = input_tensor[i:i+batchSize].to('cuda')
        current_batchSize = len(batch)
        with torch.no_grad():
            output = torch.nn.Softmax(dim=1)(model(batch))

        # Select the outputs at the given indices
        correct = output.argmax(1).cpu() == indices[i:i+current_batchSize]
        output = output[torch.arange(output.shape[0]), indices[i:i+current_batchSize]]

        # Compute the gradients of the selected outputs with respect to the input
        outputs[i:i+current_batchSize] = output.detach().to('cpu')
        corrects[i:i+current_batchSize] = correct

        del output, batch
        torch.cuda.empty_cache()
        gc.collect()
    return outputs.detach().view(*original_shape[:-3]), corrects.detach().view(*original_shape[:-3])

def compute_alphas(outputs, n_points, sample=5000):
    outputs = outputs / np.sum(outputs, axis=1).reshape(-1,1)

    outputs = outputs[:sample]

    def smooth(x, window_len=11):
        if window_len<3:
            return x

        s=np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        w=np.bartlett(window_len)

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[window_len//2:1-window_len//2]

    def linear_interpolation(data):
        log_data = np.log(data)
        log_data = smooth(log_data)
        data = np.exp(log_data)
        
        x = np.linspace(0, 1, len(data))
        f = interp1d(x, data, kind='cubic')
        return f

    def error(*delta_alphas):
        delta_alphass = delta_alphas[0] 
        delta_alphass /= np.sum(delta_alphass)

        alphas = np.cumsum(delta_alphass)  # THIS IS RIGHT RIEMANN SUM SINCE WE ARE NOT TAKING ALPHA=0
        alphas = np.clip(0, 1, alphas)     # for floating point errors
        error = np.sum(0.5 * np.square(delta_alphass) * f_grad(alphas))
        return error

    median_magnitudes = np.median(outputs, axis=0)
    f_grad = linear_interpolation(median_magnitudes)

    initial_guess = np.ones(n_points)

    res = minimize(error, x0=initial_guess, bounds=[(0, None) for _ in range(n_points)], method="Powell")

    delta_alphass = np.array(res.x)
    delta_alphass = delta_alphass/ np.sum(delta_alphass)
    alphas = np.cumsum(delta_alphass)
    alphas = np.clip(0, 1, alphas)
    alphas = np.append(0, alphas)

    return alphas

class CustomDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, batch):
        image_path, index = batch
        return self.transform(Image.open(image_path).convert("RGB")), index

class CustomBatchSampler(Sampler):
    def __init__(self, image_pathss, indexss):
        self.image_pathss = image_pathss
        self.indexss = indexss

    def __iter__(self):
        for image_paths, indexs in zip(self.image_pathss, indexss):
            yield zip(image_paths, indexs)

    def __len__(self):
        return len(self.img_path_list)