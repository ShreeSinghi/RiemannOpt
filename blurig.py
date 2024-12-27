import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import gc
from torchvision import transforms
import torch
import math

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BlurIG:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform
        self.blurrers = dict()

    def gaussian_blur(self, images, sigma):
        if sigma == 0:
            return images.cpu()

        size = float(min(2*torch.round(4*sigma)+1, 101))
        if sigma not in self.blurrers:
            self.blurrers[sigma] = transforms.GaussianBlur(size, float(sigma)).to(device)
        return self.blurrers[sigma].forward(images).cpu()
        
    def compute_outputs_gradients(self, input_tensor, indices, batchSize=64):
        gradients = []
        outputs = []

        for i in range(0, input_tensor.shape[0], batchSize):
            # Get the current batch
            batch = input_tensor[i:i+batchSize].to(device)
            batch.requires_grad_()

            current_batchSize = len(batch)
            output = torch.nn.Softmax(dim=1)(self.model(batch))[torch.arange(current_batchSize), indices[i:i+current_batchSize]]

            # Select the outputs at the given indices
            temp = output.sum()
            temp.backward()
            grad = batch.grad.detach().cpu()
            gradients.append(grad)
            batch.grad.zero_()

            outputs.append(output.detach().cpu())

            torch.cuda.empty_cache()
            gc.collect()

        return torch.cat(outputs), torch.cat(gradients, axis=0)
    
    def saliency(self, images, prediction_class, steps=20, steps_at=None, batch_size=32, max_sigma = 50, grad_step=0.01, sqrt=False, sigmas=None):
        
        if sigmas is None:
            if sqrt:
                sigmas = torch.Tensor([math.sqrt(float(i)*max_sigma/float(steps)) for i in range(0, steps+1)])
            else:
                sigmas = torch.Tensor([float(i)*max_sigma/float(steps) for i in range(0, steps+1)])

        step_vector_diff = sigmas[1:] - sigmas[:-1]

        if steps_at is None:
            steps_at = [steps]

        sequence           = torch.zeros((len(images), steps, *images.shape[1:]))
        gaussian_gradients = torch.zeros((len(images), steps, *images.shape[1:]))

        images = images.to(device)

        for i in range(steps):
            x_step = self.gaussian_blur(images, sigmas[i])
            gaussian_gradient = (self.gaussian_blur(images, sigmas[i]+grad_step)-x_step)/grad_step

            gc.collect()
            torch.cuda.empty_cache()

            gaussian_gradients[:, i] = gaussian_gradient
            sequence[:, i] = x_step

        images = images.cpu()

        target_class_idx = np.repeat(prediction_class, steps)
        outputs, gradients = self.compute_outputs_gradients(sequence.view(-1, *images.shape[1:]), target_class_idx, batchSize=batch_size)
        outputs = outputs.view(*sequence.shape[:2])
        gradients = gradients.view(*sequence.shape)

        out_ig = []
        for n in steps_at:
            gradients_copy = gradients[:,::steps//n]
            gaussian_gradients_copy = gaussian_gradients[:,::steps//n]
            step_vector_diff_copy = step_vector_diff.reshape(n, steps//n).sum(1).view(1,-1,1,1,1)

            out_ig.append((-step_vector_diff_copy*gaussian_gradients_copy*gradients_copy).sum((1,2)).numpy())

        outputs = outputs[:, -1] - outputs[:, 0]

        return out_ig, outputs.cpu().numpy()
