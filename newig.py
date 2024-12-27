import numpy as np
import gc
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import gc
from utils import compute_alphas
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NewIG:
    def __init__(self, model, transform, n_points):
        self.model = model
        self.transform = transform

        # dimensions dont matter since it will be resized anyway
        BLACK  = Image.fromarray(np.zeros((299, 299, 3), dtype=np.uint8))
        self.BLACK = transform(BLACK).unsqueeze(0)# adds extra first dimension 
        self.alphas= compute_alphas(np.load('magnitude_ig.npy'), n_points-1)
        print(self.alphas)

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

    def straight_path_images(self, images):        
        x_diff = images - self.BLACK
        path_images = []
        
        for alpha in self.alphas:
            x_step = self.BLACK + alpha * x_diff
            path_images.append(x_step)
        
        path_images = torch.stack(path_images).transpose(0, 1)

        # returns x sequence
        return path_images

    def saliency(self, images, class_idxs, n_steps, compute_at=None, compute_batchSize=128):
        if compute_at is None:
            compute_at = [n_steps]

        sequence = self.straight_path_images(images)

        image_shape = sequence.shape[2:]
        batchSize = sequence.shape[0]

        classes = np.repeat(class_idxs, n_steps)
        
        reshaped_sequence = sequence.reshape((batchSize*n_steps,)+image_shape).detach()

        output, gradients = self.compute_outputs_gradients(reshaped_sequence, classes, batchSize=compute_batchSize)
        gradients = gradients.view((batchSize, n_steps)+image_shape)
        output = output.view((batchSize, n_steps))
        sequence = sequence.detach()

        out_ig = (gradients[:,1:] * (sequence[:, 1:]-sequence[:, :-1])).sum(1).sum(1).numpy()

        del reshaped_sequence, gradients, sequence
        torch.cuda.empty_cache()
        gc.collect()

        outputs = output.cpu().numpy()
        outputs = outputs[:, -1] - outputs[:, 0]

        return [out_ig], outputs

