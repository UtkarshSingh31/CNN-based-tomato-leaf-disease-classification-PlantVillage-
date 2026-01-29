import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam[0].detach().cpu().numpy()
