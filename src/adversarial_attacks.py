from dataclasses import dataclass, field

from torch.nn import Module
from torch import Tensor, no_grad, min, max, clamp


@dataclass
class ProjectedGradientDescent:
    iters: int
    alpha: float
    loss: Module
    model: Module
    epsilon: float
    input_min: Tensor|float = field(default=None)
    input_max: Tensor|float = field(default=None)


    def generate_attacks(self, inputs: Tensor, labels: Tensor) -> Tensor:
        original_model_training_state = self.model.training
        print("original_model_training_state:", original_model_training_state)
        self.model.eval()
        print("creating adv attacks")
        adv_inputs = inputs.clone().detach()
        # If not specified, take the min and max of the batch instead of the entire dataset
        input_min = inputs.min() if self.input_min is None else self.input_min
        input_max = inputs.max() if self.input_max is None else self.input_max
        # Iteratively compute the adversarial inputs
        for _ in range(self.iters):
            # Compute the loss gradients w.r.t adversarial inputs
            # Ensure that the grad w.r.t the adv_inputs will be computed
            self.model.zero_grad()
            adv_inputs.requires_grad = True 
            outputs = self.model(adv_inputs)
            loss_values = self.loss(outputs, labels)
            loss_values.backward()
            # Move the adversarial inputs toward higher loss
            with no_grad():
                adv_inputs += adv_inputs.grad.sign() * self.alpha
                delta = clamp(adv_inputs - inputs, -self.epsilon, self.epsilon)
                adv_inputs = clamp(inputs + delta, input_min, input_max)
        self.model.train(original_model_training_state)

        return adv_inputs