import torch

from torch import Tensor
from typing import List, Optional


class ClassicSGD(torch.optim.SGD):
    def __init__(
        self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False,
    ):
        super(ClassicSGD, self).__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Overwrite step of Pytorch SGD to change velocity and param update formula

        Args:
            closure (callable, optional): Evaluates model and returns Loss. Defaults to None.

        Returns:
            loss: loss from closure. Defaults to None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]

            try:
                maximize = group["maximize"]
            except:
                maximize = False  # set maximize to false if not found

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool
):
    """Extracted Functional api from torch library:
    https://github.com/pytorch/pytorch/blob/5fdcc20d8d96a6b42387f57c2ce331516ad94228/torch/optim/_functional.py#L156
    Modified to handle learning rate differently
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        d_p = d_p.mul_(lr)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                # original: buf * momentum + g
                # new: buf * momentum + g * lr
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        # original: p = p - lr * v
        # new: p = p - v
        alpha = 1 if maximize else -1
        param.add_(d_p, alpha=alpha)

