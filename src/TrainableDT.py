import torch
import torch.nn.functional as F
from transformers import DecisionTransformerModel


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)

        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1,
                                            act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1,
                                                act_dim)[attention_mask.reshape(-1) > 0]
        probs = F.softmax(action_preds, dim=1)

        # cross entropy loss
        loss = F.cross_entropy(probs, action_targets)

        return {"loss": loss, "logits": action_preds}

    def original_forward(self, **kwargs):
        output = super().forward(**kwargs)
        action_preds = output[1]
        act_dim = kwargs["actions"].shape[2]
        action_pred = action_preds.reshape(-1, act_dim)[-1]  # next action

        probs = F.softmax(action_pred, dim=-1)
        action_pred = torch.nn.functional.one_hot(
            torch.argmax(probs, dim=-1), num_classes=act_dim
        )

        return action_pred