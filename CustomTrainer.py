import torch.nn.functional as F

#test default
def nll_loss(labels, target):
    #print(logits)
    print(labels)
    return F.nll_loss(labels, target)
#test focal_loss
def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    # Calculate standard cross-entropy loss first.
    loss = torch.nn.CrossEntropyLoss()

    ce_loss = F.cross_entropy(logits, labels, reduction='none')    
    # Get softmax probabilities.
    print(ce_loss)
    pt = torch.exp(-ce_loss)
    # Compute focal loss.
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    #input = torch.randn(3, 5, requires_grad=True)
    #target = torch.randn(3, 5).softmax(dim=1)
    #output = loss(input, target)
    #output.backward()

    return focal_loss.mean()


from transformers import Trainer
from torch import nn

class CustomTrainer(Trainer):
    def compute_losss(self, model, inputs, return_outputs, num_items_in_batch):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        reduction = "mean" if num_items_in_batch is not None else "sum"
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device, reduction=reduction))
        pt = torch.exp(-loss_fct)
        gamma=2.0; alpha=0.25
        focal_loss = alpha * (1 - pt) ** gamma * loss_fct
        #loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        loss = focal_loss
        if num_items_in_batch is not None:
            loss = loss / num_items_in_batch
        return (loss, outputs) if return_outputs else loss    
        

trainer = CustomTrainer(
#trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],#.append(tokenized_data["test"]),
    data_collator=data_collator,
    tokenizer=tokenizer,    
    compute_metrics = compute_metrics2
)

for cb in trainer.callback_handler.callbacks:
    if isinstance(cb, transformers.integrations.MLflowCallback):
        trainer.callback_handler.remove_callback(cb)
        