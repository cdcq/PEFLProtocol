import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


def local_update(model, dataloader,
                 lr=0.01, momentum=0.0, local_eps=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    model.train()
    epoch_loss = []
    accumulated_grads_local = []
    for epoch in range(local_eps):
        batch_loss = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            probs = model(images)
            loss = loss_fn(probs, labels)
            loss.backward()
            optimizer.step()

            # 累积梯度记录
            if len(accumulated_grads_local) == 0:
                for para in model.parameters():
                    # 注意要从计算图分离出来并并保存到新的内存地址
                    accumulated_grads_local.append(para.grad.detach().clone())
            else:
                for level, para in enumerate(model.parameters()):
                    accumulated_grads_local[level] += para.grad

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 从GPU转移到CPU:协议处理时使用numpy和其他库处理
        accumulated_grads_local = [grad.cpu() for grad in accumulated_grads_local]
        return accumulated_grads_local, sum(epoch_loss) / len(epoch_loss)

def posioned_local_update(model, dataloader,
                 lr=0.01, momentum=0.0, local_eps=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # TODO
    return
