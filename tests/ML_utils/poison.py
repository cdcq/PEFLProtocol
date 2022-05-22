import copy

def exec_poisoning(round_id: int, edge_id: int, trainer_count: int, poison_freq: int, start_round: int = 1) -> bool:
    # Decide whether execute poisoning or not
    if round_id >= start_round:
        if round_id % poison_freq == 0:
            if round_id % trainer_count == edge_id:
                return True

    return False


def transform_invert_and_save(img, transform_train):
    """
    将data 进行反transfrom操作,并保存图像
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img.dtype, device=img.device)
        std = torch.tensor(norm_transform[0].std, dtype=img.dtype, device=img.device)
        img.mul_(std[:, None, None]).add_(mean[:, None, None])

    img = img.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img = np.array(img) * 255

    if img.shape[2] == 3:
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
    elif img.shape[2] == 1:
        img = Image.fromarray(img.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img

def get_poison_batch(bptt, device, adversarial_index=-1, evaluation=False, poison_label_swap=1,
                     poison_frac_per_batch=0.5):
    images, targets = bptt
    poison_count = 0
    new_images = images
    new_targets = targets

    for index in range(0, len(images)):
        if evaluation:
            # 投毒测试时全部投毒
            new_targets[index] = poison_label_swap
            new_images[index] = add_pixel_pattern(images[index], adversarial_index)
            poison_count += 1

        else:
            # 训练时只部分投毒
            if index < int(len(images) * poison_frac_per_batch):
                new_targets[index] = poison_label_swap
                new_images[index] = add_pixel_pattern(images[index], adversarial_index)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

    # new_images = new_images.to(device)
    # new_targets = new_targets.to(device).long()  # self.long() is equivalent to self.to(torch.int64)
    # if evaluation:
    #     new_images.requires_grad_(False)
    #     new_targets.requires_grad_(False)
    return new_images, new_targets, poison_count


def add_pixel_pattern(ori_image, adversarial_index):
    poison_patterns_template = [
        [[0, 0], [0, 1], [0, 2], [0, 3]],
        [[0, 6], [0, 7], [0, 8], [0, 9]],
        [[3, 0], [3, 1], [3, 2], [3, 3]],
        [[3, 6], [3, 7], [3, 8], [3, 9]],
    ]

    image = copy.deepcopy(ori_image)
    if adversarial_index != -1:
        poison_patterns = poison_patterns_template[adversarial_index % 4]
    else:
        poison_patterns = []
        for pattern in poison_patterns_template:
            poison_patterns.extend(pattern)

    channel_num = image.shape[0]  # [C, H, W]
    for pattern_idx in range(len(poison_patterns)):
        pos = poison_patterns[pattern_idx]
        for channel_idx in range(channel_num):
            image[channel_idx][pos[0]][pos[1]] = 1  # 也许可以试试广播机制

    return image
