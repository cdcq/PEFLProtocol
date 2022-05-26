import copy
import os.path
from torchvision import utils
from ML_utils.get_data import transform_invert


def exec_poisoning(round_id: int, edge_id: int, trainer_count: int, poison_freq: int, start_round: int = 1) -> bool:
    # Decide whether execute poisoning or not
    if round_id >= start_round:
        if round_id % poison_freq == 0:
            if round_id % trainer_count == edge_id:
                return True

    return False


def get_poison_batch(batch, task, adversarial_index=-1, evaluation=False, poison_label_swap=1,
                     poison_frac_per_batch=0.5, save_flag=1):
    images, targets = batch
    poison_count = 0
    new_images = images
    new_targets = targets

    for index in range(0, len(images)):
        if evaluation:
            # 投毒测试时全部投毒
            new_targets[index] = poison_label_swap
            new_images[index] = add_pixel_pattern(images[index], adversarial_index)
            poison_count += 1

            # 保存图片
            if save_flag == 1:
                save_path = os.path.join("saved", "images", f"task_{task}", f"{index}.png")
                # before_unnormalize_image = new_images[index]
                image = transform_invert(img=new_images[index], task=task)
                utils.save_image(image, save_path)
                # read_image = io.read_image(save_path)
                # trans_mnist = transforms.Compose([transforms.ToPILImage(),
                #     transforms.ToTensor(),
                #     transforms.Normalize((0.1307,), (0.3081,))])
                # recovered_image = trans_mnist(read_image)
                # print(1)
        else:
            # 训练时只部分投毒
            if index < int(len(images) * poison_frac_per_batch):
                new_targets[index] = poison_label_swap
                new_images[index] = add_pixel_pattern(images[index], adversarial_index)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

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
