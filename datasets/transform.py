import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def to_rgb(image):
    if image.shape[0] == 1:
        # Single channel diagram to 3-channel diagram
        return torch.cat([image, image, image], dim=0)
    else:
        return image


def get_transform(final_size=224,
                  initial_size=256,
                  channel=3,
                  crop="RS", 
                  scale=(0.08, 1.0), 
                  ratio=(3./4., 4./3.),
                  random_invert=False,
                  random_horizontal_flip=False,
                  random_erasing=False,
                  norm_mean=(0.485, 0.456, 0.406),
                  norm_std=(0.229, 0.224, 0.225)):

    T_list = [T.Resize(initial_size, InterpolationMode.BICUBIC)]
    
    if crop == "RS":
        T_list.append(T.RandomResizedCrop(final_size, scale=scale, ratio=ratio))
    elif crop == "R":
        T_list.append(T.RandomCrop(final_size))
    else:
        T_list.append(T.CenterCrop(final_size))

    if random_invert is not False:
        T_list.append(T.RandomInvert(p=random_invert))

    if random_horizontal_flip is not False:
        T_list.append(T.RandomHorizontalFlip(p=random_horizontal_flip))

    if channel == 1:
        T_list.extend([T.Grayscale(num_output_channels=1),
                       T.ToTensor()
                       ])
    elif channel == 3:
        T_list.extend([T.ToTensor(),
                       T.Lambda(to_rgb),
                       T.Normalize(mean=norm_mean, std=norm_std)
                       ])

    if random_erasing is not False:
        T_list.append(T.RandomErasing(p=random_erasing))
    
    return T.Compose(T_list)


def get_dataset_transform(dataset, source=True, img_size=224, channel=3, train=True):
    initial_size = 256
    random_invert = False
    random_horizontal_flip = False
    random_erasing = False
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    if train:
        crop = "RS"
    else:
        crop = "C"

    if dataset.lower() == "oracle241":
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)
        if train:
            crop = "R"
            random_horizontal_flip = 0.5
            if not source:
                random_invert = 0.5
    elif dataset.lower() == "digit":
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)
        initial_size = img_size
        crop = "C"
    elif dataset.lower() == "oracle50k_obi125":
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)
        if source:
            random_invert = 1
        if train:
            crop = "R"
            random_horizontal_flip = 0.5
            
    else:
        raise NameError(f"未定义名为{dataset}的数据集的transform格式")

    return get_transform(final_size=img_size,
                         initial_size=initial_size,
                         channel=channel,
                         crop=crop,
                         random_invert=random_invert,
                         random_horizontal_flip=random_horizontal_flip,
                         random_erasing=random_erasing,
                         norm_mean=norm_mean,
                         norm_std=norm_std
                         )

