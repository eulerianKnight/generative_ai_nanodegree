from multiprocessing import cpu_count
from torchvision import datasets, transforms
import torch
import multiprocessing


def get_dataloader(
        root_path,
        image_size,
        batch_size,
        workers=multiprocessing.cpu_count()):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_train = datasets.StanfordCars(
        root=root_path,
        download=True,
        split='train',
        transform=transform
    )

    dataset_test = datasets.StanfordCars(
        root=root_path,
        download=True,
        split='test',
        transform=transform
    )

    dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])

    print(f"Using {workers} workers")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False,
    )

    return dataloader
