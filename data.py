from torch.utils.data import Dataset
import torch
from PIL import Image

from torchvision.transforms import v2

class VLMDataset(Dataset):
    def __init__(self, dataframe, transforms = None):
        super().__init__()
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self,):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        main_image = Image.open(self.dataframe.iloc[index]['image_paths_study_id']).convert("RGB")
        ref_image = Image.open(self.dataframe.iloc[index]['image_paths_ref_id']).convert("RGB")
        text = self.dataframe.iloc[index]['question']
        suffix = self.dataframe.iloc[index]['answer']
        if self.transforms is not None:
            main_image = self.transforms(main_image)
            ref_image = self.transforms(ref_image)
        return {"main_image": main_image, "ref_image": ref_image, "texts": text, "suffix": suffix}
    
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((448, 448), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.ToDtype(torch.float32, scale=True),  # replaces ToTensor()
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((448, 448), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.ToDtype(torch.float32, scale=True),  # replaces ToTensor()
])

def collate_function(examples, processor):
    main_images = [example["main_image"] for example in examples]
    ref_images = [example["ref_image"] for example in examples]
    texts = [example["texts"] for example in examples]
    suffixes = [example["suffix"] for example in examples]
    inputs = processor(
        main_images=main_images,
        ref_images=ref_images,
        text=texts,
        suffix=suffixes,
    )
    return inputs

def test_collate_function(examples, processor):
    main_images = [example["main_image"] for example in examples]
    ref_images = [example["ref_image"] for example in examples]
    texts = [example["texts"] for example in examples]
    inputs = processor(
        main_images=main_images,
        ref_images=ref_images,
        text=texts,
    )
    return inputs


def get_dataloader(dataset, collate_fn, processor, train: bool, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=train,
        batch_size=batch_size,
        collate_fn=lambda examples: collate_fn(examples, processor),
        num_workers=4,
        pin_memory=True,
        drop_last=train,
    )