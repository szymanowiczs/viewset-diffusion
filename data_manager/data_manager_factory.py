from . import QuadrupleDataset
from . import SRNDataset
from . import CO3DDataset

def get_data_manager(cfg, split='train',
                     convert_to_double_conditioning=None,
                     convert_to_single_conditioning=None,
                     **kwargs):

    assert split in ['train', 'val', 'test'], "Invalid split"

    if convert_to_double_conditioning is None:
        if cfg.data.two_training_imgs_per_example:
            convert_to_double_conditioning=True
        else:
            convert_to_double_conditioning=False

    if convert_to_single_conditioning is None:
        if cfg.data.one_training_img_per_example:
            convert_to_single_conditioning=True
        else:
            convert_to_single_conditioning=False

    if cfg.data.dataset_type == "skins":
        if split == 'train':
            dataset_name = "training"
        elif split == 'val':
            dataset_name = "ID_skin_OOD_pose_OOD_cond"
        elif split == 'test':
            dataset_name = "testing"
        dataset = QuadrupleDataset(cfg,
                            convert_to_double_conditioning=convert_to_double_conditioning,
                            convert_to_single_conditioning=convert_to_single_conditioning,
                            dataset_name=dataset_name,
                            **kwargs)
    elif cfg.data.dataset_type == "co3d":
        dataset = CO3DDataset(cfg,
                            convert_to_double_conditioning=convert_to_double_conditioning,
                            convert_to_single_conditioning=convert_to_single_conditioning,
                            dataset_name=split,
                            **kwargs)
    elif cfg.data.dataset_type == "srn":
        dataset = SRNDataset(cfg,
                            convert_to_single_conditioning,
                            convert_to_double_conditioning,
                            dataset_name=split,
                            **kwargs)
        
    return dataset