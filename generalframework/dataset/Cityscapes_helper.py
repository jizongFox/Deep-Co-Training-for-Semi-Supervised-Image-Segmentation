
citylist = [
    'aachen',
    'bremen',
    'darmstadt',
    'erfurt',
    'hanover',
    'krefeld',
    'strasbourg',
    'tubingen',
    'weimar',
    'bochum',
    'cologne',
    'dusseldorf',
    'hamburg',
    'jena',
    'monchengladbach',
    'stuttgart',
    'ulm',
    'zurich']


def get_cityscapes_dataloaders(dataset_dict: dict, dataloader_dict: dict):
    # Setup Augmentations
    augmentations = dataset_dict.get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)
    data_path = dataset_dict["root_dir"]
    lab_set = CityscapesDataset(data_path, is_transform=True, mode='train', image_size=(dataset_dict["image_size"]),
                                augmentation=data_aug)
    lab_loader = DataLoader(lab_set, **dataloader_dict)
    val_set = CityscapesDataset(data_path, is_transform=True, mode='val', image_size=dataset_dict["image_size"])
    val_loader = DataLoader(val_set, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})

    test_set = CityscapesDataset(data_path, is_transform=True, mode='test', image_size=dataset_dict["image_size"])
    test_loader = DataLoader(test_set, **{**dataloader_dict, **{'shuffle': False, 'batch_size': 1}})
    return {'train': lab_loader, 'val': val_loader, 'test': test_loader}


def extract_cities(dataloader: DataLoader, city_names: List[str]):
    '''
    this is an extractor for cityscapes dataset for selection different cities
    :param dataloader:
    :param patient_ids:
    :return: a new dataloader
    '''
    if city_names == None or city_names.__len__() == 0:
        ## return a deep copy of the dataloader
        return dcopy(dataloader)

    assert isinstance(city_names, list)
    bpattern = lambda d: str(d)
    patterns = re.compile('|'.join([bpattern(name) for name in city_names]))
    files: Dict[str, List[str]] = dcopy(dataloader.dataset.files['train'])
    new_files = [file for file in files if re.search(patterns, file)]

    new_dataloader = dcopy(dataloader)
    new_dataloader.dataset.files['train'] = new_files
    return new_dataloader


def extract_dataset_by_p(dataloader: DataLoader, p: float = 0.5, random_state=1):
    np.random.seed(random_state)
    labeled_dataloader = dcopy(dataloader)
    unlabeled_dataloader = dcopy(dataloader)
    files = labeled_dataloader.dataset.files['train']
    labeled_files = np.random.choice(files, int(len(files) * p), replace=False).tolist()
    labeled_dataloader.dataset.files['train'] = labeled_files
    unlabeled_files = [x for x in files if x not in labeled_files]
    unlabeled_dataloader.dataset.files['train'] = unlabeled_files
    assert unlabeled_files.__len__() + len(labeled_files) == len(files)
    return labeled_dataloader, unlabeled_dataloader
