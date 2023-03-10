import pickle


class BaseLoader:

    def __init__(self, data_path, pkl_path) -> None:
        super().__init__()

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.nusc_infos = data['infos']
        self.data_path = data_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)
    
    def get_data_info(self, info):
        """Get data info according to the given index.
        Please refer to the pickle files to generate required fields.

        Args:
            info (dict): a slice from self.nusc_infos.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:
                - filenames: image or point or voxel
                - transformation matrices for coordinate change
                - other meta data, e.g. sample token, timestamps.
        """
        pass
