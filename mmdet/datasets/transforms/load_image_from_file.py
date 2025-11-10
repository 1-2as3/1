import os.path as osp
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(MMCV_LoadImageFromFile):
    """Extended LoadImageFromFile transform to set modality information.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        channel_order (str): The channel order of the output image array,
            'bgr' or 'rgb'. Defaults to 'bgr'.
        imdecode_backend (str): The image decoding backend. Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict containing the file path.

        Returns:
            dict: Updated result dict with additional modality information.
        """
        # Call parent class transform first
        results = super().transform(results)
        
        # Add modality information based on file path
        if 'img_path' in results:
            norm_path = results['img_path'].replace('\\', '/').lower()
            if 'visible' in norm_path:
                results['modality'] = 'visible'
            elif 'infrared' in norm_path:
                results['modality'] = 'infrared'
            else:
                results['modality'] = 'unknown'
        
        return results