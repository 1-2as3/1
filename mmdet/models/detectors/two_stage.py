# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor or dict): Image tensor with shape (N, C, H ,W).
                For cross-modal learning, can be a dict with keys 'visible' and 'infrared'.

        Returns:
            tuple[Tensor] or dict: Multi-level features that may have
            different resolutions. Returns dict with 'vis' and 'ir' keys if input is dict.
        """
        # NaN检查：输入数据完整性验证
        if isinstance(batch_inputs, dict):
            for key in ['visible', 'infrared']:
                if key in batch_inputs and torch.isnan(batch_inputs[key]).any():
                    raise ValueError(f"[ERROR] Input '{key}' contains NaN values!")
        elif isinstance(batch_inputs, torch.Tensor):
            if torch.isnan(batch_inputs).any():
                raise ValueError("[ERROR] Input batch_inputs contains NaN values!")
        
        # Handle paired modality inputs for cross-modal learning
        if isinstance(batch_inputs, dict) and 'visible' in batch_inputs and 'infrared' in batch_inputs:
            # Process both modalities through backbone
            vis_x = self.backbone(batch_inputs['visible'])
            ir_x = self.backbone(batch_inputs['infrared'])
            
            # Apply neck if available
            if self.with_neck:
                vis_x = self.neck(vis_x)
                ir_x = self.neck(ir_x)
            
            # Return paired features
            return {'vis': vis_x, 'ir': ir_x}
        
        # Standard single modality processing
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def _validate_inputs_smoke_test(self, batch_inputs, batch_data_samples):
        """One-time smoke test on first iteration to catch type/shape issues early.
        
        Validates:
        - batch_inputs dtype (float32), device consistency, no NaN/Inf
        - For paired modality: visible and infrared shapes match
        - data_samples have required meta fields (pad_shape, img_shape)
        - gt_bboxes and labels are float32 and contiguous
        """
        import torch
        print("[Smoke Test] Validating first iteration inputs...")
        
        if isinstance(batch_inputs, dict):
            # Paired modality
            assert 'visible' in batch_inputs and 'infrared' in batch_inputs, \
                "Paired mode requires 'visible' and 'infrared' keys in batch_inputs"
            
            vis = batch_inputs['visible']
            ir = batch_inputs['infrared']
            
            # Check dtype
            assert vis.dtype == torch.float32, f"visible dtype={vis.dtype}, expected float32"
            assert ir.dtype == torch.float32, f"infrared dtype={ir.dtype}, expected float32"
            
            # Check device
            assert vis.device == ir.device, f"Device mismatch: vis={vis.device}, ir={ir.device}"
            
            # Check shapes match
            assert vis.shape == ir.shape, f"Shape mismatch: vis={vis.shape}, ir={ir.shape}"
            
            # Check for NaN/Inf
            assert not torch.isnan(vis).any(), "visible contains NaN"
            assert not torch.isinf(vis).any(), "visible contains Inf"
            assert not torch.isnan(ir).any(), "infrared contains NaN"
            assert not torch.isinf(ir).any(), "infrared contains Inf"
            
            print(f"  ✓ Paired inputs: vis/ir shape={vis.shape}, dtype={vis.dtype}, device={vis.device}")
        else:
            # Single modality
            assert isinstance(batch_inputs, torch.Tensor), "batch_inputs must be Tensor or dict"
            assert batch_inputs.dtype == torch.float32, f"batch_inputs dtype={batch_inputs.dtype}, expected float32"
            assert not torch.isnan(batch_inputs).any(), "batch_inputs contains NaN"
            assert not torch.isinf(batch_inputs).any(), "batch_inputs contains Inf"
            print(f"  ✓ Single modality: shape={batch_inputs.shape}, dtype={batch_inputs.dtype}")
        
        # Check data_samples meta info
        if batch_data_samples:
            sample = batch_data_samples[0]
            assert hasattr(sample, 'metainfo'), "data_sample missing metainfo"
            assert 'pad_shape' in sample.metainfo, "metainfo missing 'pad_shape'"
            assert 'img_shape' in sample.metainfo, "metainfo missing 'img_shape'"
            
            # Check gt_instances
            if hasattr(sample, 'gt_instances'):
                gt = sample.gt_instances
                if hasattr(gt, 'bboxes') and len(gt.bboxes) > 0:
                    # bboxes might be HorizontalBoxes or Tensor; check underlying tensor
                    bbox_tensor = gt.bboxes.tensor if hasattr(gt.bboxes, 'tensor') else gt.bboxes
                    assert bbox_tensor.dtype == torch.float32, f"gt_bboxes dtype={bbox_tensor.dtype}, expected float32"
                    assert bbox_tensor.is_contiguous(), "gt_bboxes tensor not contiguous"
                if hasattr(gt, 'labels') and len(gt.labels) > 0:
                    assert gt.labels.dtype in [torch.int64, torch.long], f"gt_labels dtype={gt.labels.dtype}"
            
            print(f"  ✓ data_samples[0]: pad_shape={sample.metainfo['pad_shape']}, "
                  f"img_shape={sample.metainfo['img_shape']}")
        
        print("[Smoke Test] ✅ All checks passed!\n")

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor or dict): Input images of shape (N, C, H, W).
                For paired modality, this is a dict with 'visible' and 'infrared' keys,
                both already preprocessed (float32, normalized, padded) by PairedDetDataPreprocessor.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        import torch
        
        # Smoke test: validate inputs on first iteration (one-time check)
        if not hasattr(self, '_first_iter_validated'):
            self._validate_inputs_smoke_test(batch_inputs, batch_data_samples)
            self._first_iter_validated = True
        
        # Extract features (handles both single-modality Tensor and paired dict)
        x = self.extract_feat(batch_inputs)
        
        # For RPN forward, use visible features if paired modality
        x_for_rpn = x['vis'] if isinstance(x, dict) and 'vis' in x else x

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x_for_rpn, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        
        # 检查空样本情况：如果所有loss都是0或NaN，返回安全的零梯度
        if not roi_losses or all(v.item() == 0 or torch.isnan(v).any() for v in roi_losses.values()):
            print("[WARN] Empty or invalid RoI losses detected, returning zero losses")
            zero_device = batch_inputs.device if isinstance(batch_inputs, torch.Tensor) else batch_inputs['visible'].device
            roi_losses = {k: torch.tensor(0.0, device=zero_device, requires_grad=True)
                          for k in ['loss_cls', 'loss_bbox', 'loss_macl', 'loss_domain']}
        
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
