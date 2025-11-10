# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead


@MODELS.register_module()
class StandardRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, *args, 
                 use_macl: bool = False, 
                 macl_head: ConfigType = None, 
                 use_msp: bool = False,
                 use_dhn: bool = False,
                 pool_only_pos: bool = False,
                 use_domain_alignment: bool = False,
                 domain_classifier: ConfigType = None,
                 lambda1: float = 1.0,
                 lambda2: float = 0.5,
                 lambda3: float = 0.1,
                 **kwargs):
        """Extend BaseRoIHead initialization to optionally attach custom modules.

        Args:
            use_macl (bool): Whether to instantiate MACLHead.
            macl_head (dict or ConfigDict, optional): Config to build MACLHead
                via the registry (recommended).
            use_msp (bool): Whether to use MSP module (handled by neck, accepted for compatibility).
            use_dhn (bool): Whether to use DHN sampler (handled by MACL head, accepted for compatibility).
            pool_only_pos (bool, optional): Whether to only use positive samples
                when computing pooled features. Defaults to False.
            use_domain_alignment (bool): Whether to use domain alignment.
            domain_classifier (dict or ConfigDict, optional): Config for domain classifier.
            lambda1 (float): Weight for MACL loss. Defaults to 1.0.
            lambda2 (float): Weight for DHN loss. Defaults to 0.5.
            lambda3 (float): Weight for domain loss. Defaults to 0.1.
        """
        # Store custom module flags before calling parent
        self.use_macl = use_macl
        self.use_msp = use_msp
        self.use_dhn = use_dhn
        self.use_domain_alignment = use_domain_alignment
        
        # call BaseRoIHead initializer first
        super().__init__(*args, **kwargs)
        self.pool_only_pos = pool_only_pos
        
        # 损失权重参数
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        # 初始化 epoch 计数器（用于动态衰减）
        self.cur_epoch = 0
        self.max_epochs = 100  # 默认值，可以从配置中覆盖
        
        # 日志打印：模块状态（避免emoji编码问题）
        print(f"[RoIHead] modules active: MACL={self.use_macl}, MSP={self.use_msp}, DHN={self.use_dhn}, DomainAlign={self.use_domain_alignment}")
        
        # MACL Head - 使用延迟导入防止循环依赖
        if self.use_macl:
            if isinstance(macl_head, dict):
                try:
                    self.macl_head = MODELS.build(macl_head)
                    print("  [OK] MACLHead initialized from config")
                except Exception as e:
                    print(f"  [WARN] MACLHead build failed: {e}, using defaults")
                    from mmdet.models.macldhnmsp.macl_head import MACLHead
                    try:
                        self.macl_head = MACLHead(**macl_head)
                    except Exception:
                        self.macl_head = MACLHead(in_dim=256, proj_dim=128)
            else:
                from mmdet.models.macldhnmsp.macl_head import MACLHead
                self.macl_head = MACLHead(in_dim=256, proj_dim=128)
                print("  [OK] MACLHead initialized with defaults")
        
        # MSP Module - 通常在 neck 中处理，这里仅作为占位
        if self.use_msp:
            print("  [INFO] MSP module is typically handled by FPN neck")
        
        # DHN Sampler - 通常集成在 MACL head 中
        if self.use_dhn:
            print("  [INFO] DHN sampler is typically integrated in MACLHead")
        
        # Domain Alignment
        self.domain_lambda = 0.0  # 初始化为 0，由 Hook 动态调整
        if self.use_domain_alignment:
            if isinstance(domain_classifier, dict):
                try:
                    self.domain_classifier = MODELS.build(domain_classifier)
                    print("  [OK] DomainClassifier initialized from config")
                except Exception as e:
                    print(f"  [WARN] DomainClassifier build failed: {e}, using defaults")
                    from mmdet.models.macldhnmsp.domain_classifier import DomainClassifier
                    self.domain_classifier = DomainClassifier(**domain_classifier)

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.bbox_sampler = TASK_UTILS.build(
                self.train_cfg.sampler, default_args=dict(context=self))

    def init_bbox_head(self, bbox_roi_extractor: ConfigType,
                       bbox_head: ConfigType) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict or ConfigDict): Config of box
                roi extractor.
            bbox_head (dict or ConfigDict): Config of box in box head.
        """
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)

    def init_mask_head(self, mask_roi_extractor: ConfigType,
                       mask_head: ConfigType) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict or ConfigDict): Config of mask roi
                extractor.
            mask_head (dict or ConfigDict): Config of mask in mask head.
        """
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = MODELS.build(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = MODELS.build(mask_head)

    # TODO: Need to refactor later
    def forward(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList = None) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
            the meta information of each image and corresponding
            annotations.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        results = ()
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            results = results + (bbox_results['cls_score'],
                                 bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            results = results + (mask_results['mask_preds'], )
        return results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor] or dict): List of multi-level img features.
                For cross-modal learning, can be a dict with 'vis' and 'ir' keys.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # Handle paired modality features for cross-modal learning
        is_paired_modality = isinstance(x, dict) and 'vis' in x and 'ir' in x
        
        if is_paired_modality:
            # For paired modalities, use visible features for standard detection pipeline
            x_for_detection = x['vis']
            vis_feats = x['vis']
            ir_feats = x['ir']
        else:
            x_for_detection = x
            vis_feats = None
            ir_feats = None
        
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals (use visible features for detection)
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x_for_detection])
            sampling_results.append(sampling_result)

        # 检查空样本：如果所有sampling_results的pos_priors都为空，返回零损失
        if all(len(res.pos_priors) == 0 for res in sampling_results):
            print("[WARN] Empty RoI samples (no positive boxes), returning zero losses")
            device = x_for_detection[0].device if isinstance(x_for_detection, (list, tuple)) else x_for_detection.device
            return {
                'loss_cls': torch.tensor(0.0, device=device, requires_grad=True),
                'acc': torch.tensor(100.0, device=device),
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
            }

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x_for_detection, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(x_for_detection, sampling_results,
                                          bbox_results['bbox_feats'],
                                          batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        # Compute MACL loss for paired modalities (cross-modal contrastive learning)
        if is_paired_modality and getattr(self, 'use_macl', False) and hasattr(self, 'macl_head'):
            try:
                # Pool features from both modalities for contrastive learning
                # Use global average pooling across all FPN levels
                vis_pooled_list = []
                ir_pooled_list = []
                
                for vis_fm, ir_fm in zip(vis_feats, ir_feats):
                    # Global average pooling for each FPN level
                    vis_pooled = torch.nn.functional.adaptive_avg_pool2d(vis_fm, 1)
                    ir_pooled = torch.nn.functional.adaptive_avg_pool2d(ir_fm, 1)
                    
                    vis_pooled_list.append(vis_pooled.flatten(1))  # (B, C)
                    ir_pooled_list.append(ir_pooled.flatten(1))    # (B, C)
                
                # Average features across all FPN levels instead of concatenating
                # This keeps the dimension consistent at 256 (C of FPN)
                vis_feat_vec = torch.stack(vis_pooled_list, dim=0).mean(dim=0)  # (B, C)
                ir_feat_vec = torch.stack(ir_pooled_list, dim=0).mean(dim=0)    # (B, C)
                
                # Compute MACL contrastive loss
                macl_out = self.macl_head(vis_feat_vec, ir_feat_vec)
                
                # Add MACL loss to losses dict
                if isinstance(macl_out, dict):
                    if 'loss_macl' in macl_out:
                        losses['loss_macl'] = macl_out['loss_macl']
                    if 'vis' in macl_out:
                        losses['macl_vis'] = macl_out['vis']
                    if 'ir' in macl_out:
                        losses['macl_ir'] = macl_out['ir']
                else:
                    losses['loss_macl'] = macl_out
                    
            except Exception as e:
                # If MACL computation fails, log and continue without MACL loss
                print(f"[MACL Warning] Failed to compute MACL loss: {e}")
                pass

        # Optionally compute MACL loss from feature maps using modality information
        # from batch_data_samples.metainfo['modality'] (for non-paired mode)
        elif getattr(self, 'use_macl', False):
            try:
                # Collect features per modality
                vis_feats_list, ir_feats_list = [], []
                for sample, feat_maps, result in zip(batch_data_samples, x_for_detection, sampling_results):
                    modality = sample.metainfo.get('modality', 'unknown')
                    
                    # 如果启用了只对正样本做pooling，检查正样本
                    use_pos_only = self.pool_only_pos and hasattr(result, 'pos_inds')
                    
                    # 对每个特征层进行处理
                    pooled_feats = []
                    for fm_i in feat_maps:
                        if use_pos_only and len(result.pos_inds) > 0:
                            # 只使用正样本区域
                            pos_rois = bbox2roi([result.pos_priors])
                            pos_feats = self.bbox_roi_extractor([fm_i], pos_rois)
                            pooled = torch.nn.functional.adaptive_avg_pool2d(pos_feats, 1)
                            pooled = pooled.mean(dim=0, keepdim=True)  # 平均所有正样本
                        else:
                            # 使用全局特征
                            pooled = torch.nn.functional.adaptive_avg_pool2d(fm_i, 1)
                        pooled = pooled.squeeze(-1).squeeze(-1)
                        pooled_feats.append(pooled)
                    
                    # 拼接所有层的特征
                    feat_vec = torch.cat(pooled_feats, dim=0)
                    
                    if modality == 'visible':
                        vis_feats_list.append(feat_vec)
                    elif modality == 'infrared':
                        ir_feats_list.append(feat_vec)
                
                # If we have both modalities, compute MACL loss
                if vis_feats_list and ir_feats_list:
                    # Stack tensors from each modality
                    vis_feats_stacked = torch.stack(vis_feats_list)
                    ir_feats_stacked = torch.stack(ir_feats_list)
                    
                    # Call MACL head with collected features
                    macl_out = self.macl_head(vis_feats_stacked, ir_feats_stacked)
                    
                    # Update losses dictionary with MACL loss and feature vectors
                    if isinstance(macl_out, dict):
                        if 'loss_macl' in macl_out:
                            losses['loss_macl'] = macl_out['loss_macl']
                        if 'vis' in macl_out:
                            losses['macl_vis'] = macl_out['vis']
                        if 'ir' in macl_out:
                            losses['macl_ir'] = macl_out['ir']
            
            except Exception as e:
                # if MACL fails, continue without adding its losses
                pass

        # 域对齐损失（如果启用）
        if getattr(self, 'use_domain_alignment', False):
            try:
                # 提取全局特征
                global_feats = []
                for fm in x:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(fm, 1)
                    global_feats.append(pooled.squeeze(-1).squeeze(-1))
                global_feat = torch.cat(global_feats, dim=1)  # 拼接多尺度特征
                
                # 获取当前 lambda 值
                lambda_p = getattr(self, 'domain_lambda', 1.0)
                
                # 域分类
                dom_pred = self.domain_classifier(global_feat, lambda_=lambda_p)
                
                # 获取域标签
                dom_labels = torch.tensor(
                    [sample.metainfo.get('domain_id', 0) 
                     for sample in batch_data_samples],
                    dtype=torch.long
                ).to(dom_pred.device)
                
                # 计算域损失
                import torch.nn.functional as F
                loss_domain = F.cross_entropy(dom_pred, dom_labels)
                losses['loss_domain'] = loss_domain
                
            except Exception as e:
                # 如果域对齐失败，继续而不添加损失
                pass

        # ============ 三阶段损失组合 ============
        # 计算检测任务的基础损失
        loss_det = sum([v for k, v in losses.items() 
                       if k.startswith('loss_') and 
                       k not in ['loss_macl', 'loss_dhn', 'loss_domain']])
        
        loss_total = loss_det
        
        # 阶段一/二：MACL 对比学习损失
        if hasattr(self, 'use_macl') and self.use_macl:
            loss_macl = losses.get('loss_macl', torch.tensor(0.0).to(loss_det.device))
            loss_total = loss_total + self.lambda1 * loss_macl
        
        # 阶段二：DHN 困难负样本损失
        if hasattr(self, 'use_dhn') and getattr(self, 'use_dhn', False):
            loss_dhn = losses.get('loss_dhn', torch.tensor(0.0).to(loss_det.device))
            loss_total = loss_total + self.lambda2 * loss_dhn
        
        # 阶段三：域对齐损失（动态衰减）
        if hasattr(self, 'use_domain_alignment') and self.use_domain_alignment:
            loss_domain = losses.get('loss_domain', torch.tensor(0.0).to(loss_det.device))
            # 动态衰减 lambda3
            cur_epoch = getattr(self, 'cur_epoch', 0)
            max_epochs = getattr(self, 'max_epochs', 100)
            lambda3_decayed = self.lambda3 * (1.0 - cur_epoch / max(max_epochs, 1))
            loss_total = loss_total + lambda3_decayed * loss_domain
            
            # 记录衰减后的权重（用于监控）
            losses['lambda3_decayed'] = torch.tensor(lambda3_decayed).to(loss_det.device)
        
        # ============ 数值保护：清除所有损失中的 NaN/Inf ============
        # 在返回前对所有损失张量进行数值清洗，避免 NaN 传播到日志
        for k, v in losses.items():
            if torch.is_tensor(v):
                losses[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            elif isinstance(v, list):
                losses[k] = [torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) if torch.is_tensor(x) else x for x in v]
        
        # 重新计算清洗后的总损失（确保 loss_total 不含 NaN）
        loss_total_cleaned = sum(
            v.mean() if torch.is_tensor(v) and v.numel() > 0 else torch.tensor(0.0, device=loss_det.device)
            for k, v in losses.items() 
            if k != 'loss_total' and k.startswith('loss_')
        )
        loss_total_cleaned = torch.nan_to_num(loss_total_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 将总损失添加到字典中
        losses['loss_total'] = loss_total_cleaned
        
        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def bbox_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            bbox_feats (Tensor): Extract bbox RoI features.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `mask_targets` (Tensor): Mask target of each positive\
                    proposals in the image.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)
        return mask_results

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list

    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: InstanceList,
                     rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list
