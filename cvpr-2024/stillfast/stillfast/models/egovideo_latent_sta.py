"""
StillFast (with EgoVideo-V) + LAS Latent Model for STA
Architecture: Two overlapping 8-frame sequences → StillFast → LAS Encoder → Temporal Difference → Action Latents
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/cluster/scratch/udemirbas')

from latent.models.auto import AutoModelForLatentAction
from einops import rearrange


class STAHeads(nn.Module):
    """Prediction heads for STA task: verb, noun, and time-to-contact"""
    
    def __init__(self, latent_dim, num_verbs, num_nouns):
        super().__init__()
        
        # Verb classification head
        self.verb_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_verbs)
        )
        
        # Noun classification head
        self.noun_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_nouns)
        )
        
        # Time-to-contact regression head
        self.ttc_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Confidence scoring head
        self.score_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, latent_dim]
        Returns:
            verb_logits: [B, num_verbs]
            noun_logits: [B, num_nouns]
            ttc_values: [B, 1]
            scores: [B]
        """
        verb_logits = self.verb_head(features)
        noun_logits = self.noun_head(features)
        ttc_values = self.ttc_head(features)
        scores = self.score_head(features).squeeze(-1)
        
        return verb_logits, noun_logits, ttc_values, scores


class StillFastFeatureExtractor(nn.Module):
    """
    Extracts encoder features from StillFast backbone at two consecutive frames
    Output format matches DinoV2: [B*T, num_patches, embed_dim]
    """
    def __init__(self, stillfast_backbone):
        super().__init__()
        self.backbone = stillfast_backbone
    
    def forward(self, still_img, fast_imgs):
        """
        Extract multi-scale encoder features from StillFast
        
        Args:
            still_img: [B, 3, H, W] single frame from Still branch
            fast_imgs: [B, T, 3, H, W] video frames from Fast branch (T=2 for sequential)
        
        Returns:
            encoder_features: [B*T, num_patches, embed_dim] - encoder-level features
        """
        # Get backbone features
        # StillFast backbone returns feature maps from combined feature pyramid
        # We'll extract features before RPN/ROI heads
        still_features = self.backbone.still_backbone(still_img)  # FPN output
        fast_features = self.backbone.fast_backbone(fast_imgs)   # 3D FPN output
        
        # Combine features (fuse still and fast branches)
        combined_features = self.backbone.combined_feature_fusion(still_features, fast_features)
        
        # Flatten spatial dimensions to get patch-like representation
        # combined_features: list of feature maps at different scales
        # Take the final combined feature map and reshape to [B*T, num_patches, embed_dim]
        
        B, T = fast_imgs.shape[0], fast_imgs.shape[1]
        
        # Use the finest resolution feature map
        final_features = combined_features[-1]  # [B, C, H, W] for Still or [B, C, T, H, W] for Fast
        
        if len(final_features.shape) == 4:  # Still branch output [B, C, H, W]
            # Spatial pooling to get fixed-size representation
            pooled = torch.nn.functional.adaptive_avg_pool2d(final_features, (16, 16))  # [B, C, 16, 16]
            B_still, C, H, W = pooled.shape
            features = rearrange(pooled, "B C H W -> B (H W) C")  # [B, 256, C]
            # Replicate for T frames
            features = features.unsqueeze(1).repeat(1, T, 1, 1)  # [B, T, 256, C]
            features = rearrange(features, "B T N C -> (B T) N C")  # [B*T, 256, C]
        else:  # Fast branch output [B, C, T, H, W]
            B_fast, C, T, H, W = final_features.shape
            # Spatial pooling
            pooled = torch.nn.functional.adaptive_avg_pool3d(final_features, (T, 16, 16))  # [B, C, T, 16, 16]
            features = rearrange(pooled, "B C T H W -> (B T) (H W) C")  # [B*T, 256, C]
        
        return features


class StillFastLatentSTA(nn.Module):
    """
    STA model using two overlapping 8-frame sequences through StillFast + LAS
    
    Pipeline:
        9 Frames total
        ├─ Sequence 1: [Frame 0-7] → Still Frame 7 + Fast Frames [0-7]
        └─ Sequence 2: [Frame 1-8] → Still Frame 8 + Fast Frames [1-8]
                ↓
        StillFast Backbone (extracts natural features)
                ↓
        Features Seq1 [B, D]  Features Seq2 [B, D]
                ↓
        Project to encoder dimension if needed
        [B, 256, D_encoder]  [B, 256, D_encoder]
                ↓
        LAS forward_encoder (process each separately)
                ↓
        Encoded_Seq1 [B, 256, D_encoder]  Encoded_Seq2 [B, 256, D_encoder]
                ↓
        Temporal Difference (in encoder space)
        action_patches = Encoded_Seq2 - Encoded_Seq1
                ↓
        MAP Block → Action Latents
                ↓
        STA Heads → Predictions
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Load StillFast model (with EgoVideo-V backbone)
        from stillfast.models.build import MODEL_REGISTRY
        self.stillfast = MODEL_REGISTRY.get(cfg.MODEL.STILLFAST.NAME)(cfg)
        
        # Freeze StillFast if specified
        if cfg.MODEL.STILLFAST.get('FREEZE', True):
            self.stillfast.eval()
            for param in self.stillfast.parameters():
                param.requires_grad = False
        
        # Load LAS latent model
        las_config_path = cfg.MODEL.LATENT.CONFIG_PATH
        self.latent_model = AutoModelForLatentAction.from_config(las_config_path)
        
        # Freeze LAS if specified
        if cfg.MODEL.LATENT.get('FREEZE_ENCODER', False):
            for param in self.latent_model.parameters():
                param.requires_grad = False
        
        # Feature projection will be initialized on first forward pass
        self.feature_embed_dim = None
        self.proj = None
        
        # STA prediction heads
        latent_action_dim = self.latent_model.action_latent_dim
        self.sta_heads = STAHeads(
            latent_dim=latent_action_dim,
            num_verbs=cfg.MODEL.NUM_VERBS,
            num_nouns=cfg.MODEL.NUM_NOUNS
        )
    
    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - 'video': [B, 9, 3, H, W] - 9 frames total for two overlapping 8-frame sequences
                - 'boxes': [B, N, 4] bounding boxes
                - 'verb_labels': [B, N] verb labels (training only)
                - 'noun_labels': [B, N] noun labels (training only)
                - 'ttc_targets': [B, N] time-to-contact targets (training only)
        
        Returns:
            loss_dict (training) or predictions (inference)
        """
        video = batch['video']  # [B, 9, 3, H, W]
        boxes = batch.get('boxes', None)
        B = video.shape[0]
        
        # Split into two overlapping 8-frame sequences
        # Sequence 1: Frames [0-7], Still Frame = Frame 7 (last)
        # Sequence 2: Frames [1-8], Still Frame = Frame 8 (last)
        
        seq1_frames = video[:, :8]  # [B, 8, 3, H, W]
        seq2_frames = video[:, 1:9]  # [B, 8, 3, H, W]
        
        still_frame_1 = seq1_frames[:, -1]  # Frame 7 - [B, 3, H, W]
        still_frame_2 = seq2_frames[:, -1]  # Frame 8 - [B, 3, H, W]
        
        # Process both sequences through StillFast
        with torch.no_grad() if self.cfg.MODEL.STILLFAST.get('FREEZE', True) else torch.enable_grad():
            # Create StillFastImageTensor inputs for each sequence
            from stillfast.datasets import StillFastImageTensor
            
            # Sequence 1
            images_seq1 = [StillFastImageTensor(still_frame_1[i:i+1], seq1_frames[i:i+1]) for i in range(B)]
            features_seq1 = self.stillfast.backbone(images_seq1)  # Natural StillFast features
            
            # Sequence 2
            images_seq2 = [StillFastImageTensor(still_frame_2[i:i+1], seq2_frames[i:i+1]) for i in range(B)]
            features_seq2 = self.stillfast.backbone(images_seq2)  # Natural StillFast features
        
        # Extract feature dimension from first forward pass
        if self.feature_embed_dim is None:
            # features are from FPN output, reshape to [B, num_patches, D]
            if isinstance(features_seq1, dict):
                feat = features_seq1['features']  # Get features from backbone output
            else:
                feat = features_seq1
            
            # Reshape FPN features to patch format for LAS
            # Assuming FPN outputs are at different scales, use the finest resolution
            if isinstance(feat, list):
                feat_map = feat[-1]  # Use finest resolution
            else:
                feat_map = feat
            
            # Spatial pooling to get fixed patch representation
            if len(feat_map.shape) == 4:  # [B, C, H, W]
                pooled = torch.nn.functional.adaptive_avg_pool2d(feat_map, (16, 16))
                self.feature_embed_dim = pooled.shape[1]
                num_patches = 256
            elif len(feat_map.shape) == 5:  # [B, C, T, H, W]
                pooled = torch.nn.functional.adaptive_avg_pool3d(feat_map, (1, 16, 16))
                self.feature_embed_dim = pooled.shape[1]
                num_patches = 256
            else:
                raise ValueError(f"Unexpected feature shape: {feat_map.shape}")
            
            # Initialize projection to LAS encoder dimension
            encoder_embed_dim = self.latent_model.encoder_embed_dim
            if self.feature_embed_dim != encoder_embed_dim:
                self.proj = nn.Linear(self.feature_embed_dim, encoder_embed_dim).to(features_seq1.device)
        
        # Extract and reshape features from both sequences
        def process_features(features):
            if isinstance(features, dict):
                feat = features['features']
            else:
                feat = features
            
            if isinstance(feat, list):
                feat_map = feat[-1]
            else:
                feat_map = feat
            
            # Spatial pooling
            if len(feat_map.shape) == 4:  # [B, C, H, W]
                pooled = torch.nn.functional.adaptive_avg_pool2d(feat_map, (16, 16))
                reshaped = rearrange(pooled, "B C H W -> B (H W) C")  # [B, 256, C]
            elif len(feat_map.shape) == 5:  # [B, C, T, H, W]
                # Average over temporal dimension
                pooled = torch.nn.functional.adaptive_avg_pool3d(feat_map, (1, 16, 16))
                pooled = pooled.squeeze(2)  # [B, C, 16, 16]
                reshaped = rearrange(pooled, "B C H W -> B (H W) C")  # [B, 256, C]
            else:
                raise ValueError(f"Unexpected feature shape: {feat_map.shape}")
            
            return reshaped
        
        feat_seq1 = process_features(features_seq1)  # [B, 256, D]
        feat_seq2 = process_features(features_seq2)  # [B, 256, D]
        
        # Project if needed
        if self.proj is not None:
            feat_seq1 = self.proj(feat_seq1)  # [B, 256, D_encoder]
            feat_seq2 = self.proj(feat_seq2)  # [B, 256, D_encoder]
        
        # Reshape for LAS encoder: [B, 1, 256, D_encoder]
        # (treating each sequence as 1 timestep for encoder input)
        feat_seq1_input = feat_seq1.unsqueeze(1)  # [B, 1, 256, D_encoder]
        feat_seq2_input = feat_seq2.unsqueeze(1)  # [B, 1, 256, D_encoder]
        
        # Create padding masks
        pad_mask = torch.ones(B, 1, device=feat_seq1.device, dtype=torch.long)
        
        # Process through LAS encoder
        with torch.no_grad() if self.cfg.MODEL.LATENT.get('FREEZE_ENCODER', False) else torch.enable_grad():
            # Pass each sequence through LAS forward_encoder
            encoded_seq1_list = self.latent_model.forward_encoder(feat_seq1_input, pad_mask)
            encoded_seq2_list = self.latent_model.forward_encoder(feat_seq2_input, pad_mask)
            
            # Extract encoded features: [T, 256, D_encoder] → [256, D_encoder]
            encoded_seq1 = encoded_seq1_list[0]  # [256, D_encoder]
            encoded_seq2 = encoded_seq2_list[0]  # [256, D_encoder]
            
            # Compute temporal difference in encoder space
            # This captures what changed across the 8-frame window
            action_patches = encoded_seq2 - encoded_seq1  # [256, D_encoder]
            
            # Reshape for MAP block: [B, 256, D_encoder]
            action_patches = action_patches.unsqueeze(0).expand(B, -1, -1)  # [B, 256, D_encoder]
            
            # Apply MAP block to get action latents
            action_features = self.latent_model.transE_MAP_block(action_patches)  # [B, n_query_tokens, action_latent_dim]
            action_features = rearrange(action_features, "B n_act d -> B (n_act d)")  # [B, n_query_tokens * latent_dim]
        
        # Get STA predictions
        verb_logits, noun_logits, ttc_values, scores = self.sta_heads(action_features)
        
        if self.training:
            return self._compute_loss(verb_logits, noun_logits, ttc_values, batch, boxes)
        else:
            return self._prepare_predictions(verb_logits, noun_logits, ttc_values, scores, boxes)
    
    def _compute_loss(self, verb_logits, noun_logits, ttc_values, batch, boxes):
        """Compute training losses"""
        batch_size = verb_logits.shape[0]
        num_boxes = boxes.shape[1] if boxes is not None else 1
        
        # Expand predictions to match boxes
        verb_expanded = verb_logits.unsqueeze(1).expand(-1, num_boxes, -1)
        noun_expanded = noun_logits.unsqueeze(1).expand(-1, num_boxes, -1)
        ttc_expanded = ttc_values.unsqueeze(1).expand(-1, num_boxes, -1)
        
        # Reshape for loss computation
        verb_flat = verb_expanded.reshape(-1, verb_logits.shape[-1])
        noun_flat = noun_expanded.reshape(-1, noun_logits.shape[-1])
        ttc_flat = ttc_expanded.reshape(-1, 1)
        
        verb_labels = batch['verb_labels'].reshape(-1)
        noun_labels = batch['noun_labels'].reshape(-1)
        ttc_targets = batch['ttc_targets'].reshape(-1, 1)
        
        # Compute losses
        verb_loss = nn.CrossEntropyLoss(ignore_index=-100)(verb_flat, verb_labels.long())
        noun_loss = nn.CrossEntropyLoss(ignore_index=-100)(noun_flat, noun_labels.long())
        
        # TTC loss only for valid targets
        valid_ttc = ~torch.isnan(ttc_targets).squeeze(-1)
        if valid_ttc.any():
            ttc_loss = nn.SmoothL1Loss()(ttc_flat[valid_ttc], ttc_targets[valid_ttc])
        else:
            ttc_loss = torch.tensor(0.0, device=verb_loss.device)
        
        # Weighted combination
        loss_weights = self.cfg.MODEL.STA_LOSS_WEIGHTS
        total_loss = loss_weights[0] * verb_loss + loss_weights[1] * noun_loss + loss_weights[2] * ttc_loss
        
        return {
            'loss_verb': verb_loss,
            'loss_noun': noun_loss,
            'loss_ttc': ttc_loss,
            'loss': total_loss
        }
    
    def _prepare_predictions(self, verb_logits, noun_logits, ttc_values, scores, boxes):
        """Prepare predictions for inference"""
        batch_size = verb_logits.shape[0]
        num_boxes = boxes.shape[1] if boxes is not None else 1
        
        predictions = []
        
        for b in range(batch_size):
            pred = {
                'boxes': boxes[b] if boxes is not None else torch.zeros(1, 4, device=verb_logits.device),
                'verbs': verb_logits[b].argmax(dim=-1).repeat(num_boxes) if num_boxes > 1 else verb_logits[b].argmax(dim=-1, keepdim=True),
                'nouns': noun_logits[b].argmax(dim=-1).repeat(num_boxes) if num_boxes > 1 else noun_logits[b].argmax(dim=-1, keepdim=True),
                'ttcs': ttc_values[b].repeat(num_boxes, 1) if num_boxes > 1 else ttc_values[b],
                'scores': scores[b].repeat(num_boxes) if num_boxes > 1 else scores[b].unsqueeze(0)
            }
            predictions.append(pred)
        
        return predictions
