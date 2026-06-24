"""
PyTorch Lightning Task for StillFast+LAS Latent STA
"""

import pytorch_lightning as pl
import torch
import itertools
import json

from .base_task import BaseTask
from stillfast.evaluation.sta_metrics import OverallMeanAveragePrecision
from stillfast.tasks.utils import PackedTensorDictionary
from stillfast.utils.distributed import list_gather


class StillFastLatentSTATask(BaseTask):
    """Task for STA using StillFast encoder features + LAS latent head"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.learning_rate = self.cfg.SOLVER.BASE_LR
        self.checkpoint_metric = 'map_box_noun_verb_ttc'
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss_dict = self.model(batch)
        
        # Extract total loss
        loss = loss_dict['loss']
        
        # Log individual losses
        for k, v in loss_dict.items():
            if k != 'loss':
                self.log(f'train/{k}', v.item())
        
        self.log('train/loss', loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        preds = self.model(batch)
        
        return {
            'uids': batch['uids'],
            'preds': PackedTensorDictionary({
                'boxes': [p['boxes'] for p in preds],
                'verbs': [p['verbs'] for p in preds],
                'nouns': [p['nouns'] for p in preds],
                'ttcs': [p['ttcs'] for p in preds],
                'scores': [p['scores'] for p in preds]
            })
        }
    
    def validation_epoch_end(self, outs):
        """Compute validation metrics"""
        outs = list_gather(outs)
        
        # Aggregate predictions
        all_uids = list(itertools.chain(*[o['uids'] for o in outs]))
        all_preds = sum([o['preds'] for o in outs]).unpack()
        
        # Compute mAP
        metric = OverallMeanAveragePrecision()
        for pred_boxes in all_preds['boxes']:
            if len(pred_boxes) > 0:
                metric.add(pred_boxes.cpu().numpy())
        
        metrics = metric.compute()
        self.log('val/map', metrics.get('map', 0.0))
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        preds = self.model(batch)
        
        return {
            'uids': batch['uids'],
            'preds': PackedTensorDictionary({
                'boxes': [p['boxes'] for p in preds],
                'verbs': [p['verbs'] for p in preds],
                'nouns': [p['nouns'] for p in preds],
                'ttcs': [p['ttcs'] for p in preds],
                'scores': [p['scores'] for p in preds]
            })
        }
    
    def test_epoch_end(self, outs):
        """Generate test output in Ego4D format"""
        outs = list_gather(outs)
        
        uids = list(itertools.chain(*[o['uids'] for o in outs]))
        preds = sum([o['preds'] for o in outs]).unpack()
        
        pred_boxes = preds['boxes']
        pred_nouns = preds['nouns']
        pred_verbs = preds['verbs']
        pred_ttcs = preds['ttcs']
        pred_scores = preds['scores']
        
        pred_detections = {
            uid: {
                "boxes": boxes,
                "nouns": nouns,
                "verbs": verbs,
                "ttcs": ttcs,
                "scores": scores
            } for uid, boxes, nouns, verbs, ttcs, scores in zip(
                uids, pred_boxes, pred_nouns, pred_verbs, pred_ttcs, pred_scores
            )
        }
        
        if self.cfg.TEST.OUTPUT_JSON:
            output_dict = {
                'version': '1.0',
                'challenge': 'ego4d_short_term_object_interaction_anticipation',
                'results': {}
            }
            
            for uid, pred in pred_detections.items():
                output_dict['results'][uid] = []
                for box, noun, verb, ttc, score in zip(
                    pred['boxes'], pred['nouns'], pred['verbs'],
                    pred['ttcs'], pred['scores']
                ):
                    output_dict['results'][uid].append({
                        'box': [float(b) for b in box],
                        'score': float(score),
                        'noun_category_id': int(noun) - 1,
                        'verb_category_id': int(verb),
                        'time_to_contact': float(ttc.item() if hasattr(ttc, 'item') else ttc)
                    })
            
            with open(self.cfg.TEST.OUTPUT_JSON, 'w') as f:
                json.dump(output_dict, f)
            
            print(f"Test results saved to {self.cfg.TEST.OUTPUT_JSON}")
        
        print(f"Test done on {len(pred_detections)} predictions, "
              f"{sum([len(b) for b in pred_boxes])} boxes in total")
