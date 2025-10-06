"""
Sound Event Detection Evaluation Metrics

This implementation follows the event-based evaluation methodology described in:
de Benito-Gorrón, D., Ramos, D., & Toledano, D.T. (2021). 
"An Analysis of Sound Event Detection under Acoustic Degradation Using Multi-Resolution Systems."
Applied Sciences, 11(23), 11561. https://doi.org/10.3390/app112311561

The evaluation uses collar-based matching where predicted events are matched to ground truth events
if both onset times (within 200ms) and offset times (within max(200ms, 20% of event duration)) align.
The primary metric is Macro F1, computed as the average of class-wise F1 scores, following the 
DCASE Challenge 2020 Task 4 evaluation protocol.

Adapted for polyphonic sound event detection in gut health monitoring with three event classes:
'b' (burst), 'mb' (multiple burst), 'h' (harmonics).
"""

import numpy as np
from collections import defaultdict


class SEDEvaluator:
    
    def __init__(self, onset_collar=0.2, offset_collar=0.2, offset_collar_rate=0.2):
        self.onset_collar = onset_collar
        self.offset_collar = offset_collar
        self.offset_collar_rate = offset_collar_rate
        
    def load_annotations(self, filepath):
        events = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        onset = float(parts[0])
                        offset = float(parts[1])
                        label = parts[2]
                        events.append({
                            'onset': onset,
                            'offset': offset,
                            'event_label': label
                        })
        except Exception as e:
            raise IOError(f"Error reading file {filepath}: {str(e)}")
        
        return events
    
    def match_events(self, reference_events, estimated_events):
        classes = set()
        for event in reference_events + estimated_events:
            classes.add(event['event_label'])
        
        results = {
            'class_wise': defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0}),
            'overall': {'TP': 0, 'FP': 0, 'FN': 0}
        }
        
        matched_estimates = set()
        
        for ref_event in reference_events:
            ref_class = ref_event['event_label']
            ref_onset = ref_event['onset']
            ref_offset = ref_event['offset']
            ref_duration = ref_offset - ref_onset
            
            offset_collar_current = max(
                self.offset_collar,
                self.offset_collar_rate * ref_duration
            )
            
            matched = False
            for est_idx, est_event in enumerate(estimated_events):
                if est_idx in matched_estimates:
                    continue
                
                est_class = est_event['event_label']
                est_onset = est_event['onset']
                est_offset = est_event['offset']
                
                if est_class != ref_class:
                    continue
                
                onset_diff = abs(est_onset - ref_onset)
                if onset_diff > self.onset_collar:
                    continue
                
                offset_diff = abs(est_offset - ref_offset)
                if offset_diff > offset_collar_current:
                    continue
                
                matched = True
                matched_estimates.add(est_idx)
                results['class_wise'][ref_class]['TP'] += 1
                results['overall']['TP'] += 1
                break
            
            if not matched:
                results['class_wise'][ref_class]['FN'] += 1
                results['overall']['FN'] += 1
        
        for est_idx, est_event in enumerate(estimated_events):
            if est_idx not in matched_estimates:
                est_class = est_event['event_label']
                results['class_wise'][est_class]['FP'] += 1
                results['overall']['FP'] += 1
        
        return results
    
    def compute_f1(self, tp, fp, fn):
        if tp == 0 and fp == 0 and fn == 0:
            return 0.0
        
        if 2 * tp + fp + fn == 0:
            return 0.0
        
        f1 = (2 * tp) / (2 * tp + fp + fn)
        return f1
    
    def compute_precision_recall(self, tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall
    
    def evaluate(self, reference_file, estimated_file):
        reference_events = self.load_annotations(reference_file)
        estimated_events = self.load_annotations(estimated_file)
        
        match_results = self.match_events(reference_events, estimated_events)
        
        evaluation = {
            'class_wise': {},
            'overall': {},
            'macro_f1': 0.0
        }
        
        f1_scores = []
        for class_label, counts in match_results['class_wise'].items():
            tp = counts['TP']
            fp = counts['FP']
            fn = counts['FN']
            
            f1 = self.compute_f1(tp, fp, fn)
            precision, recall = self.compute_precision_recall(tp, fp, fn)
            
            evaluation['class_wise'][class_label] = {
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            f1_scores.append(f1)
        
        evaluation['macro_f1'] = np.mean(f1_scores) if f1_scores else 0.0
        
        overall_tp = match_results['overall']['TP']
        overall_fp = match_results['overall']['FP']
        overall_fn = match_results['overall']['FN']
        
        overall_f1 = self.compute_f1(overall_tp, overall_fp, overall_fn)
        overall_precision, overall_recall = self.compute_precision_recall(
            overall_tp, overall_fp, overall_fn
        )
        
        evaluation['overall'] = {
            'TP': overall_tp,
            'FP': overall_fp,
            'FN': overall_fn,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1
        }
        
        return evaluation
    
    def print_results(self, evaluation):
        print("=" * 70)
        print("SOUND EVENT DETECTION EVALUATION RESULTS")
        print("=" * 70)
        print()
        
        print("Class-wise Metrics:")
        print("-" * 70)
        print(f"{'Class':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print("-" * 70)
        
        for class_label in sorted(evaluation['class_wise'].keys()):
            metrics = evaluation['class_wise'][class_label]
            print(f"{class_label:<15} "
                  f"{metrics['TP']:>6} "
                  f"{metrics['FP']:>6} "
                  f"{metrics['FN']:>6} "
                  f"{metrics['precision']:>8.3f} "
                  f"{metrics['recall']:>8.3f} "
                  f"{metrics['f1_score']:>8.3f}")
        
        print("-" * 70)
        print()
        
        print("Overall Metrics:")
        print("-" * 70)
        overall = evaluation['overall']
        print(f"Total TP: {overall['TP']}")
        print(f"Total FP: {overall['FP']}")
        print(f"Total FN: {overall['FN']}")
        print(f"Overall Precision: {overall['precision']:.3f}")
        print(f"Overall Recall: {overall['recall']:.3f}")
        print(f"Overall F1 Score: {overall['f1_score']:.3f}")
        print()
        print(f"Macro F1 Score: {evaluation['macro_f1']:.3f}")
        print("=" * 70)


if __name__ == "__main__":
    evaluator = SEDEvaluator(
        onset_collar=0.2,
        offset_collar=0.2,
        offset_collar_rate=0.2
    )
    
    try:
        results = evaluator.evaluate(
            reference_file='AS_1.txt',
            estimated_file='pred_AS_1.txt'
        )
        evaluator.print_results(results)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
