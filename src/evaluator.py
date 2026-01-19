"""
Evaluate recommendation quality
"""
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

class RecommendationEvaluator:
    """Metrics for recommendation systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def precision_at_k(recommended, relevant, k):
        """Precision@K: Fraction of recommendations that are relevant"""
        if not relevant or k == 0:
            return 0.0
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        return len(recommended_k & relevant_set) / k
    
    @staticmethod
    def recall_at_k(recommended, relevant, k):
        """Recall@K: Fraction of relevant items that are recommended"""
        if not relevant:
            return 0.0
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        return len(recommended_k & relevant_set) / len(relevant_set)
    
    @staticmethod
    def ndcg_at_k(recommended, relevant, k):
        """Normalized Discounted Cumulative Gain@K"""
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        
        dcg = sum([
            1.0 / np.log2(idx + 2)
            for idx, item in enumerate(recommended[:k])
            if item in relevant_set
        ])
        
        idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(relevant), k))])
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def hit_rate_at_k(recommended, relevant, k):
        """Hit Rate@K: Did we recommend at least one relevant item?"""
        if not relevant:
            return 0.0
        recommended_k = set(recommended[:k])
        relevant_set = set(relevant)
        return 1.0 if len(recommended_k & relevant_set) > 0 else 0.0
    
    def evaluate_model(self, engine, test_df, k_values=[5, 10, 20]):
        """Comprehensive evaluation"""
        self.logger.info("Starting evaluation...")
        
        metrics = {}
        for k in k_values:
            metrics[f'precision@{k}'] = []
            metrics[f'recall@{k}'] = []
            metrics[f'ndcg@{k}'] = []
            metrics[f'hit_rate@{k}'] = []
        
        test_grouped = test_df.groupby('user_id')['movie_id'].apply(list).to_dict()
        
        for user_id, relevant_movies in tqdm(test_grouped.items(), desc="Evaluating"):
            if user_id not in engine.preprocessor.user_id_map:
                continue
            
            max_k = max(k_values)
            recommendations = engine.recommend(user_id, n=max_k, filter_seen=True)
            rec_movie_ids = [rec['movie_id'] for rec in recommendations]
            
            for k in k_values:
                metrics[f'precision@{k}'].append(
                    self.precision_at_k(rec_movie_ids, relevant_movies, k)
                )
                metrics[f'recall@{k}'].append(
                    self.recall_at_k(rec_movie_ids, relevant_movies, k)
                )
                metrics[f'ndcg@{k}'].append(
                    self.ndcg_at_k(rec_movie_ids, relevant_movies, k)
                )
                metrics[f'hit_rate@{k}'].append(
                    self.hit_rate_at_k(rec_movie_ids, relevant_movies, k)
                )
        
        results = {metric: np.mean(values) for metric, values in metrics.items()}
        results['estimated_ctr_improvement_%'] = results['precision@10'] * 100
        
        self.logger.info("Evaluation complete!")
        
        return results
    
    def print_results(self, metrics):
        """Display results nicely"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        for metric, value in sorted(metrics.items()):
            if 'ctr' in metric.lower():
                print(f"{metric:.<50} {value:.2f}%")
            else:
                print(f"{metric:.<50} {value:.4f}")
        
        print("="*70 + "\n")