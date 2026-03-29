"""Data aggregation and RL scaffolding engine."""
from typing import List, Dict

class AnalyticsEngine:
    """Generates automated bi-weekly therapist reports and DDPG state arrays."""
    
    @staticmethod
    def build_weakness_pipeline() -> List[Dict]:
        """
        Returns the exact MongoDB aggregation pipeline needed to 
        identify recurring phoneme weaknesses across sessions.
        """
        return [
            # Stage 1: Filter only sessions needing improvement
            {"$match": {
                "gop_score": {"$exists": True}, 
                "gop_score": {"$lt": 65.0}
            }},
            # Stage 2: Group by the target word/phoneme
            {"$group": {
                "_id": "$target_phoneme",
                "average_gop": {"$avg": "$gop_score"},
                "failure_count": {"$sum": 1},
                "avg_airflow": {"$avg": "$airflow_score"}
            }},
            # Stage 3: Sort by highest failure count
            {"$sort": {"failure_count": -1, "average_gop": 1}},
            # Stage 4: Limit to top 5 for the dashboard heatmap
            {"$limit": 5}
        ]

    @staticmethod
    def generate_rl_state_vector(vsa_trend: float, recent_gop_scores: List[float]) -> dict:
        """
        Formats the current acoustic metrics into a state vector 
        for Phase 2 Federated DDPG deployment.
        """
        avg_gop = sum(recent_gop_scores) / len(recent_gop_scores) if recent_gop_scores else 0
        
        # Determine rule-based difficulty phase to seed the RL agent
        current_phase = 1
        if avg_gop > 80.0:
            current_phase = 3
        elif avg_gop > 60.0:
            current_phase = 2

        return {
            "agent_state": {
                "vsa_velocity": vsa_trend,
                "gop_moving_average": avg_gop,
                "recommended_difficulty_tier": current_phase,
                "exploration_noise": 0.15 # Epsilon value for DDPG exploration
            },
            "is_ready_for_rl_transition": len(recent_gop_scores) > 100
        }

# Global instance
analytics_engine = AnalyticsEngine()
