from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict

@router.get("/analytics/phoneme-weaknesses")
async def get_phoneme_weaknesses(
    db = Depends(get_database),
    current_user: dict = Depends(get_current_admin_user)
):
    """
    MongoDB Aggregation Pipeline to detect recurring syllable weaknesses.
    Powers the Remote Therapist Dashboard.
    """
    try:
        pipeline = [
            # Stage 1: Filter only completed sessions with valid scores
            {"$match": {"gop_score": {"$exists": True}, "gop_score": {"$lt": 65}}},
            # Stage 2: Group by the target phoneme/word
            {"$group": {
                "_id": "$target_word",
                "average_gop": {"$avg": "$gop_score"},
                "failure_count": {"$sum": 1},
                "avg_airflow": {"$avg": "$airflow_score"}
            }},
            # Stage 3: Sort to find the most problematic phonemes
            {"$sort": {"failure_count": -1, "average_gop": 1}},
            # Stage 4: Limit to top 5 weaknesses for the dashboard
            {"$limit": 5}
        ]
        
        weaknesses = await db.sessions.aggregate(pipeline).to_list(length=5)
        
        # RL State Scaffolding (Preparing for DDPG integration)
        rl_state_vector = {
            "current_difficulty_phase": 1,
            "exploration_rate": 0.1,
            "state_features": [w["average_gop"] for w in weaknesses]
        }
        
        return {
            "status": "success",
            "therapist_action_items": weaknesses,
            "rl_agent_state": rl_state_vector
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
