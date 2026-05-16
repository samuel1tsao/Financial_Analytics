import sys
sys.path.append('./backend')
from vector_encoder import RLRecommender
import logging

logging.basicConfig(level=logging.INFO)

r = RLRecommender()
print(f"Number of models loaded: {len(r.models)}")
