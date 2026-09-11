"""Make local synthetic performance-review music (one build, steady, one build)."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ghost_in_the_deck.demo import create_demo
if __name__=='__main__':
    print(create_demo(ROOT/'out/performance_music',seconds=70,performance=True))
