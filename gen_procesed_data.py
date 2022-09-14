from pathlib import Path
from dataset import FakeQM9


# path = Path('data')
# dataset = FakeQM9(path)

path = Path('data')
dataset = FakeQM9(path, mode='test')
