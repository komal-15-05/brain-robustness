"""
brainscore_test.py
------------------
Scores the AlexNet model on the public Brain-Score benchmark
"MajajHong2015public.IT-pls" following the official brain-score/vision README.
"""
import brainscore_vision
from brainscore_vision import score

print("=" * 60)
print("Brain-Score: AlexNet on MajajHong2015public.IT-pls")
print("=" * 60)

result = score(model_identifier="alexnet", benchmark_identifier="MajajHong2015public.IT-pls")

print(f"\nScore : {result}")
print(f"  raw : {result.values}")
print()
