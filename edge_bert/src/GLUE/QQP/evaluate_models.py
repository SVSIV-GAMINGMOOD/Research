from pathlib import Path
import sys


SRC_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "src")
sys.path.append(str(SRC_ROOT))

from GLUE.QQP.evaluation.evaluate_all_onnx_models import main


if __name__ == "__main__":
    main()
