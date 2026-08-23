#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import sys

root = Path(__file__).resolve().parents[1]
test_file = root / "src" / "warehouse_core" / "test" / "test_core.py"
spec = importlib.util.spec_from_file_location("warehouse_core_tests", test_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tests = [getattr(module, name) for name in dir(module) if name.startswith("test_")]
for test in tests:
    test()
print(f"Logic tests passed: {len(tests)}")
