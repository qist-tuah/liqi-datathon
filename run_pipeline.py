"""Single entry point.  Usage:  python run_pipeline.py [convert|profile|clean|all]"""
import argparse
import time

from src import clean, convert, profiling

STEPS = {"convert": convert.run, "profile": profiling.run, "clean": clean.run}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=[*STEPS, "all"])
    args = parser.parse_args()
    for name in STEPS if args.step == "all" else [args.step]:
        print(f"\n=== {name} ===")
        start = time.time()
        STEPS[name]()
        print(f"({name} took {time.time() - start:,.1f}s)")
