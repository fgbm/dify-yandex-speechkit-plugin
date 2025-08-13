#!/usr/bin/env python3
"""
Test runner for Yandex SpeechKit plugin tests
"""

import logging
import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_environment():
    """Load environment variables from .env file if it exists"""
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value


def run_all_tests():
    """Discover and run all tests"""
    # Load environment variables
    load_environment()

    # Discover tests
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern="test_*.py")

    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2, stream=sys.stdout, descriptions=True, failfast=False
    )

    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # Return exit code
    return 0 if result.wasSuccessful() else 1


def run_integration_tests():
    """Run integration tests that require real API credentials"""
    api_key = os.getenv("YANDEX_API_KEY")
    if not api_key:
        print("Warning: YANDEX_API_KEY not set. Skipping integration tests.")
        print("Set YANDEX_API_KEY in .env file to run integration tests.")
        return 0

    print("Running integration tests with real API...")
    # Here you could add specific integration test logic
    return run_all_tests()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Yandex SpeechKit plugin tests")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests with real API (requires YANDEX_API_KEY)",
    )
    parser.add_argument(
        "--pattern", default="test_*.py", help="Test file pattern (default: test_*.py)"
    )

    args = parser.parse_args()

    if args.integration:
        exit_code = run_integration_tests()
    else:
        exit_code = run_all_tests()

    sys.exit(exit_code)
