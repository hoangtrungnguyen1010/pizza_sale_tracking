#!/usr/bin/env python3
"""
Test runner script for the pizza counting application.
This script sets up the proper Python path and runs all tests.
"""

import sys
import os
import unittest
import subprocess

def setup_environment():
    """Set up the Python environment for running tests."""
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, 'app')
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    
    print(f"Project root: {project_root}")
    print(f"App path: {app_path}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

def run_single_test(test_file):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    try:
        # Run the test file
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"Test {test_file} timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

def run_all_tests():
    """Run all test files in the project."""
    setup_environment()
    
    # List of test files to run
    test_files = [
        'test_core_objects_comprehensive.py',
        'test_trackers_comprehensive.py', 
        'test_matching_strategies.py',
        'test_configuration_system.py',
        'test_structure_only.py',
        'test_video_processing_simple.py',
        'test_simple_pizza_logic.py',
        'test_pizza_tracker.py'
    ]
    
    # Filter to only existing files
    existing_tests = [f for f in test_files if os.path.exists(f)]
    
    print(f"Found {len(existing_tests)} test files to run:")
    for test in existing_tests:
        print(f"  - {test}")
    
    # Run each test
    results = {}
    for test_file in existing_tests:
        success = run_single_test(test_file)
        results[test_file] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_file, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_file}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

def run_with_unittest_discover():
    """Run tests using unittest discover."""
    setup_environment()
    
    print("Running tests with unittest discover...")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-v'], 
                              capture_output=True, text=True, timeout=60)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Tests timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pizza counting application tests')
    parser.add_argument('--discover', action='store_true', 
                       help='Use unittest discover instead of individual files')
    parser.add_argument('--test', type=str, 
                       help='Run a specific test file')
    
    args = parser.parse_args()
    
    if args.test:
        setup_environment()
        success = run_single_test(args.test)
        sys.exit(0 if success else 1)
    elif args.discover:
        success = run_with_unittest_discover()
        sys.exit(0 if success else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1) 