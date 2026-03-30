"""
=============================================================================
SETUP WINDOWS TASK SCHEDULER FOR AUTOMATED DATA COLLECTION
=============================================================================
Creates a Windows Task Scheduler task that runs the data collector
every 15 minutes automatically.

Usage:
  python setup_scheduler.py --create    # Create the scheduled task
  python setup_scheduler.py --delete    # Remove the scheduled task
  python setup_scheduler.py --status    # Check task status

Author: PhD Research - Traffic Flow Prediction
=============================================================================
"""

import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_SCRIPT = os.path.join(SCRIPT_DIR, "realtime_data_collector.py")
TASK_NAME = "BangaloreTrafficDataCollector"


def find_python():
    """Find the Python executable path."""
    python_path = sys.executable
    if os.path.exists(python_path):
        return python_path
    # Fallback
    for path in ['python', 'python3', 'py']:
        try:
            result = subprocess.run([path, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return path
        except FileNotFoundError:
            continue
    return None


def create_task():
    """Create Windows Task Scheduler task for every 15 minutes."""
    python_path = find_python()
    if not python_path:
        print("ERROR: Python not found!")
        return False

    print(f"Python: {python_path}")
    print(f"Script: {COLLECTOR_SCRIPT}")
    print(f"Task name: {TASK_NAME}")
    print()

    # Create the scheduled task using schtasks
    cmd = [
        'schtasks', '/create',
        '/tn', TASK_NAME,
        '/tr', f'"{python_path}" "{COLLECTOR_SCRIPT}"',
        '/sc', 'MINUTE',
        '/mo', '15',
        '/st', '00:00',
        '/f',  # Force overwrite if exists
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\nSUCCESS! Task '{TASK_NAME}' created.")
        print("The collector will run every 15 minutes automatically.")
        print(f"\nData will be saved to: {os.path.join(SCRIPT_DIR, 'data', 'raw_traffic_data.csv')}")
        print(f"Logs will be saved to: {os.path.join(SCRIPT_DIR, 'logs', 'collection.log')}")
        print(f"\nTo stop: python setup_scheduler.py --delete")
        print(f"To check: python setup_scheduler.py --status")
        return True
    else:
        print(f"\nFAILED to create task.")
        print(f"Error: {result.stderr}")
        print(f"\nTry running as Administrator, or use the manual method:")
        print(f"  1. Open Task Scheduler (taskschd.msc)")
        print(f"  2. Create Basic Task -> Name: {TASK_NAME}")
        print(f"  3. Trigger: Repeat every 15 minutes")
        print(f"  4. Action: Start a program")
        print(f"     Program: {python_path}")
        print(f"     Arguments: \"{COLLECTOR_SCRIPT}\"")
        print(f"     Start in: {SCRIPT_DIR}")
        return False


def delete_task():
    """Delete the scheduled task."""
    cmd = ['schtasks', '/delete', '/tn', TASK_NAME, '/f']
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' deleted successfully.")
    else:
        print(f"Could not delete task: {result.stderr}")


def check_status():
    """Check if the scheduled task exists and its status."""
    cmd = ['schtasks', '/query', '/tn', TASK_NAME, '/v', '/fo', 'LIST']
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Task '{TASK_NAME}' found:")
        print(result.stdout)
    else:
        print(f"Task '{TASK_NAME}' not found.")
        print("Run 'python setup_scheduler.py --create' to set it up.")

    # Also check data file
    data_file = os.path.join(SCRIPT_DIR, 'data', 'raw_traffic_data.csv')
    if os.path.exists(data_file):
        size = os.path.getsize(data_file)
        lines = sum(1 for _ in open(data_file, encoding='utf-8'))
        print(f"\nData file: {data_file}")
        print(f"  Size: {size/1024:.1f} KB")
        print(f"  Records: {lines - 1}")  # minus header
    else:
        print(f"\nNo data collected yet.")


def create_batch_scripts():
    """Create convenience batch scripts for start/stop."""

    start_bat = os.path.join(SCRIPT_DIR, "START_COLLECTION.bat")
    with open(start_bat, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'echo Starting Bangalore Traffic Data Collection...\n')
        f.write(f'echo This will collect data every 15 minutes.\n')
        f.write(f'echo Press Ctrl+C to stop.\n')
        f.write(f'echo.\n')
        f.write(f'"{find_python()}" "{COLLECTOR_SCRIPT}" --continuous\n')
        f.write(f'pause\n')
    print(f"Created: {start_bat}")

    stop_bat = os.path.join(SCRIPT_DIR, "STOP_COLLECTION.bat")
    with open(stop_bat, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'echo Stopping scheduled task...\n')
        f.write(f'schtasks /delete /tn {TASK_NAME} /f\n')
        f.write(f'echo Done.\n')
        f.write(f'pause\n')
    print(f"Created: {stop_bat}")

    test_bat = os.path.join(SCRIPT_DIR, "TEST_APIS.bat")
    with open(test_bat, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'echo Testing API Connections...\n')
        f.write(f'"{find_python()}" "{COLLECTOR_SCRIPT}" --test\n')
        f.write(f'pause\n')
    print(f"Created: {test_bat}")

    preprocess_bat = os.path.join(SCRIPT_DIR, "PREPROCESS_DATA.bat")
    preprocess_script = os.path.join(SCRIPT_DIR, "preprocess_realtime_data.py")
    with open(preprocess_bat, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'echo Preprocessing raw data to daily format...\n')
        f.write(f'"{find_python()}" "{preprocess_script}"\n')
        f.write(f'echo.\n')
        f.write(f'echo Generating quality report...\n')
        f.write(f'"{find_python()}" "{preprocess_script}" --report\n')
        f.write(f'pause\n')
    print(f"Created: {preprocess_bat}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup Data Collection Scheduler')
    parser.add_argument('--create', action='store_true', help='Create scheduled task')
    parser.add_argument('--delete', action='store_true', help='Delete scheduled task')
    parser.add_argument('--status', action='store_true', help='Check task status')
    parser.add_argument('--batch', action='store_true', help='Create batch scripts')
    args = parser.parse_args()

    if args.create:
        create_task()
        create_batch_scripts()
    elif args.delete:
        delete_task()
    elif args.status:
        check_status()
    elif args.batch:
        create_batch_scripts()
    else:
        print("Usage:")
        print("  python setup_scheduler.py --create   # Create scheduled task")
        print("  python setup_scheduler.py --delete   # Remove scheduled task")
        print("  python setup_scheduler.py --status   # Check status")
        print("  python setup_scheduler.py --batch    # Create .bat scripts")
