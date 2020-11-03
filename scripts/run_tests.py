import os
import json
import psutil
import subprocess
from apscheduler.schedulers.background import BackgroundScheduler

TRIAL_TO_TRAIN = 1
scheduler = BackgroundScheduler()

def checkProcRunning(procName):
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

# This job runs once an hour
def try_and_run_trial():
    # Check if behavenet model is running
    proc_running = checkProcRunning('python behavenet')
    print('Process currently running: ', proc_running)
    if not proc_running:
        # 1) Update decoding_data file with new trial
        with open('../.behavenet/decoding_data.json', 'a') as f:
            print('Updating JSON file for new trial ', TRIAL_TO_TRAIN)
            data = json.loads(f)
            data['session'] = TRIAL_TO_TRAIN
            json.dump(data, f)

        print('Kicking off new process...')
        proc = subprocess.Popen(['nohup python', 'behavenet/fitting/decoder_grid_search.py --data_config ~/.behavenet/decoding_data.json --model_config ~/.behavenet/decoding_ae_model.json --training_config ~/.behavenet/decoding_training.json --compute_config ~/.behavenet/decoding_compute.json > plaw_ctypes.out &2>&1'])
        print('Updating trial_count...')
        TRIAL_TO_TRAIN += 1
        if TRIAL_TO_TRAIN > 4:
            scheduler.shutdown()

print('Adding job to scheduler')
scheduler.add_job(try_and_run_trial, 'interval', minutes=60)
print('Starting scheduler...')
scheduler.start()

