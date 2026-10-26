import os
import sys
import time
import subprocess

def run_orchestrator():
    print("=" * 70)
    print(" ⛏️ MINE VEHICLE FOG SAFETY & DIGITAL TWIN SYSTEM LAUNCHER ⛏️")
    print("=" * 70)

    # 1. Dataset Generation Check
    csv_path = os.path.join('data', 'mine_telemetry.csv')
    if not os.path.exists(csv_path):
        print("\n[Step 1/4] Generating synthetic telemetry dataset...")
        subprocess.run([sys.executable, 'data/generate_dataset.py'], check=True)
    else:
        print("\n[Step 1/4] Dataset file data/mine_telemetry.csv already present.")

    # 2. Model Training Check
    metrics_path = os.path.join('models', 'model_metrics.json')
    if not os.path.exists(metrics_path):
        print("\n[Step 2/4] Training ML & Deep Neural Network models...")
        subprocess.run([sys.executable, 'models/train_pipeline.py'], check=True)
    else:
        print("\n[Step 2/4] Trained model artifacts already present in models/.")

    # 3. Launch FastAPI Backend
    print("\n[Step 3/4] Launching FastAPI REST Server on http://127.0.0.1:8000...")
    api_cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    api_process = subprocess.Popen(api_cmd)
    time.sleep(2.5)

    # 4. Launch Streamlit Web Application
    print("\n[Step 4/4] Launching Streamlit Interactive Web Application...")
    app_cmd = [sys.executable, "-m", "streamlit", "run", "app/web_app.py", "--server.port", "8501"]
    
    try:
        subprocess.run(app_cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down system servers...")
        api_process.terminate()
        print("System shutdown complete.")

if __name__ == '__main__':
    run_orchestrator()
