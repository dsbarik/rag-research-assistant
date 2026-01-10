import subprocess
import sys
import os
import signal
import time


# Prevent "leaked semaphore" warnings from Hugging Face libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Setup environment: Add 'src' to path and force logs to show instantly
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONUNBUFFERED"] = "1"

    print("🚀 Starting RAG System...")

    # 1. Start Backend (Uvicorn)
    # preexec_fn=os.setsid groups this process so we can kill it cleanly later
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid 
    )

    # Wait for Backend to initialize
    time.sleep(2)

    # 2. Start Frontend (Gradio)
    ui_process = subprocess.Popen(
        [sys.executable, "ui/gradio_app.py"],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        preexec_fn=os.setsid
    )

    print("✅ Services are running. Press Ctrl+C to stop.")

    try:
        # Keep main script alive to monitor services
        while True:
            time.sleep(1)
            if api_process.poll() is not None or ui_process.poll() is not None:
                print("⚠️ A service exited unexpectedly.")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        # 3. Clean Shutdown
        # Kill the entire process group for each service to prevent orphans
        try:
            if api_process.poll() is None:
                os.killpg(os.getpgid(api_process.pid), signal.SIGTERM)
            
            if ui_process.poll() is None:
                os.killpg(os.getpgid(ui_process.pid), signal.SIGTERM)
        except Exception:
            pass
        
        print("👋 Cleanup complete.")

if __name__ == "__main__":
    main()