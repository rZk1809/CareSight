"""Quick test to verify dashboard can start."""

import sys
import subprocess
import time
import requests
from pathlib import Path

def test_dashboard_startup():
    """Test that the dashboard can start and respond."""
    
    # Set up environment
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "src" / "dashboards" / "streamlit_app.py"
    
    print("🧪 Testing dashboard startup...")
    print(f"📁 Project root: {project_root}")
    print(f"📄 Dashboard file: {dashboard_path}")
    
    # Start Streamlit in background
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.address=localhost",
        "--server.port=8502",  # Use different port to avoid conflicts
        "--server.headless=true"
    ]
    
    try:
        print("🚀 Starting Streamlit process...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root
        )
        
        # Wait a bit for startup
        print("⏳ Waiting for startup...")
        time.sleep(10)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Process is running")
            
            # Try to access the dashboard
            try:
                response = requests.get("http://localhost:8502", timeout=5)
                print(f"✅ Dashboard responded with status: {response.status_code}")
                success = True
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Dashboard not accessible: {e}")
                success = False
            
            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
            print("🛑 Process terminated")
            
        else:
            # Process died, get error output
            stdout, stderr = process.communicate()
            print("❌ Process died during startup")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            success = False
            
        return success
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_dashboard_startup()
    if success:
        print("\n🎉 Dashboard startup test PASSED!")
        print("\n💡 To run the dashboard manually:")
        print("   python -m streamlit run src/dashboards/streamlit_app.py --server.address=localhost")
        print("   OR")
        print("   python scripts/run_dashboard.py")
        print("   OR")
        print("   run_dashboard.bat  (on Windows)")
    else:
        print("\n❌ Dashboard startup test FAILED!")
        print("Please check the error messages above.")
    
    sys.exit(0 if success else 1)
