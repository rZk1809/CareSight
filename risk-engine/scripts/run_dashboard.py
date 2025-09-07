"""Script to run the CareSight Risk Engine Streamlit dashboard."""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit dashboard with proper setup."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "src" / "dashboards" / "streamlit_app.py"
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Set Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root / "src")
    
    print("🚀 Starting CareSight Risk Engine Dashboard...")
    print(f"📁 Project root: {project_root}")
    print(f"📄 Dashboard file: {dashboard_path}")
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the dashboard")
    print("\n" + "="*60)

    try:
        # Run Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path), "--server.port=8501", "--server.address=localhost"]
        subprocess.run(cmd, env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running dashboard: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
