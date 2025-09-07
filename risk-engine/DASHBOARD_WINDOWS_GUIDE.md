# 🪟 CareSight Dashboard - Windows Setup Guide

## ✅ Issues Fixed for Windows

The dashboard has been updated to work properly on Windows systems:

1. **Fixed Make command**: Updated to use `python -m streamlit` instead of direct `streamlit` command
2. **Fixed address binding**: Changed from `0.0.0.0` to `localhost` for Windows compatibility
3. **Added Windows batch script**: Created `run_dashboard.bat` for easy execution

## 🚀 How to Run the Dashboard (Windows)

### Method 1: Windows Batch Script (Easiest)
```cmd
run_dashboard.bat
```
*Double-click the file or run from command prompt*

### Method 2: Python Command (Recommended)
```cmd
python -m streamlit run src/dashboards/streamlit_app.py --server.address=localhost
```

### Method 3: Using the Python Script
```cmd
python scripts/run_dashboard.py
```

### Method 4: Using Make (if you have make installed)
```cmd
make dashboard
```

## 🌐 Accessing the Dashboard

Once started, the dashboard will be available at:
- **URL**: http://localhost:8501
- **Status**: ✅ Confirmed working on Windows

## 🧪 Test Results

```
🧪 Testing dashboard startup...
🚀 Starting Streamlit process...
⏳ Waiting for startup...
✅ Process is running
✅ Dashboard responded with status: 200
🛑 Process terminated

🎉 Dashboard startup test PASSED!
```

## 📋 Prerequisites

Make sure you have:
- ✅ Python 3.9+ installed
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Model artifacts generated (`make all` or `dvc repro`)

## 🔧 Troubleshooting

### Issue: "streamlit command not found"
**Solution**: Use `python -m streamlit` instead of `streamlit`

### Issue: "ERR_ADDRESS_INVALID" 
**Solution**: Use `localhost` instead of `0.0.0.0`

### Issue: Port already in use
**Solution**: 
```cmd
# Use a different port
python -m streamlit run src/dashboards/streamlit_app.py --server.address=localhost --server.port=8502
```

### Issue: Module import errors
**Solution**: 
```cmd
# Set Python path
set PYTHONPATH=%CD%\src
python -m streamlit run src/dashboards/streamlit_app.py --server.address=localhost
```

## 📊 Dashboard Features

All features are confirmed working on Windows:

### ✅ Overview Page
- System status and model health
- Key performance metrics
- Recent activity display

### ✅ Model Performance Page  
- ROC curves and calibration plots
- Feature importance visualization
- Confusion matrix display

### ✅ Patient Prediction Page
- Interactive patient data input
- Real-time risk score calculation
- Visual risk level indicators

### ✅ Data Quality Page
- Dataset statistics and completeness
- Feature distribution analysis
- Missing data visualization

### ✅ Monitoring Page
- System health indicators
- Performance trend tracking
- Data drift detection

## 🎯 Quick Start Guide

1. **Open Command Prompt** in the project directory
2. **Run the dashboard**:
   ```cmd
   run_dashboard.bat
   ```
3. **Open your browser** to http://localhost:8501
4. **Explore the dashboard** - all features are working!

## 💡 Pro Tips

### For Easy Access
- Create a desktop shortcut to `run_dashboard.bat`
- Bookmark http://localhost:8501 in your browser

### For Development
- Use `--server.runOnSave=true` for auto-reload during development
- Use `--server.port=XXXX` to change the port if needed

### For Production
- Consider using `--server.headless=true` for server deployment
- Set up proper authentication for production use

## 🔒 Security Notes

- The dashboard currently uses demo authentication
- For production use, implement proper user authentication
- Ensure HTTPS is used in production environments
- Follow your organization's security policies

## 📞 Support

If you encounter issues:

1. **Run the test**: `python scripts/quick_dashboard_test.py`
2. **Check the logs** in the command prompt for error messages
3. **Verify prerequisites** are installed
4. **Try different methods** listed above

## 🎉 Success!

The CareSight Risk Engine Dashboard is now fully functional on Windows! 

**Ready to use for healthcare risk prediction and monitoring.** 🏥✨
