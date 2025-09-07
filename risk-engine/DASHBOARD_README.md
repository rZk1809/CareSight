# CareSight Risk Engine - Dashboard Guide

## 🎯 Overview

The CareSight Risk Engine Dashboard is a modern, interactive web application built with Streamlit that provides a comprehensive interface for healthcare professionals to:

- Monitor model performance and health
- Make real-time risk predictions for patients
- Analyze data quality and feature distributions
- Track system metrics and alerts

## 🚀 Quick Start

### Prerequisites

1. **Complete the ML pipeline** first to generate required model artifacts:
   ```bash
   make all
   ```

2. **Verify all components** are working:
   ```bash
   python scripts/test_dashboard.py
   ```

### Running the Dashboard

#### Option 1: Using Make (Recommended)
```bash
make dashboard
```

#### Option 2: Using Streamlit directly
```bash
streamlit run src/dashboards/streamlit_app.py
```

#### Option 3: Using the run script
```bash
python scripts/run_dashboard.py
```

### Accessing the Dashboard

Once started, the dashboard will be available at:
- **URL**: http://localhost:8501
- **Default Port**: 8501

## 📊 Dashboard Features

### 1. Overview Page
- **System Status**: Model health and availability
- **Key Metrics**: Training samples, validation samples, AUROC, positive rate
- **Recent Activity**: Latest predictions and system updates

### 2. Model Performance Page
- **Performance Metrics**: AUROC, AUPRC, Brier Score
- **Confusion Matrix**: Classification results visualization
- **ROC Curve**: Receiver Operating Characteristic analysis
- **Calibration Plot**: Model calibration quality assessment
- **Feature Importance**: Top contributing features

### 3. Patient Prediction Page
- **Interactive Form**: Input patient clinical features
- **Real-time Predictions**: Instant risk score calculation
- **Risk Interpretation**: Color-coded risk levels with recommendations
- **Risk Gauge**: Visual risk score representation
- **Clinical Guidance**: Actionable recommendations based on risk level

### 4. Data Quality Page
- **Dataset Overview**: Sample counts and basic statistics
- **Feature Completeness**: Missing data analysis
- **Feature Distributions**: Histograms and box plots
- **Data Validation**: Quality checks and alerts

### 5. Monitoring Page
- **Model Health**: Uptime, response time, prediction volume
- **Alerts & Warnings**: System notifications and issues
- **Performance Trends**: Historical performance tracking
- **Data Drift Detection**: Feature drift monitoring

## 🔧 Configuration

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Optional: API base URL for integration
export API_BASE_URL=http://localhost:8000

# Optional: Custom port
export STREAMLIT_SERVER_PORT=8501

# Optional: Enable debug mode
export STREAMLIT_LOGGER_LEVEL=debug
```

### Streamlit Configuration

Create `.streamlit/config.toml` for custom settings:

```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 📋 Required Data Files

The dashboard requires these files to function properly:

### Required Files
- `data/models/lgbm/lgbm.pkl` - Trained LightGBM model
- `data/models/lgbm/calibrator_isotonic.pkl` - Probability calibrator
- `data/reports/metrics.json` - Model evaluation metrics

### Optional Files (for enhanced features)
- `data/models/lgbm/val.parquet` - Validation data with predictions
- `data/processed/train.parquet` - Training dataset
- `data/reports/monitoring_report.json` - Monitoring results

### Generating Required Files

If files are missing, run the complete pipeline:

```bash
# Generate all required files
make all

# Or run individual stages
make data    # Generate training data
make train   # Train model and generate artifacts
```

## 🎨 User Interface Guide

### Navigation
- Use the **sidebar** to switch between different pages
- Each page focuses on a specific aspect of the system
- Pages are designed for different user roles and use cases

### Patient Prediction Workflow
1. Navigate to **"Patient Prediction"** page
2. Fill in the patient's clinical features:
   - **Clinical Counts**: Observations, encounters, medications
   - **Lab Values**: HbA1c measurements
   - **Vital Signs**: Blood pressure readings
3. Click **"Predict Risk"** to get results
4. Review the risk score and recommendations
5. Use the visual risk gauge for quick assessment

### Interpreting Risk Levels
- **🟢 Low Risk (0.0-0.25)**: Continue routine care
- **🟡 Medium Risk (0.25-0.5)**: Monitor closely, consider preventive measures
- **🔴 High Risk (0.5-1.0)**: Immediate clinical review and intervention

## 🔍 Troubleshooting

### Common Issues

#### 1. Dashboard Won't Start
```bash
# Check if Streamlit is installed
pip install streamlit

# Verify Python path
export PYTHONPATH=/path/to/risk-engine/src

# Check for port conflicts
lsof -i :8501  # On Unix/Mac
netstat -an | findstr :8501  # On Windows
```

#### 2. Missing Data Files
```bash
# Run the test script to identify missing files
python scripts/test_dashboard.py

# Generate missing files
make all
```

#### 3. Import Errors
```bash
# Ensure you're in the project root directory
cd /path/to/risk-engine

# Set Python path
export PYTHONPATH=$(pwd)/src

# Install missing dependencies
pip install -r requirements.txt
```

#### 4. Performance Issues
- **Large datasets**: The dashboard caches data automatically
- **Slow loading**: Check if model files are accessible
- **Memory usage**: Restart the dashboard if memory usage is high

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set debug level
export STREAMLIT_LOGGER_LEVEL=debug

# Run with verbose output
streamlit run src/dashboards/streamlit_app.py --logger.level=debug
```

### Log Files

Check these locations for logs:
- **Streamlit logs**: Usually in terminal output
- **Application logs**: `logs/` directory (if configured)
- **System logs**: Check system log files for errors

## 🔒 Security Considerations

### Production Deployment

For production use, consider these security measures:

1. **Authentication**: Add user authentication
2. **HTTPS**: Use SSL/TLS encryption
3. **Access Control**: Implement role-based access
4. **Data Privacy**: Ensure patient data protection
5. **Audit Logging**: Track user actions

### Example Production Configuration

```toml
# .streamlit/config.toml for production
[server]
port = 8501
address = "0.0.0.0"
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

## 🚀 Advanced Usage

### Custom Styling

Modify the CSS in `streamlit_app.py` to customize appearance:

```python
# Custom CSS example
st.markdown("""
<style>
    .main-header {
        color: #your-color;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)
```

### Integration with External Systems

The dashboard can be integrated with:
- **EHR Systems**: For real-time patient data
- **Monitoring Tools**: For system health tracking
- **Notification Systems**: For alerts and warnings

### Performance Optimization

For better performance:
- Use `@st.cache_data` for expensive computations
- Limit data size for visualizations
- Implement pagination for large datasets
- Use efficient data formats (Parquet vs CSV)

## 📞 Support

### Getting Help

1. **Check the logs** for error messages
2. **Run the test script** to identify issues
3. **Review this documentation** for common solutions
4. **Check the project issues** on GitHub

### Reporting Issues

When reporting issues, include:
- Error messages and stack traces
- Steps to reproduce the problem
- System information (OS, Python version)
- Dashboard configuration

## 🎯 Best Practices

### For Healthcare Professionals
1. **Validate predictions** with clinical judgment
2. **Understand model limitations** and confidence intervals
3. **Use risk scores** as decision support, not replacement
4. **Monitor patient outcomes** to validate predictions

### For System Administrators
1. **Regular monitoring** of system health
2. **Backup model artifacts** and configurations
3. **Update dependencies** regularly for security
4. **Monitor resource usage** and performance

### For Developers
1. **Test changes** thoroughly before deployment
2. **Follow coding standards** and documentation
3. **Implement proper error handling**
4. **Use version control** for all changes

## 🔄 Updates and Maintenance

### Regular Maintenance Tasks
- **Update model artifacts** when new models are trained
- **Refresh data files** with latest datasets
- **Monitor system performance** and resource usage
- **Review and update documentation**

### Version Updates
- **Streamlit updates**: Check compatibility with new versions
- **Dependency updates**: Update requirements.txt regularly
- **Security patches**: Apply security updates promptly

---

**Note**: This dashboard is designed for healthcare decision support. Always validate predictions with clinical expertise and follow your organization's protocols for patient care.
