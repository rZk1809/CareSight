# CareSight Risk Engine - CI/CD Pipeline

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the CareSight Risk Engine.

## Overview

The CI/CD pipeline is implemented using GitHub Actions and provides:

- **Automated Testing**: Unit tests, integration tests, and end-to-end testing
- **Code Quality**: Linting, formatting, and security scanning
- **Model Monitoring**: Automated drift detection and performance monitoring
- **Automated Retraining**: Scheduled model retraining based on drift and performance
- **Deployment**: Automated deployment to staging and production environments
- **Containerization**: Docker-based deployment with health checks

## Pipeline Components

### 1. Continuous Integration (CI)

#### Test Job (`test`)
- **Triggers**: Push to main/develop, Pull requests
- **Actions**:
  - Code linting with flake8
  - Code formatting check with black
  - Type checking with mypy
  - Unit tests with pytest and coverage
  - Integration tests for all components
  - Upload coverage reports to Codecov

#### Security Job (`security`)
- **Triggers**: Push to main/develop, Pull requests
- **Actions**:
  - Dependency vulnerability scanning with safety
  - Static security analysis with bandit
  - Upload security reports as artifacts

### 2. Continuous Deployment (CD)

#### Build and Deploy Job (`build-and-deploy`)
- **Triggers**: Push to main branch (after tests pass)
- **Actions**:
  - Build Docker image
  - Run integration tests in container
  - Deploy to staging environment
  - Run smoke tests
  - Deploy to production (if staging tests pass)

### 3. Automated Model Management

#### Model Retraining Job (`model-retraining`)
- **Triggers**: Scheduled (daily at 2 AM UTC)
- **Actions**:
  - Check for data drift
  - Retrain model if drift detected
  - Evaluate new model performance
  - Update model registry
  - Create pull request with new model

#### Monitoring Job (`monitoring`)
- **Triggers**: Scheduled (daily at 2 AM UTC)
- **Actions**:
  - Run drift detection
  - Monitor model performance
  - Generate monitoring reports
  - Send alerts if issues detected

### 4. Release Management

#### Release Job (`release`)
- **Triggers**: Successful deployment to main
- **Actions**:
  - Generate changelog from commits
  - Create GitHub release with version tag
  - Attach build artifacts

## Configuration

### Environment Variables

Set these secrets in your GitHub repository:

```bash
# Required for deployment
AWS_ACCESS_KEY_ID          # AWS access key for deployment
AWS_SECRET_ACCESS_KEY      # AWS secret key for deployment
DOCKER_REGISTRY_URL        # Docker registry URL
DOCKER_REGISTRY_USERNAME   # Docker registry username
DOCKER_REGISTRY_PASSWORD   # Docker registry password

# Optional for notifications
SLACK_WEBHOOK_URL          # Slack webhook for alerts
EMAIL_SMTP_SERVER          # SMTP server for email alerts
EMAIL_USERNAME             # Email username
EMAIL_PASSWORD             # Email password

# Optional for external services
CODECOV_TOKEN              # Codecov token for coverage reports
```

### DVC Configuration

If using DVC with cloud storage, configure remote storage:

```bash
# AWS S3
dvc remote add -d myremote s3://my-bucket/dvc-cache
dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY

# Google Cloud Storage
dvc remote add -d myremote gs://my-bucket/dvc-cache
dvc remote modify myremote credentialpath /path/to/credentials.json

# Azure Blob Storage
dvc remote add -d myremote azure://my-container/dvc-cache
dvc remote modify myremote connection_string $AZURE_STORAGE_CONNECTION_STRING
```

## Local Development

### Docker Compose

Use Docker Compose for local development:

```bash
# Start core services (API + Dashboard)
docker-compose up

# Start with MLflow tracking
docker-compose --profile mlflow up

# Start with database
docker-compose --profile database up

# Run monitoring
docker-compose --profile monitoring up monitoring

# Run training
docker-compose --profile training up training
```

### Manual Testing

Run individual test suites:

```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests
python scripts/test_pipeline.py
python scripts/test_training.py
python scripts/test_api.py
python scripts/test_explainability.py
python scripts/test_monitoring.py

# API tests (requires running API)
python scripts/test_api.py

# Drift detection
python scripts/check_drift.py

# Monitoring
python scripts/run_monitoring.py
```

## Deployment Strategies

### Staging Environment

The staging environment is automatically deployed on every push to main:

- Uses the same Docker image as production
- Runs integration tests
- Validates API endpoints
- Tests model predictions

### Production Environment

Production deployment happens after staging validation:

- Blue-green deployment strategy
- Health checks before traffic switching
- Rollback capability
- Monitoring and alerting

### Model Deployment

Models are deployed through the model registry:

1. New models are registered in MLflow
2. Models are promoted through stages: Staging → Production
3. A/B testing can be configured for gradual rollout
4. Automatic rollback if performance degrades

## Monitoring and Alerting

### Automated Monitoring

The pipeline includes automated monitoring for:

- **Data Drift**: Statistical tests to detect distribution changes
- **Model Performance**: Tracking of key metrics over time
- **Bias Detection**: Fairness metrics across demographic groups
- **System Health**: API availability and response times

### Alert Channels

Configure alerts through:

- **Slack**: Real-time notifications for critical issues
- **Email**: Daily/weekly summary reports
- **GitHub Issues**: Automatic issue creation for failures
- **PagerDuty**: On-call alerts for production incidents

### Monitoring Dashboard

Access monitoring dashboards at:

- **MLflow UI**: http://localhost:5000 (model tracking)
- **Streamlit Dashboard**: http://localhost:8501 (business metrics)
- **API Health**: http://localhost:8000/health (system status)

## Troubleshooting

### Common Issues

1. **Tests Failing**:
   - Check test logs in GitHub Actions
   - Run tests locally to reproduce
   - Verify dependencies are up to date

2. **Docker Build Failures**:
   - Check Dockerfile syntax
   - Verify all required files are copied
   - Check for dependency conflicts

3. **Deployment Failures**:
   - Verify environment variables are set
   - Check deployment logs
   - Validate infrastructure configuration

4. **Model Training Failures**:
   - Check data availability
   - Verify DVC pipeline configuration
   - Review training logs

### Debug Commands

```bash
# Check pipeline status
dvc status

# Reproduce pipeline locally
dvc repro

# Check Docker image
docker build -t caresight-risk-engine .
docker run --rm caresight-risk-engine python scripts/test_pipeline.py

# Test API locally
docker-compose up -d api
python scripts/test_api.py
```

## Best Practices

### Code Quality

- Use type hints for all functions
- Maintain test coverage above 80%
- Follow PEP 8 style guidelines
- Write comprehensive docstrings

### Security

- Never commit secrets to version control
- Use environment variables for configuration
- Regularly update dependencies
- Scan for vulnerabilities

### Model Management

- Version all model artifacts
- Track experiment metadata
- Validate model performance before deployment
- Implement gradual rollout strategies

### Monitoring

- Set up comprehensive logging
- Monitor key business metrics
- Implement alerting for critical issues
- Regular review of monitoring data

## Support

For issues with the CI/CD pipeline:

1. Check the GitHub Actions logs
2. Review this documentation
3. Check the troubleshooting section
4. Create an issue in the repository
5. Contact the development team

## Contributing

When contributing to the CI/CD pipeline:

1. Test changes locally first
2. Update documentation
3. Add appropriate tests
4. Follow the existing patterns
5. Get review from team members
