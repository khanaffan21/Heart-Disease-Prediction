# Contributing to Heart Disease Predictor

Thank you for interest in contributing! This document outlines the contribution workflow and best practices.

## Code of Conduct

- Be respectful and inclusive
- Focus on the code, not the person
- Help others learn and grow

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/khanaffan21/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Make Changes

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Keep functions small and focused
- Use meaningful variable names

### 2. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_model.py::test_model_training -v

# With coverage
pytest tests/ --cov=scripts --cov-report=html
```

### 3. Train & Validate Model

```bash
python scripts/train.py
```

This regenerates:
- `models/heart_disease_model.pkl`
- `models/preprocessing_pipeline.pkl`
- `MODEL_REPORT.md`

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add feature description

More detailed explanation if needed.
- Bullet point 1
- Bullet point 2"
```

**Commit message format:**
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring
- `docs:` Documentation update
- `test:` Test additions/changes
- `chore:` Maintenance tasks

### 5. Push & Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Reference any related issues (#123)
- Screenshots if UI changes

## Areas for Contribution

### 🔍 Code Improvements
- [ ] Improve model accuracy (hyperparameter tuning)
- [ ] Add cross-validation
- [ ] Implement ensemble methods
- [ ] Optimize preprocessing pipeline
- [ ] Add caching for predictions

### 🧪 Testing
- [ ] Expand test coverage
- [ ] Add integration tests
- [ ] Create end-to-end tests
- [ ] Add performance benchmarks
- [ ] Improve error handling

### 📚 Documentation
- [ ] Improve README clarity
- [ ] Add API usage examples
- [ ] Create troubleshooting guide
- [ ] Document model limitations
- [ ] Add architecture diagrams

### 🎨 Frontend
- [ ] Improve UI/UX
- [ ] Add dark/light mode toggle
- [ ] Enhance mobile responsiveness
- [ ] Add data visualization improvements
- [ ] Improve accessibility (a11y)

### 🚀 Deployment
- [ ] Docker support
- [ ] Kubernetes manifests
- [ ] AWS/GCP integration
- [ ] CI/CD improvements
- [ ] Load testing

### 📊 Model Enhancements
- [ ] Feature engineering
- [ ] Handle class imbalance better
- [ ] Add confidence intervals
- [ ] Implement SHAP explanations
- [ ] Multi-class classification

## File Structure

```
Heart-Disease-Prediction/
├── scripts/
│   └── train.py              # Model training logic (core)
├── api/
│   └── app.py                # Flask API (core)
├── tests/                     # All tests
├── index.html                 # Frontend (core)
├── heart.csv                  # Data (core—don't modify)
├── models/                    # Generated artifacts
├── README.md                  # User documentation
└── .github/workflows/         # CI/CD pipelines
```

**Core files** (touch carefully):
- `scripts/train.py` – Ensure backward compatibility
- `api/app.py` – Test all endpoints thoroughly
- `index.html` – Maintain backward compatibility
- `heart.csv` – Read-only; never modify

## Testing Checklist

Before submitting a PR, ensure:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Model training completes: `python scripts/train.py`
- [ ] API starts: `python api/app.py`
- [ ] No Python syntax errors
- [ ] Docstrings added for new functions
- [ ] Code follows PEP 8 style
- [ ] No hardcoded values (use config)
- [ ] Error handling included
- [ ] Edge cases considered

## Review Process

1. **Automated checks** (GitHub Actions)
   - Tests must pass
   - Linting must pass
   - Coverage must not decrease significantly

2. **Code review**
   - 1–2 approvals required
   - Address feedback constructively
   - Request re-review after changes

3. **Merge**
   - Rebase on main branch
   - Delete feature branch
   - Close related issues

## Performance Guidelines

When adding features:

| Metric | Threshold |
|--------|-----------|
| Model inference | < 10ms per prediction |
| API response | < 500ms (excluding network) |
| Frontend load | < 2s on 4G |
| Test suite | < 30s total |

## Documentation Standards

### Python Functions
```python
def train_model(X, y, continuous_cols, all_feature_names):
    """
    Train logistic regression on preprocessed data.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target variable (0/1)
        continuous_cols (list): Names of continuous features
        all_feature_names (list): All feature names after encoding
    
    Returns:
        tuple: (model, scaler, X_train, X_test, y_train, y_test, feature_names)
    
    Raises:
        ValueError: If data is invalid
    
    Example:
        >>> model, scaler, *_ = train_model(X, y, cont_cols, feat_names)
        >>> pred = model.predict(X_test)
    """
    ...
```

### Markdown Files
```markdown
## Section

Brief description.

### Subsection

### Code Example
\`\`\`python
code here
\`\`\`
```

## Reporting Issues

1. **Bug reports**
   - Describe the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version & OS
   - Stack trace (if error)

2. **Feature requests**
   - Clear description
   - Motivation/use case
   - Proposed solution
   - Alternative approaches

3. **Questions**
   - Ask in GitHub Discussions
   - Search existing issues first

## Style Guide

### Python Code

```python
# ✅ Good
def calculate_risk(probability, threshold=0.5):
    """Calculate risk level from probability."""
    return probability > threshold

# ❌ Bad
def calc_risk(p, t=0.5):
    return p > t

# ✅ Good
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

# ❌ Bad
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
```

### HTML/CSS

```html
<!-- ✅ Good -->
<button 
  class="predict-btn" 
  id="predict-btn" 
  onclick="runPrediction()"
  aria-label="Analyze clinical risk">
  Analyze
</button>

<!-- ❌ Bad -->
<button onclick="runPrediction()">Analyze</button>
```

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Docstring Conventions](https://pep257.dev/)

## Getting Help

- **Questions?** Open a GitHub Discussion
- **Stuck?** Mention `@khanaffan21` in your PR
- **Ideas?** Start an issue for discussion

---

**Thank you for contributing!** 🙏
