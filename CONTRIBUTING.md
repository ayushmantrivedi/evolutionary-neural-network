# 🤝 Contributing to Evolutionary Neural Network

Thank you for your interest in contributing to the Evolutionary Neural Network project! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/evolutionary-neural-network.git`
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Make** your changes
5. **Test** your changes
6. **Commit** your changes: `git commit -m 'Add amazing feature'`
7. **Push** to your branch: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

## 📋 Development Setup

### Prerequisites
```bash
# Install Python 3.7+
python --version

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hope

# Run specific test file
pytest tests/test_evolution.py
```

### Code Quality
```bash
# Format code
black hope.py

# Lint code
flake8 hope.py

# Type checking
mypy hope.py
```

## 🎯 Areas for Contribution

### High Priority
- [ ] **Performance Optimization**: Improve training speed and memory efficiency
- [ ] **GPU Support**: Add CUDA acceleration for large datasets
- [ ] **Distributed Training**: Multi-machine evolution support
- [ ] **Advanced Selection**: Tournament, rank-based, and other selection strategies
- [ ] **Multi-objective Evolution**: Handle multiple fitness objectives

### Medium Priority
- [ ] **Real-time Evolution**: Streaming data support
- [ ] **Hyperparameter Tuning**: Automatic optimization of evolutionary parameters
- [ ] **Visualization Tools**: Training progress and population dynamics
- [ ] **Model Persistence**: Save/load trained models
- [ ] **API Interface**: Clean API for easy integration

### Low Priority
- [ ] **Documentation**: Improve docstrings and tutorials
- [ ] **Examples**: More dataset examples and use cases
- [ ] **Benchmarks**: Compare with other evolutionary algorithms
- [ ] **Web Interface**: Simple web UI for experimentation

## 📝 Code Style Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible

### Documentation
- Update README.md for new features
- Add docstrings to new functions
- Include examples in docstrings
- Update architecture.md for structural changes

### Testing
- Write tests for new functionality
- Maintain >80% code coverage
- Test edge cases and error conditions
- Use descriptive test names

## 🧪 Testing Guidelines

### Unit Tests
```python
def test_evolutionary_neuron_initialization():
    """Test that evolutionary neuron initializes correctly."""
    neuron = EvoNeuron(input_dim=10)
    assert len(neuron.population) == 20
    assert all('weights' in ind for ind in neuron.population)
```

### Integration Tests
```python
def test_full_training_pipeline():
    """Test complete training pipeline with sample data."""
    X, y = create_sample_data()
    model = MultiClassEvoNet(input_dim=X.shape[1], num_classes=3)
    model.train(X, y, epochs=5)
    accuracy, loss = model.evaluate(X, y)
    assert accuracy > 0.5  # Should perform better than random
```

### Performance Tests
```python
def test_training_speed():
    """Test that training completes within reasonable time."""
    import time
    start_time = time.time()
    # Run training
    training_time = time.time() - start_time
    assert training_time < 60  # Should complete within 60 seconds
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: Python version, OS, dependency versions
2. **Reproduction**: Steps to reproduce the issue
3. **Expected vs Actual**: What you expected vs what happened
4. **Error Messages**: Full error traceback
5. **Data**: Sample data that causes the issue (if applicable)

### Bug Report Template
```markdown
**Bug Description**
Brief description of the issue.

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happened.

**Environment**
- Python: 3.9.7
- OS: Windows 10
- Dependencies: [list versions]

**Additional Information**
Any other relevant information.
```

## 💡 Feature Requests

When requesting features, please include:

1. **Use Case**: Why this feature is needed
2. **Proposed Solution**: How you think it should work
3. **Alternatives**: Other approaches you've considered
4. **Impact**: How it affects existing functionality

### Feature Request Template
```markdown
**Feature Description**
Brief description of the requested feature.

**Use Case**
Why this feature is needed and how it would be used.

**Proposed Solution**
How you think the feature should be implemented.

**Alternatives Considered**
Other approaches you've thought about.

**Additional Context**
Any other relevant information.
```

## 🔄 Pull Request Process

### Before Submitting
1. **Test**: Ensure all tests pass
2. **Format**: Run black and flake8
3. **Document**: Update documentation if needed
4. **Rebase**: Keep commits clean and focused

### Pull Request Template
```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass

**Documentation**
- [ ] README updated
- [ ] Docstrings added
- [ ] Architecture docs updated

**Additional Notes**
Any other information.
```

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Email**: Contact maintainers directly for sensitive issues

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the future of evolutionary neural networks! 🧬🧠**
