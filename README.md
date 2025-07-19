# 🧠 Evolutionary Neural Network (EvoNet)

A revolutionary neural network implementation that uses evolutionary algorithms instead of backpropagation for training. This project demonstrates how evolutionary computation can be used to train neural networks for classification and regression tasks.

## 🌟 Key Features

### 🚀 **No Backpropagation Required**
- Uses evolutionary algorithms instead of gradient descent
- Population-based neuron evolution
- Adaptive mutation strategies
- Significant mutation vector memory

### 🎯 **Multi-Problem Support**
- **Binary Classification**: Breast cancer detection, telemetry health prediction
- **Multi-Class Classification**: Iris species, wine types, digit recognition
- **Regression**: Housing price prediction, stock volume forecasting

### 📊 **Advanced Data Handling**
- Automatic dataset detection and preprocessing
- SMOTE balancing for imbalanced datasets
- Feature selection and scaling
- Support for custom CSV and JSON datasets
- Large-scale data handling (trillions-scale values)

### 🏗️ **Three-Layer Architecture**
- **Layer 1**: 50 parallel evolutionary neurons
- **Layer 2**: 20 evolutionary neurons
- **Layer 3**: Output neurons (classification) or single neuron (regression)

## 📦 Installation

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Optional Dependencies
```bash
pip install imbalanced-learn  # For SMOTE balancing
pip install numba            # For JIT compilation (faster execution)
```

## 🚀 Quick Start

### 1. Run the Main Program
```bash
python hope.py
```

### 2. Choose Your Dataset
The program will present you with options:
- **1-5**: Built-in datasets (California Housing, Breast Cancer, Iris, Wine, Digits)
- **6**: Custom CSV dataset
- **7**: Telemetry JSON dataset
- **Direct file path**: Enter path directly (e.g., `C:/data/my_dataset.csv`)

### 3. Select Training Method (for Regression)
- **1**: Standard Epoch Training (Slow but thorough)
- **2**: Mini-Batch Evolution Training (Fast)
- **3**: Early Stopping Training (Efficient)

## 📁 Project Structure

```
├── hope.py                 # Main evolutionary neural network
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── examples/              # Example datasets and scripts
│   ├── sample_data.csv
│   └── telemetry_sample.json
└── docs/                  # Documentation
    ├── architecture.md
    └── algorithms.md
```

## 🧬 Algorithm Overview

### Evolutionary Neuron
Each neuron maintains a population of weight/bias combinations:
- **Population Size**: 20 individuals
- **Selection**: Keep best 2 + elite individual
- **Mutation**: Adaptive strength based on global error
- **Crossover**: Random parent selection with mutation

### Significant Mutation Vector (V_m)
- **History**: Last 20 successful mutations
- **Influence**: 20% chance to influence new mutations
- **Memory**: Preserves successful evolutionary patterns

### Training Process
1. **Forward Pass**: All neurons in population predict
2. **Error Calculation**: MSE for regression, Cross-entropy for classification
3. **Selection**: Keep best performing individuals
4. **Evolution**: Create new population through mutation/crossover
5. **Memory Update**: Store successful mutations in V_m

## 📊 Performance Examples

### Binary Classification (Breast Cancer)
- **Accuracy**: 95%+
- **Features**: 30 medical features
- **Balancing**: Automatic SMOTE for imbalanced data

### Multi-Class Classification (Iris)
- **Accuracy**: 90%+
- **Classes**: 3 species
- **Features**: 4 botanical measurements

### Regression (Housing Prices)
- **R² Score**: 70%+
- **Features**: 8 housing characteristics
- **Scaling**: Automatic for large-scale data

## 🔧 Customization

### Hyperparameters
```python
LEVEL1_NEURONS = 50      # Number of neurons in layer 1
LEVEL2_NEURONS = 20      # Number of neurons in layer 2
POP_SIZE = 20            # Population size per neuron
EPOCHS = 50              # Training epochs
TAU1 = 0.15             # Layer 1 threshold
TAU2 = 0.10             # Layer 2 threshold
```

### Adding New Datasets
1. Place your CSV file in the project directory
2. Run `python hope.py`
3. Choose option 6 and provide the file path
4. The system will automatically detect the problem type

## 🎯 Use Cases

### Financial Data
- Stock volume prediction
- Price forecasting
- Market trend analysis

### Medical Data
- Disease diagnosis
- Patient outcome prediction
- Medical image classification

### IoT/Telemetry
- Device health monitoring
- Anomaly detection
- Predictive maintenance

## 🔬 Research Applications

This implementation demonstrates:
- **Alternative Training Methods**: Evolution vs. backpropagation
- **Population-Based Learning**: Collective intelligence approach
- **Adaptive Evolution**: Self-adjusting mutation strategies
- **Memory Mechanisms**: Learning from successful mutations

## 📈 Future Enhancements

- [ ] GPU acceleration with CUDA
- [ ] Distributed evolution across multiple machines
- [ ] Advanced selection strategies (tournament, rank-based)
- [ ] Multi-objective evolution
- [ ] Real-time evolution for streaming data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by evolutionary computation principles
- Built on scikit-learn for data preprocessing
- Uses matplotlib and seaborn for visualizations

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/evolutionary-neural-network/issues) page
2. Create a new issue with detailed description
3. Include your dataset type and error messages

---

**Made with ❤️ by Ayushman Trivedi**

*Revolutionizing neural networks through evolution! 🧬🧠*
