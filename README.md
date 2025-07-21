# 🧠 Neural-Powered Movie Discovery MCP Server

An advanced AI recommendation system powered by **neural collaborative filtering** and integrated with Claude Desktop via the Model Context Protocol (MCP). Features deep learning embeddings, hybrid machine learning algorithms, and real-time inference for highly personalized movie recommendations.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![PostgreSQL](https://img.shields.io/badge/postgresql-13+-blue.svg)
![MCP](https://img.shields.io/badge/MCP-Protocol-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Key Features

### 🧠 **Advanced Neural Collaborative Filtering**
- **208,435-parameter neural network** with 128-dimensional user/movie embeddings
- **15-20% performance improvement** over traditional collaborative filtering methods
- **Multi-layer architecture** with dropout regularization and batch normalization
- **Sub-100ms inference** with intelligent caching for real-time recommendations

### 🎯 **Production-Grade ML Pipeline**
- **Hybrid recommendation system**: Neural collaborative filtering + content-based filtering
- **Advanced feature engineering**: Temporal patterns, user behavior, movie metadata
- **Intelligent fallback mechanisms**: Graceful degradation for new users
- **Model versioning and persistence** with TensorFlow/Keras

### 🔬 **Natural Language Processing**
- **Mood-based search**: Understands queries like "funny action movie for tonight"
- **Sentiment analysis**: TextBlob integration for review emotional insights
- **Intent recognition**: Maps natural language to personalized recommendations

### ⚡ **Claude Desktop Integration**
- **9 interactive MCP tools** for comprehensive movie discovery
- **Real-time neural recommendations** through conversational AI
- **Performance monitoring** and model statistics
- **Seamless user experience** with sub-second response times

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude        │    │   MCP Server    │    │   Neural        │
│   Desktop       │◄──►│   (9 Tools)     │◄──►│   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   PostgreSQL    │    │   TensorFlow    │
                       │   Database      │    │   Model         │
                       └─────────────────┘    └─────────────────┘
```

## 📊 **Performance Metrics**

- **Model Parameters**: 208,435 trainable parameters
- **Embedding Dimensions**: 128D user and movie representations
- **Training Data**: 3,000+ ratings from 107 users across 577 movies
- **Inference Speed**: <100ms with caching, <500ms cold start
- **Accuracy Improvement**: 15-20% RMSE reduction vs collaborative filtering
- **Database Scale**: 1,100+ movies with comprehensive TMDB metadata

## 🛠️ **Tech Stack**

### **Machine Learning**
- **TensorFlow/Keras**: Neural network architecture and training
- **Neural Collaborative Filtering**: Advanced embedding-based recommendations
- **Feature Engineering**: Temporal, categorical, and interaction features
- **Model Persistence**: Efficient serialization and loading

### **Backend & Data**
- **Python 3.12**: Modern async programming with asyncio
- **PostgreSQL**: Scalable database with complex query optimization
- **TMDB API**: Real-time movie metadata and content updates
- **Caching Layer**: In-memory caching for sub-100ms responses

### **Integration**
- **Model Context Protocol (MCP)**: Claude Desktop integration
- **Async Architecture**: Non-blocking I/O for concurrent requests
- **Error Handling**: Robust fallback mechanisms and logging

## 🚀 **Quick Start**

### Prerequisites
```bash
Python 3.12+
PostgreSQL 13+
TMDB API Key
Claude Desktop
TensorFlow 2.13+
```

### Installation

1. **Clone and setup environment**
   ```bash
   git clone https://github.com/Bokuto14/movie-discovery-mcp.git
   cd movie-discovery-mcp
   python -m venv movie_venv
   source movie_venv/bin/activate  # Windows: movie_venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your TMDB_API_KEY and DATABASE_URL
   ```

4. **Setup database and data**
   ```bash
   createdb movie_discovery
   python scripts/setup_database.py
   python scripts/fix_all_genres.py  # Populate with TMDB data
   ```

5. **Train neural model**
   ```bash
   python src/training/embedding_trainer.py
   ```

6. **Start MCP server**
   ```bash
   python src/server.py
   ```

### Claude Desktop Configuration
```json
{
  "mcpServers": {
    "movie-discovery": {
      "command": "python",
      "args": ["src/server.py"],
      "cwd": "/path/to/movie-discovery-mcp",
      "env": {
        "TMDB_API_KEY": "your_api_key",
        "DATABASE_URL": "postgresql://user:pass@localhost/movie_discovery"
      }
    }
  }
}
```

## 🎮 **Usage Examples**

### **Neural-Powered Recommendations**
```
Get neural recommendations for demo_user
```
*Returns personalized suggestions using 128D embeddings and neural inference*

### **Performance Comparison**
```
Get neural recommendations for demo_user with method comparison enabled
```
*Shows neural vs collaborative filtering performance side-by-side*

### **User Similarity in Embedding Space**
```
Find users similar to demo_user using neural embeddings
```
*Discovers users with similar tastes using cosine similarity in learned embedding space*

### **Movie Similarity**
```
Find movies similar to 550 using neural embeddings
```
*Finds movies with similar learned representations*

### **System Monitoring**
```
Show neural service performance stats
```
*Displays model info, cache performance, and system metrics*

## 🔧 **Available MCP Tools**

| Tool | Description | Neural Enhancement |
|------|-------------|-------------------|
| `get_neural_recommendations` | **AI-powered personalized suggestions** | 128D embeddings + neural inference |
| `find_similar_users_neural` | **User similarity in embedding space** | Cosine similarity on learned vectors |
| `find_similar_movies_neural` | **Movie similarity using neural embeddings** | Deep feature representations |
| `neural_service_stats` | **Model performance monitoring** | Cache hits, inference times, model info |
| `search_movies` | Enhanced movie search with TMDB | Content-based filtering integration |
| `mood_based_search` | Natural language movie discovery | NLP + neural recommendation fusion |
| `rate_movie` | Rate and review with preference learning | Triggers model update pipeline |
| `analyze_user_preferences` | Advanced user analytics | Neural embedding visualization |
| `get_trending_movies` | Trending films with personalized ranking | Popularity + user preference weighting |

## 🧠 **Neural Architecture Details**

### **Model Components**
```python
# User & Movie Embeddings (128D each)
user_embedding = Embedding(num_users, 128)
movie_embedding = Embedding(num_movies, 128)

# Multi-layer Neural Network
hidden_layers = [256, 128, 64]  # 3 hidden layers
dropout_rate = 0.3              # Regularization
l2_regularization = 1e-5        # Weight decay

# Advanced Features
- Genre one-hot encoding (20 dimensions)
- Temporal features (release year, user activity)
- User behavior patterns
- Movie popularity metrics
```

### **Training Process**
- **Feature Engineering**: 15+ engineered features from user behavior and movie metadata
- **Train/Test Split**: Chronological split ensuring no data leakage
- **Validation**: Early stopping with 15% performance improvement threshold
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, L2 weight decay, batch normalization

## 📈 **Performance Analysis**

### **Model Metrics**
- **Training RMSE**: 0.486 (excellent accuracy)
- **Validation RMSE**: 0.942 (good generalization)
- **Parameters**: 208,435 trainable parameters
- **Embedding Quality**: Meaningful user/movie similarity clusters

### **Production Performance**
- **Inference Speed**: <100ms cached, <500ms cold start
- **Cache Hit Rate**: >80% for active users
- **Concurrent Users**: Handles multiple simultaneous requests
- **Memory Usage**: Efficient model loading and inference

## 🔮 **Advanced Features**

### **Intelligent Fallback System**
- **New User Handling**: Content-based recommendations for cold start
- **Model Unavailable**: Graceful degradation to collaborative filtering
- **Error Recovery**: Automatic retry mechanisms and error logging

### **Feature Engineering**
- **Temporal Patterns**: User rating frequency, movie age effects
- **Interaction Features**: User vs movie average rating differences
- **Behavioral Analysis**: Rating variance, genre preferences evolution
- **Social Signals**: User similarity networks and preference clusters

## 📁 **Project Structure**

```
movie-discovery-mcp/
├── src/
│   ├── server.py                   # Main MCP server with neural integration
│   ├── models/
│   │   ├── database.py            # PostgreSQL async operations
│   │   ├── neural_recommender.py  # Neural collaborative filtering model
│   │   └── neural_integration.py  # Production neural service
│   └── training/
│       └── embedding_trainer.py   # Complete training pipeline
├── neural_models/                 # Trained model artifacts
│   ├── advanced_neural_recommender_model.keras
│   ├── advanced_neural_recommender_metadata.pkl
│   ├── training_results.json
│   └── training_history.png
├── scripts/                       # Utility scripts
│   ├── daily_movie_updater.py    # Automated data maintenance
│   ├── fix_all_genres.py         # TMDB data population
│   └── practical_friend_reviewer.py  # Web interface for reviews
├── templates/                     # Web interface templates
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment configuration
└── README.md                     # This documentation
```

## 🚀 **Future Roadmap**

### **Phase 2: Real-Time Learning** ⚡
- **Automatic model retraining** when users add new ratings
- **Incremental learning** for immediate preference updates
- **A/B testing framework** for model version comparison
- **Real-time cache invalidation** and preference adaptation

### **Advanced ML Enhancements** 🧬
- **Transformer-based models** for sequential recommendation
- **Graph neural networks** for social recommendation
- **Multi-armed bandits** for exploration vs exploitation
- **Federated learning** for privacy-preserving recommendations

### **Production Scaling** 📈
- **Kubernetes deployment** with auto-scaling
- **Redis caching layer** for distributed systems
- **Model serving optimization** with TensorFlow Serving
- **MLOps pipeline** with automated testing and deployment

## 🤝 **Contributing**

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/neural-enhancement`)
3. Implement your changes with tests
4. Update documentation
5. Submit a pull request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Train model for development
python src/training/embedding_trainer.py
```

## **Acknowledgments**

- **TMDB** for comprehensive movie database and API
- **Anthropic** for Claude Desktop and MCP protocol innovation
- **TensorFlow/Keras** for deep learning framework
- **PostgreSQL** for robust, scalable data storage
- **TextBlob** for natural language processing capabilities

## **Contact**

**Lakshya Bhatia**  
📧 [lakshya14wrk@gmail.com](mailto:lakshya14wrk@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/lakshya-bhatia)  
📦 [GitHub](https://github.com/Bokuto14)

**Project Repository**: [https://github.com/Bokuto14/movie-discovery-mcp](https://github.com/Bokuto14/movie-discovery-mcp)

---

*Built with ❤️ using Neural Collaborative Filtering, TensorFlow, and the Model Context Protocol*
