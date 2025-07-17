# 🎬 AI-Powered Movie Discovery MCP Server

An intelligent movie recommendation system that integrates with Claude Desktop via the Model Context Protocol (MCP). Features hybrid machine learning algorithms, natural language processing, and collaborative filtering to deliver personalized movie recommendations.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/postgresql-13+-blue.svg)
![MCP](https://img.shields.io/badge/MCP-Protocol-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 🤖 AI-Powered Recommendations
- **Hybrid ML System**: Combines collaborative filtering with content-based filtering
- **User Similarity Analysis**: Finds users with similar tastes using correlation algorithms
- **Personalized Scoring**: Predicts ratings based on viewing history and preferences

### 🧠 Natural Language Processing
- **Mood-Based Search**: Understands queries like "funny action movie for tonight"
- **Sentiment Analysis**: Analyzes user reviews using TextBlob for emotional insights
- **Intent Recognition**: Maps natural language to genre preferences and filters

### 👥 Social Features
- **Friend Review System**: Web interface for friends to rate and review movies
- **Collaborative Filtering**: Leverages social data for better recommendations
- **User Analytics**: Detailed preference analysis and viewing pattern insights

### 🎯 Claude Desktop Integration
- **MCP Protocol**: Seamless integration with Claude Desktop
- **8 Interactive Tools**: Search, recommend, rate, analyze preferences, and more
- **Persistent Memory**: Remembers user preferences across conversations

## 🛠️ Tech Stack

- **Backend**: Python 3.12, AsyncIO, PostgreSQL
- **ML/NLP**: TextBlob, Collaborative Filtering, Content-Based Filtering
- **APIs**: TMDB (The Movie Database) for real-time movie data
- **Database**: PostgreSQL with asyncpg for async operations
- **Web Interface**: Flask for friend review collection
- **Protocol**: Model Context Protocol (MCP) for Claude Desktop integration

## 📊 Database Schema

- **Movies**: 1100+ films with TMDB metadata, genres, ratings
- **Users**: User profiles with preferences and viewing history  
- **Ratings**: 3000+ user ratings with optional reviews
- **Analytics**: User activity logs and recommendation tracking

## 🚀 Installation

### Prerequisites
- Python 3.12+
- PostgreSQL 13+
- TMDB API Key
- Claude Desktop (for MCP integration)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-discovery-mcp.git
   cd movie-discovery-mcp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv movie_venv
   source movie_venv/bin/activate  # On Windows: movie_venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your TMDB API key and database URL
   ```

5. **Set up database**
   ```bash
   createdb movie_discovery
   python setup_database.py
   ```

6. **Populate with movie data**
   ```bash
   python fix_all_genres.py  # Fetch movie data from TMDB
   ```

## 🎮 Usage

### Start MCP Server
```bash
python -m src.server
```

### Configure Claude Desktop
Add to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "movie-discovery": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/movie-discovery-mcp"
    }
  }
}
```

### Example Commands in Claude Desktop

**Get Personalized Recommendations:**
```
Get movie recommendations for Lakshya
```

**Mood-Based Search:**
```
Find me a funny action movie with comedy and adventure
```

**Rate a Movie:**
```
Rate movie ID 299534 as 5 stars with review "Amazing superhero epic!"
```

**Analyze Preferences:**
```
Analyze Lakshya's viewing preferences
```

## 🔧 Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_movies` | Search for movies using TMDB API | query, limit |
| `get_personalized_recommendations` | AI-powered personalized suggestions | username, limit |
| `mood_based_search` | Natural language movie search | query, username, limit |
| `rate_movie` | Add rating and review | username, movie_id, rating, review |
| `analyze_user_preferences` | User preference analytics | username |
| `analyze_movie_sentiment` | Sentiment analysis of reviews | movie_id |
| `get_trending_movies` | Current trending films | time_period, limit |
| `find_similar_movies` | Content-based similarity search | movie_id, limit |

## 🌐 Friend Review System

Start the web interface for collecting friend reviews:

```bash
python practical_friend_reviewer.py
```

Friends can visit `http://localhost:5000` to:
- Rate movies from your database
- Write detailed reviews
- Generate automatic import scripts
- Contribute to collaborative filtering

## 📈 Machine Learning Algorithms

### Collaborative Filtering
- **User-Based**: Finds similar users based on rating patterns
- **Pearson Correlation**: Measures user similarity 
- **Prediction Algorithm**: Weighted average of similar users' ratings

### Content-Based Filtering
- **Genre Matching**: Recommends based on preferred genres
- **Metadata Analysis**: Uses director, cast, keywords
- **TF-IDF Similarity**: Text-based content matching

### Hybrid Approach
- **Weighted Combination**: Balances collaborative and content methods
- **Cold Start Solution**: Uses content-based for new users
- **Popularity Fallback**: Trending movies for sparse data

## 🧪 Daily Updates

Automatically fetch new movies and maintain fresh data:

```bash
python daily_movie_updater.py
```

This script:
- Fetches trending movies from TMDB
- Updates existing movie metadata
- Respects API rate limits (1000 requests/day)
- Maintains data quality and consistency

## 📁 Project Structure

```
movie-discovery-mcp/
├── src/
│   ├── server.py              # Main MCP server
│   ├── models/
│   │   ├── database.py        # Database models and operations
│   │   └── __init__.py
│   └── __init__.py
├── templates/
│   └── friend_review.html     # Friend review interface
├── friend_reviews/            # Generated import scripts
├── practical_friend_reviewer.py  # Web server for reviews
├── daily_movie_updater.py     # Automated data updates
├── fix_all_genres.py          # Genre data maintenance
├── requirements.txt           # Python dependencies
├── .env.example              # Environment configuration template
└── README.md                 # This file
```

## 🎯 Recommendation Algorithm Flow

1. **User Profile Analysis**: Extract genre preferences, rating patterns
2. **Similarity Calculation**: Find users with similar tastes using correlation
3. **Candidate Generation**: Collect highly-rated movies from similar users
4. **Content Filtering**: Apply genre preferences and metadata matching
5. **Scoring & Ranking**: Combine collaborative and content-based scores
6. **Diversity Enhancement**: Ensure variety in final recommendations

## 🔮 Future Enhancements

- **Neural Collaborative Filtering**: Deep learning recommendation models
- **Movie Embeddings**: Vector representations using Word2Vec/FastText
- **Advanced NLP**: Integration with spaCy or Transformers for better understanding
- **Real-time Learning**: Online learning algorithms for immediate preference updates
- **Streaming Integration**: API connections to Netflix, Hulu, Disney+ for availability
- **Social Features**: User profiles, friend networks, movie discussions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TMDB** for comprehensive movie database and API
- **Anthropic** for Claude Desktop and MCP protocol
- **PostgreSQL** for robust data storage
- **TextBlob** for natural language processing capabilities

## Contact

Lakshya Bhatia - [lakshya14wrk@gmail.com](mailto:lakshya14wrk@gmail.com)

Project Link: [https://github.com/yourusername/movie-discovery-mcp](https://github.com/Bokuto14/movie-discovery-mcp)

---
