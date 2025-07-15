AI-Powered Movie Discovery MCP Server
This project is an advanced, AI-driven movie discovery and recommendation engine built as a Multi-Capability Protocol (MCP) server. It integrates a hybrid recommendation model, natural language processing (NLP) for sentiment analysis and mood-based searching, and a robust PostgreSQL backend to deliver a comprehensive and interactive movie discovery experience.

The server is designed to be used with an agent like Claude, exposing its various machine learning and data processing capabilities as distinct, callable tools.

Key Features
Hybrid Recommendation System: Combines collaborative filtering (finding similar users based on rating correlations) and content-based filtering (recommending movies with similar genres, keywords, etc.) to provide nuanced and accurate suggestions.

NLP-Powered Mood Search: Allows users to find movies using natural language queries like "find me a funny movie for tonight" or "something thrilling but not too scary."

Sentiment Analysis: Automatically analyzes the sentiment of user-provided reviews (positive, negative, neutral) using textblob and extracts key aspects discussed (e.g., acting, plot, visuals).

Comprehensive Database: Utilizes a PostgreSQL database to store detailed movie information, user data, ratings, and activity logs, all fetched and updated via the TMDB API.

User Analytics: Provides endpoints to analyze and summarize a user's viewing history, genre preferences, and rating habits.

MCP Server Architecture: Exposes all core functionalities as tools that can be listed and called by an MCP-compatible agent, enabling complex, conversational interactions.

Tech Stack
Backend: Python, Asyncio

Protocol: Multi-Capability Protocol (MCP)

Database: PostgreSQL with asyncpg

Machine Learning:

NLP: textblob

Data & ML Ops: pandas, scikit-learn

API: Integration with The Movie Database (TMDB) API

Setup and Installation
Clone the repository:

git clone https://github.com/your-username/AI-Movie-Discovery-Server.git
cd AI-Movie-Discovery-Server

Create and activate a virtual environment:

python -m venv movie_venv
source movie_venv/bin/activate  # On Windows, use `movie_venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Set up NLP components:
This will download the necessary corpora for textblob.

python setup_nlp.py

Set up the database:
This project uses PostgreSQL. Make sure you have it installed and running on your local machine.

Configure environment variables:
Create a .env file in the root directory and add your credentials. Do not commit this file to GitHub.

DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DATABASE_NAME"
TMDB_API_KEY="your_tmdb_api_key_here"

Run the database setup script:
To create all the necessary tables and indexes in your PostgreSQL database, run the setup script.

python setup_database.py

How to Run the Server
To start the MCP server, run the main server.py file from the root directory:

python server.py

The server will now be running and ready to accept connections from an MCP client or agent. You can interact with its tools, such as search_movies, get_personalized_recommendations, and mood_based_search.