"""
Practical Friend Review System
=============================
Creates a simple web server that friends can access to add reviews.
Automatically pulls movies from your database.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
import webbrowser
from threading import Timer

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'src' in current_dir else current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.database import movie_db, initialize_database
except ImportError:
    sys.path.append('.')
    from models.database import movie_db, initialize_database

app = Flask(__name__)

class FriendReviewServer:
    def __init__(self):
        self.movies_cache = []
        self.reviews_collected = []
    
    async def load_popular_movies(self, limit=25):
        """Load popular movies from your database for friends to review"""
        await initialize_database()
        
        async with movie_db.pool.acquire() as conn:
            # Get movies with highest ratings and most reviews
            movies = await conn.fetch("""
                SELECT 
                    m.movie_id,
                    m.title,
                    EXTRACT(YEAR FROM m.release_date) as year,
                    m.genres::text as genres_text,
                    m.vote_average,
                    COALESCE(ur.avg_rating, 0) as user_avg_rating,
                    COALESCE(ur.rating_count, 0) as user_rating_count,
                    (m.vote_average * 0.3 + COALESCE(ur.avg_rating, 0) * 0.7) as combined_score
                FROM movies m
                LEFT JOIN (
                    SELECT 
                        movie_id,
                        AVG(rating) as avg_rating,
                        COUNT(*) as rating_count
                    FROM user_ratings 
                    GROUP BY movie_id
                ) ur ON m.movie_id = ur.movie_id
                WHERE m.vote_average > 6.0
                ORDER BY 
                    combined_score DESC,
                    user_rating_count DESC
                LIMIT $1
            """, limit)
            
            # Format for frontend
            formatted_movies = []
            for movie in movies:
                try:
                    # Parse genres
                    genres_data = json.loads(movie['genres_text']) if movie['genres_text'] else []
                    genres = [g['name'] for g in genres_data if isinstance(g, dict) and 'name' in g]
                    
                    formatted_movies.append({
                        'id': movie['movie_id'],
                        'title': movie['title'],
                        'year': int(movie['year']) if movie['year'] else 'Unknown',
                        'genres': genres[:3],  # Limit to 3 genres
                        'tmdb_rating': round(movie['vote_average'], 1),
                        'your_users_rating': round(float(movie['user_avg_rating']), 1) if movie['user_avg_rating'] else None,
                        'rating_count': movie['user_rating_count']
                    })
                except Exception as e:
                    print(f"Error processing movie {movie['title']}: {e}")
                    continue
            
            self.movies_cache = formatted_movies
            return formatted_movies

# Create server instance
review_server = FriendReviewServer()

@app.route('/')
def index():
    """Serve the review page"""
    return render_template('friend_review.html', movies=review_server.movies_cache)

@app.route('/api/movies')
def get_movies():
    """API endpoint to get movies"""
    return jsonify(review_server.movies_cache)

@app.route('/api/submit_reviews', methods=['POST'])
def submit_reviews():
    """Handle review submission"""
    try:
        data = request.json
        
        # Validate data
        if not data.get('reviewer') or not data.get('reviews'):
            return jsonify({'error': 'Missing reviewer name or reviews'}), 400
        
        # Store reviews
        review_data = {
            'reviewer': data['reviewer'],
            'timestamp': datetime.now().isoformat(),
            'reviews': data['reviews'],
            'total_reviews': len(data['reviews'])
        }
        
        review_server.reviews_collected.append(review_data)
        
        # Generate import script
        script_content = generate_import_script(review_data)
        
        # Save to file
        filename = f"import_{data['reviewer'].lower().replace(' ', '_')}_reviews.py"
        filepath = os.path.join('friend_reviews', filename)
        
        # Create directory if it doesn't exist
        os.makedirs('friend_reviews', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return jsonify({
            'success': True,
            'message': f'Reviews saved! Import script created: {filename}',
            'script_path': filepath,
            'review_count': len(data['reviews'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_script/<reviewer>')
def download_script(reviewer):
    """Download import script for a reviewer"""
    filename = f"import_{reviewer.lower().replace(' ', '_')}_reviews.py"
    return send_from_directory('friend_reviews', filename, as_attachment=True)

def generate_import_script(review_data):
    """Generate Python import script for reviews"""
    script = f'''"""
Import script for {review_data["reviewer"]}'s movie reviews
Generated on: {review_data["timestamp"]}
Total reviews: {review_data["total_reviews"]}
"""

import asyncio
import sys
import os

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.models.database import movie_db, initialize_database
except ImportError:
    sys.path.append('.')
    from models.database import movie_db, initialize_database

async def import_reviews():
    """Import {review_data["reviewer"]}'s reviews into the database"""
    print(f"🎬 Importing reviews from {review_data['reviewer']}...")
    
    try:
        # Initialize database
        await initialize_database()
        
        # Create or get user
        user = await movie_db.create_user(
            username="{review_data['reviewer'].replace(' ', '_').lower()}",
            email="{review_data['reviewer'].replace(' ', '_').lower()}@friend.local"
        )
        
        print(f"👤 User created/found: {{user['username']}}")
        
        # Review data
        reviews = {json.dumps(review_data['reviews'], indent=8)}
        
        success_count = 0
        error_count = 0
        
        for review in reviews:
            try:
                success = await movie_db.add_user_rating(
                    user_id=user['user_id'],
                    movie_id=review['movie_id'],
                    rating=float(review['rating']),
                    review_text=review.get('review') if review.get('review') else None
                )
                
                if success:
                    success_count += 1
                    print(f"✅ {{review['movie_title']}} - {{review['rating']}}/5")
                else:
                    error_count += 1
                    print(f"❌ Failed: {{review['movie_title']}}")
                    
            except Exception as e:
                error_count += 1
                print(f"❌ Error with {{review.get('movie_title', 'Unknown')}}: {{e}}")
        
        print(f"\\n🎉 Import Complete!")
        print(f"   ✅ Successful: {{success_count}}")
        print(f"   ❌ Errors: {{error_count}}")
        print(f"   📊 Total: {{len(reviews)}}")
        
    except Exception as e:
        print(f"❌ Import failed: {{e}}")
        import traceback
        traceback.print_exc()
    finally:
        await movie_db.close_connection_pool()

if __name__ == "__main__":
    asyncio.run(import_reviews())
'''
    return script

# HTML template for the review page
FRIEND_REVIEW_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎬 Movie Reviews for {{ friend_name }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .user-info {
            background: #f8fafc;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 2px solid #e2e8f0;
        }
        
        .user-info input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            margin-top: 8px;
            transition: border-color 0.2s;
        }
        
        .user-info input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .progress {
            background: #e2e8f0;
            height: 6px;
            border-radius: 3px;
            margin: 20px 0;
            overflow: hidden;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .movie-card {
            background: #fff;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .movie-card:hover {
            border-color: #667eea;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.1);
            transform: translateY(-2px);
        }
        
        .movie-card.rated {
            border-color: #10b981;
            background: #f0fdf4;
        }
        
        .movie-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }
        
        .movie-title {
            font-size: 20px;
            font-weight: bold;
            color: #1a202c;
            margin-bottom: 4px;
        }
        
        .movie-info {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        .movie-stats {
            display: flex;
            gap: 16px;
            font-size: 12px;
            color: #64748b;
        }
        
        .rating-section {
            margin-top: 20px;
        }
        
        .star-rating {
            display: flex;
            gap: 4px;
            margin: 12px 0;
        }
        
        .star {
            font-size: 28px;
            cursor: pointer;
            color: #e2e8f0;
            transition: all 0.2s ease;
            user-select: none;
        }
        
        .star:hover,
        .star.active {
            color: #fbbf24;
            transform: scale(1.1);
        }
        
        .star.active {
            text-shadow: 0 0 10px rgba(251, 191, 36, 0.5);
        }
        
        textarea {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
            min-height: 80px;
            margin-top: 10px;
            font-family: inherit;
            transition: border-color 0.2s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn:disabled {
            background: #cbd5e1;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .export-section {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #e2e8f0;
            text-align: center;
        }
        
        .success {
            color: #10b981;
            font-weight: 600;
            margin-top: 15px;
            padding: 12px;
            background: #f0fdf4;
            border-radius: 8px;
            border: 1px solid #bbf7d0;
        }
        
        .stats {
            display: flex;
            justify-content: space-around;
            background: #f8fafc;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 14px;
            color: #64748b;
            margin-top: 4px;
        }
        
        .genre-tag {
            display: inline-block;
            background: #e0e7ff;
            color: #3730a3;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Movie Review Collector</h1>
        <p class="subtitle">Help improve the movie recommendation system by rating these popular movies!</p>
        
        <div class="user-info">
            <label><strong>Your Name:</strong></label>
            <input type="text" id="userName" placeholder="Enter your name" required>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-number" id="totalMovies">{{ movies|length }}</div>
                <div class="stat-label">Movies to Rate</div>
            </div>
            <div class="stat">
                <div class="stat-number" id="completedRatings">0</div>
                <div class="stat-label">Rated</div>
            </div>
            <div class="stat">
                <div class="stat-number" id="progressPercent">0%</div>
                <div class="stat-label">Complete</div>
            </div>
        </div>
        
        <div class="progress">
            <div class="progress-bar" id="progressBar"></div>
        </div>
        
        <div id="movieList"></div>
        
        <div class="export-section">
            <button class="btn" onclick="exportData()">
                📤 Submit All Reviews
            </button>
            <div id="exportStatus"></div>
        </div>
    </div>

    <script>
        const movies = {{ movies|tojson }};
        let ratings = {};
        
        function initializeMovieList() {
            const movieList = document.getElementById('movieList');
            movieList.innerHTML = '';
            
            movies.forEach((movie, index) => {
                const movieCard = createMovieCard(movie, index);
                movieList.appendChild(movieCard);
            });
        }
        
        function createMovieCard(movie, index) {
            const card = document.createElement('div');
            card.className = 'movie-card';
            card.id = `movie-${movie.id}`;
            
            const genreTags = movie.genres.map(g => 
                `<span class="genre-tag">${g}</span>`
            ).join('');
            
            const userRatingDisplay = movie.your_users_rating 
                ? `<span>👥 Users: ${movie.your_users_rating}/10 (${movie.rating_count} ratings)</span>`
                : '<span>👥 No user ratings yet</span>';
            
            card.innerHTML = `
                <div class="movie-header">
                    <div>
                        <div class="movie-title">${movie.title}</div>
                        <div class="movie-info">${movie.year} • ${genreTags}</div>
                        <div class="movie-stats">
                            <span>⭐ TMDb: ${movie.tmdb_rating}/10</span>
                            ${userRatingDisplay}
                        </div>
                    </div>
                </div>
                <div class="rating-section">
                    <label><strong>Your Rating:</strong></label>
                    <div class="star-rating" data-movie-id="${movie.id}">
                        <span class="star" data-rating="1">★</span>
                        <span class="star" data-rating="2">★</span>
                        <span class="star" data-rating="3">★</span>
                        <span class="star" data-rating="4">★</span>
                        <span class="star" data-rating="5">★</span>
                    </div>
                    <textarea placeholder="Optional: Write a review or comment about this movie..." id="review-${movie.id}"></textarea>
                </div>
            `;
            
            // Add star rating functionality
            const stars = card.querySelectorAll('.star');
            stars.forEach(star => {
                star.addEventListener('click', function() {
                    const rating = parseInt(this.dataset.rating);
                    const movieId = this.parentElement.dataset.movieId;
                    
                    // Update visual state
                    stars.forEach((s, index) => {
                        if (index < rating) {
                            s.classList.add('active');
                        } else {
                            s.classList.remove('active');
                        }
                    });
                    
                    // Store rating
                    if (!ratings[movieId]) ratings[movieId] = {};
                    ratings[movieId].rating = rating;
                    ratings[movieId].movieTitle = movie.title;
                    
                    // Update card appearance
                    card.classList.add('rated');
                    
                    updateProgress();
                });
            });
            
            return card;
        }
        
        function updateProgress() {
            const totalMovies = movies.length;
            const ratedMovies = Object.keys(ratings).length;
            const percentage = Math.round((ratedMovies / totalMovies) * 100);
            
            document.getElementById('progressBar').style.width = percentage + '%';
            document.getElementById('completedRatings').textContent = ratedMovies;
            document.getElementById('progressPercent').textContent = percentage + '%';
        }
        
        async function exportData() {
            const userName = document.getElementById('userName').value.trim();
            if (!userName) {
                alert('Please enter your name first!');
                return;
            }
            
            if (Object.keys(ratings).length === 0) {
                alert('Please rate at least one movie before submitting!');
                return;
            }
            
            // Collect all data
            const reviewData = {
                reviewer: userName,
                reviews: []
            };
            
            Object.keys(ratings).forEach(movieId => {
                const reviewText = document.getElementById(`review-${movieId}`).value.trim();
                reviewData.reviews.push({
                    movie_id: parseInt(movieId),
                    movie_title: ratings[movieId].movieTitle,
                    rating: ratings[movieId].rating,
                    review: reviewText || null
                });
            });
            
            try {
                // Submit to server
                const response = await fetch('/api/submit_reviews', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(reviewData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('exportStatus').innerHTML = `
                        <div class="success">
                            <p>✅ ${result.message}</p>
                            <p>📊 Reviews submitted: ${result.review_count}</p>
                            <p>🎯 The recommendation system will be updated with your ratings!</p>
                        </div>
                    `;
                } else {
                    alert('Error: ' + result.error);
                }
                
            } catch (error) {
                alert('Error submitting reviews: ' + error.message);
            }
        }
        
        // Initialize on load
        initializeMovieList();
    </script>
</body>
</html>'''

def save_html_template():
    """Save the HTML template"""
    template_dir = 'templates'
    os.makedirs(template_dir, exist_ok=True)
    
    with open(f'{template_dir}/friend_review.html', 'w', encoding='utf-8') as f:
        f.write(FRIEND_REVIEW_HTML)

async def start_server(host='localhost', port=5000):
    """Start the review server"""
    print("🎬 Starting Friend Review Server...")
    print("=" * 50)
    
    # Load movies from database
    print("📊 Loading movies from your database...")
    await review_server.load_popular_movies(25)
    print(f"✅ Loaded {len(review_server.movies_cache)} popular movies")
    
    # Save HTML template
    save_html_template()
    
    # Start server
    print(f"\n🌐 Server starting at: http://{host}:{port}")
    print("\n📱 Share this URL with friends:")
    print(f"   http://{host}:{port}")
    print("\n💡 Instructions for friends:")
    print("   1. Open the URL in their browser")
    print("   2. Enter their name")
    print("   3. Rate movies they've seen")
    print("   4. Submit reviews")
    print("   5. Send you the generated import script")
    
    print("\n🔄 To import friend reviews:")
    print("   Run the generated Python scripts in friend_reviews/ folder")
    
    # Auto-open browser
    Timer(1.5, lambda: webbrowser.open(f'http://{host}:{port}')).start()
    
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start Friend Review Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    
    args = parser.parse_args()
    
    asyncio.run(start_server(args.host, args.port))