import pandas as pd

# # Load dataset from CSV
# csv_path = '/content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
# recipes_df = pd.read_csv(csv_path)

# # Count the occurrences of each title
# title_counts = recipes_df['Title'].value_counts()

# unique_titles_count = len(recipes_df['Title'].unique())
# print(f"Total number of Dish titles: {unique_titles_count}")






import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# # Load dataset from CSV
# csv_path = './content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
# recipes_df = pd.read_csv(csv_path)

# # Use TF-IDF to vectorize ingredients
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(recipes_df['Ingredients'])

# # Compute cosine similarity between recipes
# cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend recipes based on user-provided ingredients
def recommend_recipes(user_ingredients, recipes_df, tfidf_vectorizer, cosine_similarities,tfidf_matrix):
    # Add user's recipe to the DataFrame
    user_recipe = {'Title': 'User Recipe', 'Ingredients': user_ingredients}
    recipes_df = recipes_df.append(user_recipe, ignore_index=True)

    # Use TF-IDF to vectorize ingredients
    user_tfidf_matrix = tfidf_vectorizer.transform([user_ingredients])

    # Compute cosine similarity between the user's recipe and all recipes
    user_cosine_similarities = linear_kernel(user_tfidf_matrix, tfidf_matrix).flatten()

    # Sort recipes based on similarity scores
    similarity_scores = list(enumerate(user_cosine_similarities))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Recommend top 3 recipes (excluding the user's recipe)
    top_recipes_indices = [index for index, _ in similarity_scores[1:4]]
    recommended_recipes = recipes_df.iloc[top_recipes_indices]['Title'].tolist()

    return recommended_recipes





# # Example of recommending recipes for user-provided ingredients
# user_ingredients = 'Cauliflower'
# recommended_recipes = recommend_recipes(user_ingredients, recipes_df, tfidf_vectorizer, cosine_similarities)

# # Print the recommended recipes
# print("Recommended Recipes:")
# for recipe in recommended_recipes:
#     print("-", recipe)







