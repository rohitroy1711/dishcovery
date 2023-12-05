import h5py
from bson import ObjectId
import gridfs
from flask import Flask, request,Response, render_template, session, redirect
from flask_pymongo import PyMongo
import pymongo
from Testing_model import predict
from Ingredients_to_reciepe import recommend_recipes
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
from flask import request
from PIL import Image
import io
import tensorflow as tf
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"]="1"

my_collections = pymongo.MongoClient("mongodb://localhost:27017/")
my_db = my_collections['Dishcovery']
user_col = my_db['User']

app = Flask(__name__)
app.secret_key = "Dishcovery"
app.config['MONGO_URI']='mongodb://localhost:27017/Dishcovery.images'
mongo=PyMongo(app)


@app.route("/")
def userLogin():
    return render_template("/userLogin.html")


@app.route("/userLogin1", methods=['post'])
def userLogin1():
    email = request.form.get('email')
    password = request.form.get('password')
    print(email,password)
    query = {"email": email, "password": password}
    count = user_col.count_documents(query)
    if count > 0:
        user = user_col.find_one(query)
        session['user_id'] = str(user['_id'])
        session['role'] = 'User'
        return redirect("/home")
    else:
        return render_template("userLogin.html", message="Invalid Login Details",color="red")



@app.route("/userRegister")
def userRegister():
    return render_template("/userRegister.html")


@app.route("/userRegister1", methods=['post'])
def userRegister1():
    fname = request.form.get('fname')
    lname = request.form.get('lname')
    email = request.form.get('email')
    password = request.form.get('password')
    query = {"email": email}
    count = user_col.count_documents(query)
    if count > 0:
        return render_template("userRegister.html", message="Duplicate Details!!!.....", color="red")
    query = {"FirstName": fname, "LastName": lname, "email": email, "password": password}
    result = user_col.insert_one(query)
    return render_template("userLogin.html", message="User Registered successfully", color="green")


@app.route("/home")
def userHome():
    return render_template("userHome.html")


@app.route("/fileupload", methods=['post'])
def fileupload():
    img=request.files['imageupload']
    print("******************************************************")
    image = Image.open(io.BytesIO(img.read()))
    
    # Convert the PIL image to the correct format
    image = image.resize((64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    # Now call the predict function with the processed image
    result = predict(input_arr)
    print(result)
    #file_id=mongo.save_file(img.filename,img)
    print("******************************************************")
    # Load dataset from CSV
    csv_path = './content/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    recipes_df = pd.read_csv(csv_path)
    # Use TF-IDF to vectorize ingredients
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(recipes_df['Ingredients'])

    # Compute cosine similarity between recipes
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    user_ingredients = result+',bolognese'
    print(user_ingredients)
    print("going to method")
    recommended_recipes = recommend_recipes(user_ingredients, recipes_df, tfidf_vectorizer, cosine_similarities,tfidf_matrix)

    # Print the recommended recipes
    print("Recommended Recipes:")
    for recipe in recommended_recipes:
        print("-", recipe)
    return redirect("/home")


if __name__=="__main__":
    app.run(debug=True,port=5010)
