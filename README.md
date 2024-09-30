# UnsupervisedML
The project is based on a hypothetical company called Moosic, a startup that curates playlists manually.
As their business grows, they need to speed up the process using data science. I’m tasked with using audio features from Spotify (like tempo, energy, and danceability) 
to group songs into mood-based playlists using K-Means clustering.
Some members of the team are skeptical about whether these audio features can truly capture the “mood” of a song—something they believe only a human can do. 
I’m exploring how effective data science can be in bridging that gap.

# The Script 
Files:
* SpotifySongsKmeans.ipynb - main file
* functions_ML.py - functions
* 3_spotify_5000_songs.csv - data

Used Libraries:
* pandas
* sklearn
  * Kmeans() from sklearn.cluster
  * RobustScaler() from sklearn.preprocessing
  * silhouette_score() from sklearn.metrics
  * PCA() from sklearn.decomposition
* seaborn, matplotlib
  
# What the script does:
* Load songs with metrics from spotify<br>(danceability, energy, key, loudness, mode, speechiness, instrumentalness, liveness, valence, tempo, type, duration_ms, time_signature)
* some cleaning of the data
* dropping metrics: mode, duration, time_signature, type<br>as these metrics supposedly don't help with recognizing the tone of songs
* preprocesssing: elaborate data with a **robust scaler** as the metrics have different scales<br>using RobustScaler() from sklearn.preprocessing
* Applying a **Principal Component Analysis** to reduce data to it's most important features<br>using PCA() from sklearn.decomposition
* Assessing the **optimal k** for the clustering:
    * Calculation of the intertia score:<br>Find the k with minimized the euclidean distances<br>uses KMeans.intertia_
    * Evaluation of the silhouette coefficient<br>Assessing if for which cluster size the clusters are most distinguishable from each other<br> using the silhoutte_score() function from sklearn.metrics
* Clustering with a **Kmeans Algorithm** with the optimal k<br>using KMeans from sklearn.cluster
* Assessing the playlist by sampling 20 songs a given playlist
* Assessing the spotiry metrics


# A few key challenges I encountered:
Understanding K-Means: I spent time really digging into how the algorithm works beyond just coding it. It was crucial to understand the math and logic behind it.
Choosing the Right Number of Clusters: Finding the right number of playlists (K) wasn’t straightforward. I looked at inertia and silhouette scores, but it’s also about aligning with business goals.
Working with Scikit-learn: The coding part wasn’t too hard, but understanding what happens under the hood in scikit-learn was a challenge worth tackling.

