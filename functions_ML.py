import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import seaborn as sns
#import plotly.graph_objects as go
import matplotlib.pyplot as plt


# +++++++++++++++++++++++++++++++++++Scaling ++++++++++++++++++++++++++++
def scaling(df,type):
    """
        Description:
        Scales Dataframe to different weights, only ints/floats in df
        Parameters:
        df: dataframe| only ints/floats
        type: str | indicating which scaler to use
                    'minmax'
                    'std'
                    'robust'
                    'quartile'
                    'power'
        Returns:
            df with scaled values

        """
    # Initialise the transformer (optionally, set parameters)
    if type == 'minmax':
        myscaler = MinMaxScaler().set_output(transform="pandas")
    elif type == 'std':
        myscaler = StandardScaler().set_output(transform="pandas")
    elif type == 'robust':
        myscaler = RobustScaler().set_output(transform="pandas")
    elif type == 'quartile':
        myscaler = QuantileTransformer(n_quantiles = df.shape[0]).set_output(transform="pandas")
    elif type == 'power':
        myscaler = PowerTransformer().set_output(transform="pandas")
        
    # Use the transformer to transform the data
    df_scaled = myscaler.fit_transform(df)

    return df_scaled



# ++++++++++++++++++++++++++ Check elbow inertia ++++++++++++++++++++++++++++

def inertia_plot(df, max_k, seed):
    """
        Description:
        Calculation of inertia scores and returning a plot of intertia score
        versus k. Detecting an elbow in this plots shows mathematical
        best k-value with minimized euklidian distances. 
        df: dataframe| only ints/floats
        max_k: int | maximum k value to use
        seed: int | seed

        Returns:
            None

        """
    # Calculating the Inertia and put it into a list
    # Create an empty list to store the inertia scores
    inertia_list = []
    
    # Iterate over the range of cluster numbers
    for i in range(1, max_k + 1):
        # Create a KMeans object with the specified number of clusters
        myKMeans = KMeans(n_clusters = i,
                          random_state = seed)
        # Fit the KMeans model to the scaled data
        myKMeans.fit(df)
    
        # Append the inertia score to the list
        inertia_list.append(myKMeans.inertia_)


    # plotting
    # Set the Seaborn theme to darkgrid
    sns.set_theme(style='darkgrid')
    
    (
    # Create a line plot of the inertia scores
    sns.relplot(y = inertia_list,
                x = range(1, max_k + 1),
                kind = 'line',
                marker = 'o',
                height = 4,
                aspect = 1.5)
    # Set the title of the plot
    .set(title=f"Inertia score from 1 to {max_k} clusters")
    # Set the axis labels
    .set_axis_labels("Number of clusters", "Inertia score")
    );


# ++++++++++++++++++++++++++ Silhouette Scores ++++++++++++++++++++++++++++

def silhouette_plot(df,min_k, max_k, seed):
    """
        Description:
        Calculation of the Silhouette score and returning a plot of the 
        Sihouette score versus k-value. Finding local maxima to identify 
        best distinguishable clusters. 
        df: dataframe| only ints/floats
        min_k: int | minimum k value to use
        max_k: int | maximum k value to use
        seed: int | seed
        
        Returns:
            None

        """

    sil_scores = []
    
    for j in range(min_k, max_k):
    
        # Create a KMeans object with the specified number of clusters
        myKMeans = KMeans(n_clusters = j,
                        random_state = seed)
    
        # Fit the KMeans model to the scaled data
        myKMeans.fit(df)
        # Get the cluster labels
        labels = myKMeans.labels_
    
        # Calculate the silhouette score
        score = silhouette_score(df, labels)
    
        # Append the silhouette score to the list
        sil_scores.append(score)
    
    # plotting
    # Set the Seaborn theme to darkgrid
    sns.set_theme(style='darkgrid')
    
    (
    # Create a line plot of the inertia scores
    sns.relplot(y =sil_scores,
                x = range(min_k, max_k),
                kind = 'line',
                marker = 'o',
                height = 4,
                aspect = 1.5)
    # Set the title of the plot
    .set(title=f"Silhouette score from 2 to {max_k - 1} clusters")
    .set_axis_labels("Number of clusters", "Silhouette score")
    );


# +++++++++++++++++++++ Two Dimensional Clustering ++++++++++++++++++++++++
# define function which does the kmean clustering and plot it
def two_dimension_exploration(df, feature1, feature2, k, random_seed):
    # create df with both features
    two_feature_df = df.loc[:, [feature1, feature2]]

    # initialise the model
    myKMeans = KMeans(n_clusters = k, random_state = random_seed)

    # fit the model to the data
    myKMeans.fit(two_feature_df)

    # plot
    # figure size
    plt.figure(figsize=(4,3))
    
    plt.scatter(x = two_feature_df.iloc[:, 0],
            y = two_feature_df.iloc[:, 1],
            c = myKMeans.labels_,  # use kmean labels for colouring
            cmap = 'viridis')
    # insert annotations
    #for idx, row in two_feature_df.iterrows():
     #   plt.annotate(idx, (row[feature1], row[feature2]), xytext=(5, 0), textcoords='offset points')


    # # title and labels
    plt.title('KMeans Clustering')
    plt.xlabel(two_feature_df.columns[0])
    plt.ylabel(two_feature_df.columns[1])

    # Display the plot
    plt.show()

# +++++++++++++++++++++ Two Dimensional Clustering ++++++++++++++++++++++++
# Code from Julia
# define a functions which finds clusters using kmean with all dimensions
def clustering_n_dim(df, k, random_seed):
    # initialize kmean
    myKMeans = KMeans(n_clusters = k,
                        random_state = random_seed)
    
    # fit the model to the data
    myKMeans.fit(df)

    # get the clusters, insert to df and sort by clusters
    table = myKMeans.labels_
    df["table"] = table
    df.sort_values(by="table")

    return df










