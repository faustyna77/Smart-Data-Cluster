# Smart Data Cluster

This project is a data analysis and clustering application built with Streamlit. It provides multiple machine learning algorithms for data clustering and association analysis. Users can explore the following features:

* Association Analysis using the Apriori algorithm to detect frequent itemsets and association rules.
* K-Means Clustering for dividing data into a specified number of clusters.
* DBSCAN Clustering to identify dense clusters and handle noise in the data.
* Hierarchical Clustering with a dendrogram visualization for analyzing cluster relationships.
* Additional Clustering Methods like Gaussian Mixture Models (GMM) and Spectral Clustering for more advanced data segmentation


## Running the Application

### Generating .csv file
User can generate a new .csv file using `generator.py` or use the existing `sales_data.csv` file which will be selected in the web application for analysis.
Both the generator and the finished .csv file are located in the `/data` folder.

### Installed Dependencies
The application requires the following dependencies, which can be installed via pip: <br>
``` 
pip install -r requirements.txt
```

### Start the Streamlit Application

After installing all dependencies, run the Streamlit application using the following command:

```
streamlit run app.py
```
Once the application is running, visit the following address in your browser:
```
http://localhost:8501
```
