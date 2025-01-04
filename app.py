import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import umap


# Custom K-Means Implementation
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _init_centroids(self, X):
        # Random initialization of centroids
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def fit(self, X):
        # Normalize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iters):
            # Assign clusters
            clusters = [[] for _ in range(self.n_clusters)]

            for point in X:
                distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(point)

            # Store previous centroids
            prev_centroids = self.centroids.copy()

            # Update centroids
            self.centroids = np.array([np.mean(cluster, axis=0) if len(cluster) > 0 else prev_centroids[i]
                                       for i, cluster in enumerate(clusters)])

            # Check convergence
            if np.allclose(self.centroids, prev_centroids):
                break

        # Assign final labels
        self.labels = np.array([np.argmin([self._euclidean_distance(point, centroid)
                                           for centroid in self.centroids]) for point in X])

        return self


# Custom DBSCAN Implementation
class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X):
        # Normalize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        n_points = X.shape[0]
        self.labels = np.zeros(n_points, dtype=int)
        current_label = 0

        for point_idx in range(n_points):
            if self.labels[point_idx] != 0:
                continue

            # Find neighbors
            neighbors = [
                idx for idx in range(n_points)
                if self._euclidean_distance(X[point_idx], X[idx]) < self.eps
            ]

            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -1  # Noise point
            else:
                current_label += 1
                self.labels[point_idx] = current_label

                # Expand cluster
                seed_set = set(neighbors)
                while seed_set:
                    current_point = seed_set.pop()

                    if self.labels[current_point] == -1:
                        self.labels[current_point] = current_label

                    if self.labels[current_point] != 0:
                        continue

                    self.labels[current_point] = current_label

                    current_neighbors = [
                        idx for idx in range(n_points)
                        if self._euclidean_distance(X[current_point], X[idx]) < self.eps
                    ]

                    if len(current_neighbors) >= self.min_samples:
                        seed_set.update(current_neighbors)

        return self


def perform_apriori_analysis(transactions):
    # Konwersja zestawów produktów do listy list, aby uniknąć problemów z serializacją
    transactions = [list(map(str, transaction)) for transaction in transactions]

    # Użycie TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Znajdź częste zbiory (itemsets)
    frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)

    # Zabezpieczenie przed błędem num_itemsets
    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    except TypeError:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1,
                                  num_itemsets=len(frequent_itemsets))

    # Konwersja frozenset na zwykłe zbiory, aby uniknąć problemów z serializacją
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: set(x))
    rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: set(x))

    return frequent_itemsets, rules


def optimize_spectral_clustering(X, n_clusters=3):
    # Użycie UMAP do redukcji wymiarowości przed klastrowaniem
    reducer = umap.UMAP(n_components=5, random_state=42)
    X_reduced = reducer.fit_transform(X)

    from sklearn.cluster import SpectralClustering
    from sklearn.preprocessing import StandardScaler

    # Standaryzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    # Spektralne grupowanie na zredukowanych danych
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        random_state=42,
        n_neighbors=10
    )
    return spectral.fit_predict(X_scaled)


def main():
    st.title("🛒 Wielowymiarowa Analiza Danych Sprzedażowych")

    # Sidebar
    st.sidebar.header("Opcje Analizy")
    uploaded_file = st.sidebar.file_uploader("Wgraj plik CSV", type=['csv'])

    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("### Wczytane dane:")
        st.dataframe(df.head())

        # Prepare product transactions for Apriori
        transactions = df.groupby('Transaction_ID')['Product'].apply(list).tolist()

        # Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Analiza Asocjacji",
            "Grupowanie K-Means",
            "Grupowanie DBSCAN",
            "Grupowanie Hierarchiczne",
            "Pozostałe Metody Grupowania"
        ])

        with tab1:
            st.header("Analiza Asocjacji - Algorytm Apriori")
            if st.button("Przeprowadź Analizę Asocjacji"):
                frequent_itemsets, rules = perform_apriori_analysis(transactions)

                st.subheader("Częste Zestawy Produktów")
                st.write("""
                ### Interpretacja Częstych Zestawów Produktów
                - **Support (Wsparcie)**: Procent wszystkich transakcji, które zawierają dany zestaw produktów
                - **Im wyższa wartość support, tym częściej produkty są kupowane razem**
                - Przykład: Support 0.05 oznacza, że 5% wszystkich transakcji zawiera ten zestaw produktów
                """)
                st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

                st.subheader("Reguły Asocjacyjne")
                st.write("""
                ### Interpretacja Reguł Asocjacyjnych
                - **Antecedents**: Produkty inicjujące regułę
                - **Consequents**: Produkty, które są kupowane razem z antecedents
                - **Lift**: Miara siły związku między produktami
                  - Lift > 1: Produkty kupowane częściej razem niż losowo
                  - Lift = 1: Produkty kupowane niezależnie
                  - Lift < 1: Produkty rzadziej kupowane razem
                """)
                st.dataframe(rules.sort_values(by='lift', ascending=False).head(10))

        with tab2:
            st.header("Grupowanie K-Means")
            n_clusters = st.slider("Liczba klastrów", 2, 10, 3)

            # Prepare data for clustering
            features_for_clustering = ['Quantity', 'Price_Per_Unit', 'Total_Price']
            X = df[features_for_clustering].values

            kmeans = CustomKMeans(n_clusters=n_clusters)
            kmeans.fit(X)

            # Visualize clusters
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
            plt.title('K-Means Clustering')
            plt.xlabel('Quantity')
            plt.ylabel('Price Per Unit')
            plt.colorbar(scatter, label='Klaster')
            st.pyplot(plt)

            st.write("""
            ### Interpretacja Grupowania K-Means
            - Każdy kolor reprezentuje osobny klaster (grupę) produktów
            - Produkty w tym samym kolorze są do siebie podobne pod względem ilości i ceny
            - Algorytm dzieli dane na K równomiernych, kulistych klastrów
            - Centroid (środek klastra) reprezentuje typowy punkt dla danej grupy

            **Jak czytać wykres**:
            - Oś X: Ilość produktów
            - Oś Y: Cena produktu
            - Kolor: Przynależność do klastra
            """)

        with tab3:
            st.header("Grupowanie DBSCAN")
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
            min_samples = st.slider("Minimalna liczba próbek", 2, 20, 5)

            X = df[['Quantity', 'Price_Per_Unit', 'Total_Price']].values

            dbscan = CustomDBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X)

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels, cmap='viridis')
            plt.title('DBSCAN Clustering')
            plt.xlabel('Quantity')
            plt.ylabel('Price Per Unit')
            plt.colorbar(scatter, label='Klaster')
            st.pyplot(plt)

            st.write("""
            ### Interpretacja Grupowania DBSCAN
            - Algorytm grupuje punkty na podstawie gęstości
            - Czarny kolor: Punkty odbiegające od normy (szum)
            - Różne kolory: Odrębne, gęste skupiska danych

            **Kluczowe parametry**:
            - Epsilon (eps): Maksymalna odległość między punktami w klastrze
            - Min Samples: Minimalna liczba punktów wymagana do utworzenia klastra

            **Jak czytać wykres**:
            - Punkty blisko siebie w gęstym obszarze należą do tego samego klastra
            - Punkty oddalone są oznaczane jako szum
            """)

        with tab4:
            st.header("Grupowanie Hierarchiczne")
            # Compute linkage matrix
            X = df[['Quantity', 'Price_Per_Unit', 'Total_Price']].values
            Z = linkage(X, method='ward')

            plt.figure(figsize=(10, 6))
            dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90)
            plt.title('Dendrogram Grupowania Hierarchicznego')
            plt.xlabel('Próbki')
            plt.ylabel('Odległość')
            st.pyplot(plt)

            st.write("""
            ### Interpretacja Dendrogramu
            - Dendrogram to diagram przypominający drzewo genealogiczne danych
            - Pionowa oś (Y) pokazuje odległość/podobieństwo między grupami
            - Im niżej się łączą gałęzie, tym większe podobieństwo między grupami

            **Jak czytać**:
            - Najniższe połączenia reprezentują najbardziej podobne grupy produktów
            - Wysokość pionowych linii wskazuje na różnice między klastrami
            - Możesz "uciąć" drzewo na różnych wysokościach, aby otrzymać różną liczbę klastrów
            """)

        with tab5:
            st.header("Pozostałe Metody Grupowania")
            from sklearn.mixture import GaussianMixture

            X = df[['Quantity', 'Price_Per_Unit', 'Total_Price']].values

            st.subheader("Gaussian Mixture Models")
            gmm = GaussianMixture(n_components=3)
            gmm_labels = gmm.fit_predict(X)

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis')
            plt.title('Gaussian Mixture Models Clustering')
            plt.xlabel('Quantity')
            plt.ylabel('Price Per Unit')
            plt.colorbar(scatter, label='Klaster')
            st.pyplot(plt)

            st.write("""
            ### Interpretacja Gaussian Mixture Models
            - Zakłada, że dane pochodzą z mieszaniny rozkładów gaussowskich
            - Każdy klaster reprezentuje osobny rozkład prawdopodobieństwa
            - Algorytm przypisuje prawdopodobieństwo przynależności do klastra

            **Charakterystyka**:
            - Elastyczny w grupowaniu danych o różnych kształtach
            - Uwzględnia niepewność przynależności do klastra
            - Każdy punkt może mieć częściową przynależność do kilku klastrów
            """)

            st.subheader("Spectral Clustering")
            spectral_labels = optimize_spectral_clustering(X)

            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis')
            plt.title('Spectral Clustering')
            plt.xlabel('Quantity')
            plt.ylabel('Price Per Unit')
            plt.colorbar(scatter, label='Klaster')
            st.pyplot(plt)

            st.write("""
            ### Interpretacja Spectral Clustering
            - Grupowanie oparte na teorii spektralnej grafów
            - Przekształca dane do przestrzeni o niższej wymiarowości przed grupowaniem
            - Efektywny dla danych o skomplikowanej, nieliniowej strukturze

            **Kluczowe cechy**:
            - Radzi sobie z grupami o nieregularnych kształtach
            - Wykorzystuje własności macierzy podobieństwa
            - Działa dobrze, gdy klastra nie są kulistymi skupiskami
            """)


if __name__ == "__main__":
    st.set_page_config(page_title="Analiza Sprzedaży", page_icon="🛒")
    main()