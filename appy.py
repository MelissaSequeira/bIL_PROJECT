import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🎓 Educational Data Mining Dashboard")

uploaded_file = st.file_uploader("📂 Upload Student Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Auto-generate a performance category if none exists
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

    if not categorical_cols and len(numeric_cols) >= 2:
        # Create a total and performance level
        df['Total'] = df[numeric_cols].sum(axis=1)
        df['Performance_Level'] = pd.cut(df['Total'], bins=[0, 150, 200, df['Total'].max()],
                                         labels=['Low', 'Medium', 'High'])
        categorical_cols.append('Performance_Level')
        st.info("🛠️ Auto-generated 'Performance_Level' column for classification.")

    tab1, tab2, tab3 = st.tabs(["K-Means Clustering", "Decision Tree", "Linear Regression"])

    # --- K-MEANS ---
    with tab1:
        st.header("🔹 K-Means Clustering")

        k = st.slider("Number of Clusters (k)", 2, 10, 3)
        features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:2])

        if len(features) >= 2:
            X = df[features]

            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X)
            df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax)
            ax.set_title("Clustering Visualization (PCA)")
            st.pyplot(fig)

            st.write("📌 Cluster Counts:")
            st.dataframe(df['Cluster'].value_counts())
        else:
            st.warning("Please select at least two numeric features for clustering.")

    # --- DECISION TREE ---
    with tab2:
        st.header("🌲 Decision Tree Classification")

        if categorical_cols and numeric_cols:
            target = st.selectbox("Select Target (Class column)", categorical_cols)
            features = st.multiselect("Select features for training", numeric_cols, default=numeric_cols)

            if target and features:
                X = df[features]
                y = df[target]

                # Encode target if needed
                if y.dtype == 'object':
                    y = pd.factorize(y)[0]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                tree = DecisionTreeClassifier(max_depth=4, random_state=42)
                tree.fit(X_train, y_train)
                y_pred = tree.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                st.success(f"🎯 Accuracy: {acc:.2f}")

                fig, ax = plt.subplots(figsize=(12, 6))
                plot_tree(tree, feature_names=features, class_names=True, filled=True, ax=ax)
                st.pyplot(fig)
        else:
            st.warning("You need both numeric features and a categorical target column.")

    # --- LINEAR REGRESSION ---
    with tab3:
        st.header("📈 Linear Regression")

        target = st.selectbox("Select Target (to predict)", numeric_cols, key="reg_target")
        features = st.multiselect("Select independent variables", [col for col in numeric_cols if col != target])

        if target and features:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            st.success(f"📉 RMSE: {rmse:.2f}")
            st.info(f"📊 R² Score: {r2:.2f}")

            if len(features) == 1:
                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, label="Actual")
                ax.plot(X_test, y_pred, color='red', label="Predicted")
                ax.set_xlabel(features[0])
                ax.set_ylabel(target)
                ax.set_title("Linear Regression Fit")
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("Please select a target and at least one independent variable.")
else:
    st.info("📤 Upload a CSV file to get started.")
