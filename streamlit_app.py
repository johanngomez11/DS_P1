import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Cargar los datos
@st.cache
def load_data():
    try:
        housing = pd.read_csv("datasets/housing/housing.csv")
        st.write("Columnas del archivo CSV:", housing.columns.tolist())  # Imprime las columnas
        return housing
    except Exception as e:
        st.error(f"Error al cargar el archivo CSV: {e}")
        return None

# Preprocesamiento de datos
def preprocess_data(housing):
    if "median_income" not in housing.columns:
        raise KeyError("La columna 'median_income' no existe en el archivo CSV.")
    
    # Crear categorías de ingresos
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    
    # División estratificada
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    # Eliminar la columna income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    return strat_train_set, strat_test_set

# Limpiar y codificar datos
def clean_and_encode_data(housing):
    # Verificar valores faltantes
    if housing.isnull().any().any():
        # Eliminar filas con valores faltantes
        housing = housing.dropna()
    
    # Codificar la columna categórica 'ocean_proximity'
    if "ocean_proximity" in housing.columns:
        cat_encoder = OneHotEncoder()
        housing_cat_encoded = cat_encoder.fit_transform(housing[["ocean_proximity"]])
        housing_cat_encoded = pd.DataFrame(housing_cat_encoded.toarray(), columns=cat_encoder.get_feature_names_out(["ocean_proximity"]))
        
        # Eliminar la columna categórica original y concatenar las nuevas columnas codificadas
        housing = housing.drop("ocean_proximity", axis=1)
        housing = pd.concat([housing, housing_cat_encoded], axis=1)
    
    return housing

# Entrenar y evaluar modelos
def train_and_evaluate_models(train_set, test_set):
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()
    
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    
    # Verificar valores faltantes
    if housing.isnull().any().any():
        st.error("El conjunto de datos todavía contiene valores faltantes (NaN). Por favor, limpia los datos antes de continuar.")
        return
    
    # Verificar valores infinitos
    if np.isinf(housing.select_dtypes(include=[np.number])).any().any():
        st.error("El conjunto de datos contiene valores infinitos (inf). Por favor, limpia los datos antes de continuar.")
        return
    
    # Definir los parámetros de los modelos
    param_grid_linear = {
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }
    
    param_grid_tree = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    param_grid_forest = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    
    models = {
        "Linear Regression": (LinearRegression(), param_grid_linear),
        "Decision Tree Regression": (DecisionTreeRegressor(), param_grid_tree),
        "Random Forest Regression": (RandomForestRegressor(), param_grid_forest)
    }
    
    results = {}
    
    for model_name, (model, param_grid) in models.items():
        try:
            grid_search = GridSearchCV(model, param_grid, cv=5,
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True)
            grid_search.fit(housing, housing_labels)
            
            final_model = grid_search.best_estimator_
            final_predictions = final_model.predict(X_test)
            final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
            
            # Intervalo de confianza
            confidence = 0.95
            squared_errors = (final_predictions - y_test) ** 2
            confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                             loc=squared_errors.mean(),
                                             scale=stats.sem(squared_errors)))
            
            results[model_name] = {
                "rmse": final_rmse,
                "confidence_interval": confidence_interval
            }
        except Exception as e:
            st.error(f"Error durante el entrenamiento del modelo {model_name}: {e}")
            continue
    
    return results

# Interfaz de Streamlit
def main():
    st.title("Predicción de Precios de Viviendas")
    
    # Cargar datos
    housing = load_data()
    
    if housing is not None:
        st.write("Datos cargados correctamente. Columnas disponibles:", housing.columns.tolist())
        
        # Verifica si la columna 'median_income' existe
        if "median_income" not in housing.columns:
            st.error("La columna 'median_income' no existe en el archivo CSV.")
        else:
            # Limpiar y codificar datos
            housing = clean_and_encode_data(housing)
            
            # Preprocesar datos
            train_set, test_set = preprocess_data(housing)
            
            # Entrenar y evaluar modelos
            if st.button("Entrenar y evaluar modelos"):
                results = train_and_evaluate_models(train_set, test_set)
                
                if results:
                    st.write("Resultados de todos los modelos:")
                    for model_name, result in results.items():
                        st.write(f"- {model_name}:")
                        st.write(f"  RMSE: {result['rmse']}")
                        st.write(f"  Intervalo de confianza: {result['confidence_interval']}")
                    
                    best_model = min(results, key=lambda x: results[x]['rmse'])
                    st.write(f"\nEl mejor modelo es: {best_model} con un RMSE de {results[best_model]['rmse']}")

if __name__ == "__main__":
    main()
