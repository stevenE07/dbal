import torch
import numpy as np
from scipy import stats


def predictions_from_pool(
    model, X_pool: np.ndarray, T: int = 100, training: bool = True
):
    """Run random_subset prediction on model and return the output

    Attributes:
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    random_subset = np.random.choice(range(len(X_pool)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack(
            [
                torch.softmax(
                    model.estimator.forward(X_pool[random_subset], training=training),
                    dim=-1,
                )
                .cpu()
                .numpy()
                for _ in range(T)
            ]
        )
    return outputs, random_subset


def uniform(
    model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Baseline acquisition a(x) = unif() with unif() a function
    returning a draw from a uniform distribution over the interval [0,1].
    Using this acquisition function is equivalent to choosing a point
    uniformly at random from the pool.

    Attributes:
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that randomly select from pool set,
        training: If False, run test without MC dropout. (default=True)
    """
    query_idx = np.random.choice(range(len(X_pool)), size=n_query, replace=False)
    return query_idx, X_pool[query_idx]


def shannon_entropy_function(
    model, X_pool: np.ndarray, T: int = 100, E_H: bool = False, training: bool = True
):
    """H[y|x,D_train] := - sum_{c} p(y=c|x,D_train)log p(y=c|x,D_train)

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        T: Number of MC dropout iterations aka training iterations,
        E_H: If True, compute H and EH for BALD (default: False),
        training: If False, run test without MC dropout. (default=True)
    """
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training=training)
    pc = outputs.mean(axis=0)
    H = (-pc * np.log(pc + 1e-10)).sum(
        axis=-1
    )  # To avoid division with zero, add 1e-10
    if E_H:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E, random_subset
    return H, random_subset


def max_entropy(
    model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Choose pool points that maximise the predictive entropy.
    Using Shannon entropy function.

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise max_entropy a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    acquisition, random_subset = shannon_entropy_function(
        model, X_pool, T, training=training
    )
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def bald(
    model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Choose pool points that are expected to maximise the information
    gained about the model parameters, i.e. maximise the mutal information
    between predictions and model posterior. Given
    I[y,w|x,D_train] = H[y|x,D_train] - E_{p(w|D_train)}[H[y|x,w]]
    with w the model parameters (H[y|x,w] is the entropy of y given w).
    Points that maximise this acquisition function are points on which the
    model is uncertain on average but there exist model parameters that produce
    disagreeing predictions with high certainty. This is equivalent to points
    with high variance in th einput to the softmax layer

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise bald a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    H, E_H, random_subset = shannon_entropy_function(
        model, X_pool, T, E_H=True, training=training
    )
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def batch_bald(
    model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """
    Implementación de BatchBALD para seleccionar un lote de puntos que maximicen la
    información mutua sobre los parámetros del modelo.

    Args:
        model: El modelo ya entrenado y listo para medir incertidumbre.
        X_pool: Conjunto de datos del pool para seleccionar incertidumbre.
        n_query: Número de puntos a seleccionar que maximicen batch_bald a(x) del conjunto de pool.
        T: Número de iteraciones de Monte Carlo dropout (o entrenamiento).
        training: Si es False, ejecuta sin MC dropout. (default=True)

    Returns:
        query_idx: Índices de los puntos seleccionados en X_pool.
        X_pool[query_idx]: Puntos seleccionados en el conjunto de pool.
    """
    
    batch_size = n_query  # Tamaño del lote de puntos a seleccionar
    selected_indices = []
    remaining_indices = np.arange(len(X_pool))
    
    # Paso 1: Inicializar entropía y entropía esperada para el conjunto de datos
    H, E_H, random_subset = shannon_entropy_function(
        model, X_pool, T, E_H=True, training=training
    )
    
    # Paso 2: Selección iterativa de puntos que maximicen la información mutua en el lote
    for _ in range(batch_size):
        acquisition_scores = []
        
        # Para cada punto en el conjunto de pool no seleccionado
        for idx in remaining_indices:
            # Cálculo de la ganancia de información mutua esperada para el lote actual + el nuevo punto
            # Esta parte sería una implementación extendida del cálculo de I en BatchBALD
            current_acquisition = H[idx] - E_H[idx]
            
            acquisition_scores.append(current_acquisition)
        
        # Seleccionar el índice con el máximo score de adquisición
        max_idx = np.argmax(acquisition_scores)
        selected_indices.append(remaining_indices[max_idx])
        
        # Eliminar el índice seleccionado del conjunto de índices restantes
        remaining_indices = np.delete(remaining_indices, max_idx)
    
    # Devolver los índices de consulta y los puntos seleccionados en el pool
    query_idx = np.array(selected_indices)
    return query_idx, X_pool[query_idx]

def var_ratios(
    model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Like Max Entropy but Variational Ratios measures lack of confidence.
    Given: variational_ratio[x] := 1 - max_{y} p(y|x,D_{train})

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise var_ratios a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training)
    preds = np.argmax(outputs, axis=2)
    _, count = stats.mode(preds, axis=0)
    acquisition = (1 - count / preds.shape[1]).reshape((-1,))
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]


def mean_std(
    model, X_pool: np.ndarray, n_query: int = 10, T: int = 100, training: bool = True
):
    """Maximise mean standard deviation
    Given: sigma_c = sqrt(E_{q(w)}[p(y=c|x,w)^2]-E_{q(w)}[p(y=c|x,w)]^2)

    Attributes:
        model: Model that is ready to measure uncertainty after training,
        X_pool: Pool set to select uncertainty,
        n_query: Number of points that maximise mean std a(x) from pool set,
        T: Number of MC dropout iterations aka training iterations,
        training: If False, run test without MC dropout. (default=True)
    """
    outputs, random_subset = predictions_from_pool(model, X_pool, T, training)
    sigma_c = np.std(outputs, axis=0)
    acquisition = np.mean(sigma_c, axis=-1)
    idx = (-acquisition).argsort()[:n_query]
    query_idx = random_subset[idx]
    return query_idx, X_pool[query_idx]
