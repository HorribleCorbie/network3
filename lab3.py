# %%
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import torch.utils.data as utils_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.linalg import norm
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
from sklearn.model_selection import RandomizedSearchCV
import os
import joblib
from sanfis import SANFIS
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
class MetricsCalculator:
    def calculate_comprehensive_metrics(self, y_true, y_pred, model_name=""):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred) + 1e-10)
        mape = np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100
        print(f"\n{model_name} - МЕТРИКИ КАЧЕСТВА:")
        print("-" * 40)
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.6f}")
        print(f"MAPE: {mape:.4f}%")
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

    def approximation_quality(self, deviation, y_pred, y_true):
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        if len(y_pred) != len(y_true):
            print(f"ОШИБКА: Разные размеры y_pred({len(y_pred)}) и y_true({len(y_true)})")
            return 0
        absolute_errors = np.abs(y_pred - y_true)
        within_deviation = np.sum(absolute_errors <= deviation)
        percentage = (within_deviation / len(y_true)) * 100
        return min(percentage, 100.0)
# %%
def visualize_and_evaluate_3d_model(
    model,
    X_test,
    y_test,
    y_pred,
    function,
    title_name,
    x1_range=(-3, 3),
    x2_range=(-2, 2),
    resolution=80,
    scaler_X=None,
    scaler_y=None
):
    x1 = np.linspace(*x1_range, resolution)
    x2 = np.linspace(*x2_range, resolution)

    X1, X2 = np.meshgrid(x1, x2)

    Z_true = function(X1, X2)

    X_grid = np.column_stack((X1.ravel(), X2.ravel()))

    if isinstance(model, SANFIS):
        device = next(model.parameters()).device

        X_grid_tensor = torch.as_tensor(
            X_grid,
            dtype=torch.float32,
            device=device
        )

        Z_pred = np.asarray(
            model.predict(X_grid_tensor)
        ).reshape(X1.shape)

    elif isinstance(model, torch.nn.Module):
        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            X_grid_tensor = torch.tensor(
                X_grid,
                dtype=torch.float32,
                device=device
            )
            Z_pred = (
                model(X_grid_tensor)
                .detach()
                .cpu()
                .numpy()
                .reshape(X1.shape)
            )
    else:
        X_grid_input = X_grid

        if scaler_X is not None:
            X_grid_input = scaler_X.transform(X_grid)

        if hasattr(model, "predict"):
            Z_pred = model.predict(X_grid_input).reshape(X1.shape)
        else:
            Z_pred = model.test(X_grid_input).reshape(X1.shape)

        if scaler_y is not None:
            Z_pred = scaler_y.inverse_transform(
                Z_pred.reshape(-1, 1)
            ).reshape(X1.shape)

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.detach().cpu().numpy()

    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X1,
        y=X2,
        z=Z_true,
        name="Исходная функция",
        colorscale="Viridis",
        opacity=0.65,
        showscale=False
    ))

    fig.add_trace(go.Surface(
        x=X1,
        y=X2,
        z=Z_pred,
        name="Аппроксимация модели",
        colorscale="Plasma",
        opacity=0.65,
        showscale=False
    ))

    fig.add_trace(go.Scatter3d(
        x=X_test[:, 0],
        y=X_test[:, 1],
        z=y_test,
        mode="markers",
        name="Фактические значения",
        marker=dict(size=3, color="blue")
    ))

    fig.add_trace(go.Scatter3d(
        x=X_test[:, 0],
        y=X_test[:, 1],
        z=y_pred,
        mode="markers",
        name="Предсказания модели",
        marker=dict(size=3, color="red")
    ))

    fig.update_layout(
        title=title_name,
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="z = f(x1, x2)"
        ),
        height=750
    )

    fig.show()


def visualize_and_evaluate_2d_model(X, Y, predict, title_name):
    df = pd.DataFrame({'X': X.ravel(), 'Y': Y.ravel(), 'Predicted': predict.ravel()})
    fig = px.scatter(df, x='X', y='Y')
    fig.add_scatter(x=df['X'], y=df['Predicted'], mode='markers',
                    marker=dict(size=5, color='yellow'), name='Predicted Points')
    fig.update_layout(xaxis_title='X', yaxis_title='Y', title=title_name)
    fig.show()

class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters

        self.centers = None
        self.sigmas = None
        self.W = np.random.rand(numCenters, outdim)

    def _initialize(self, X):
        kmeans = KMeans(
            n_clusters=self.numCenters,
            random_state=42,
            n_init=10
        )

        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        distances = np.linalg.norm(
            self.centers[:, None, :] - self.centers[None, :, :],
            axis=2
        )

        np.fill_diagonal(distances, np.inf)

        nearest_distances = np.sort(distances, axis=1)[:, :3]

        self.sigmas = np.mean(nearest_distances, axis=1)

        self.sigmas = np.maximum(self.sigmas, 1e-6)

    def _basisfunc(self, c, d, sigma):
        distance = norm(c - d)
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))

    def _calcAct(self, X):
        G = np.zeros((X.shape[0], self.numCenters))

        for i, center in enumerate(self.centers):
            for j, x in enumerate(X):
                G[j, i] = self._basisfunc(
                    center,
                    x,
                    self.sigmas[i]
                )

        return G

    def train(self, X, Y, epochs=1000, lr=0.01, adam=False):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, self.outdim)

        self._initialize(X)

        G = self._calcAct(X)

        if adam:
            G_tensor = torch.tensor(G, dtype=torch.float32)
            Y_tensor = torch.tensor(Y, dtype=torch.float32)

            W = nn.Parameter(
                torch.tensor(self.W, dtype=torch.float32)
            )

            optimizer = torch.optim.Adam(
                [W],
                lr=lr
            )

            loss_fct = nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(
                G_tensor, Y_tensor
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=64,
                shuffle=True
            )

            for epoch in range(epochs):
                epoch_loss = 0

                for batch_G, batch_Y in loader:
                    prediction = batch_G @ W
                    loss = loss_fct(prediction, batch_Y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], "
                        f"Loss: {epoch_loss / len(loader):.6f}"
                    )

            self.W = W.detach().numpy()

        else:
            self.W = np.dot(np.linalg.pinv(G), Y)

    def test(self, X):
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y

    def plot_gaussians(self, X_train, Y_train):
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values

        if self.indim != 1:
            raise ValueError("Метод предназначен для одномерных данных.")

        x = np.linspace(
            X_train.min(),
            X_train.max(),
            500
        )

        centers = np.array(self.centers).flatten()

        G = self._calcAct(x.reshape(-1, 1))

        contributions = G * self.W.flatten()

        y_pred = np.sum(contributions, axis=1)

        plt.figure(figsize=(10, 5))

        plt.scatter(
            centers,
            np.zeros(len(centers)),
            color='red',
            marker='x',
            s=100,
            label='Центры нейронов'
        )

        for i, center in enumerate(centers):
            plt.axvline(
                center,
                linestyle='--',
                alpha=0.4
            )
            plt.text(
                center,
                0.03,
                f'C{i + 1} = {center:.3f}',
                rotation=90
            )

        plt.title('Центры нейронов RBF')
        plt.xlabel('x')
        plt.yticks([])
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))

        for i in range(self.numCenters):
            plt.plot(
                x,
                G[:, i],
                label=f'Нейрон {i + 1}'
            )

        plt.title('Функции активации нейронов RBF')
        plt.xlabel('x')
        plt.ylabel('Активация')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))

        for i in range(self.numCenters):
            plt.plot(
                x,
                contributions[:, i],
                label=f'Вклад нейрона {i + 1}'
            )

        plt.title('Вклады отдельных нейронов в аппроксимацию')
        plt.xlabel('x')
        plt.ylabel('Вклад')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))

        plt.plot(
            x,
            y_pred,
            color='black',
            label='Сумма вкладов нейронов'
        )

        plt.title('Итоговая аппроксимирующая функция RBF')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()

        individual = G * self.W.flatten()

        y_pred = np.sum(individual, axis=1)

        plt.figure(figsize=(12, 6))

        plt.plot(
            X_train.flatten(),
            np.asarray(Y_train).flatten(),
            'ko',
            markersize=3,
            label='Исходная функция'
        )

        for i in range(self.numCenters):
            plt.plot(
                x,
                individual[:, i],
                label=f'Нейрон {i + 1}'
            )

        plt.plot(
            x,
            y_pred,
            'r--',
            linewidth=2.5,
            label='Итоговая аппроксимация RBF'
        )

        plt.title('Аппроксимация функции нейронами RBF')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()

# %%
def print_rules(fis, title):
    params = dict(fis.named_parameters())

    mu = params['layers.fuzzylayer.fuzzyfication.0._mu']
    sigma = params['layers.fuzzylayer.fuzzyfication.0._sigma']
    weight = params['layers.consequence._weight']
    bias = params['layers.consequence._bias']

    mu = mu.detach().cpu().numpy().flatten()
    sigma = sigma.detach().cpu().numpy().flatten()
    weight = weight.detach().cpu().numpy().flatten()
    bias = bias.detach().cpu().numpy().flatten()

    print(f'\n{title}')

    for i in range(len(mu)):
        sign = '+' if bias[i] >= 0 else '-'

        print(
            f'Правило {i + 1}: ЕСЛИ x принадлежит A{i + 1} '
            f'(mu={mu[i]:.4f}, sigma={sigma[i]:.4f}), '
            f'ТО y = {weight[i]:.4f}x {sign} {abs(bias[i]):.4f}'
        )
