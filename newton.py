# import von numpy fuer arrays und wesentliche mathematische Funktionen
import numpy as np
# import von plotly fuer die Plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Um plots zu kopieren
from copy import deepcopy

class plot(go.Figure):
    result = None

    def __init__(self, *args):
        go.Figure.__init__(self, *args)

    def plot_function(self, xmin, xmax, f, anzahl_schritte = 500):
        plot.xmin = xmin
        plot.xmax = xmax
        x_axis = np.linspace(xmin, xmax, anzahl_schritte)
        y_axis = f(x_axis)
        self.add_scatter(x=x_axis, y=[0] * anzahl_schritte, line_color = "grey", showlegend = False)
        self.add_scatter(x=x_axis, y=y_axis, showlegend = False)
        self.update_layout(template='plotly_white')

    def add_point(self, x, f):
        self.add_scatter(x=[x], y=[f(x)],
                         showlegend = False)

    def add_tangent(self, x, f, df):
        self.add_scatter(
            x = [x-10, x+10],
            y = [f(x) - 10*df(x), f(x) + 10*df(x)],
            mode = "lines",
            showlegend = False)
        self.function_zoom(f)
        self.update_layout()

    def add_newton_iteration(self, x0, function, df, Iterationen = 1, color = "blue"):
        x = np.zeros(Iterationen + 1)
        f_x = np.zeros(Iterationen + 1)
        x[0] = np.array(x0)
        f_x[0] = function(x[0])
        for i in range(Iterationen):
            x[i + 1] = x[i] - function(x[i])/df(x[i])
            f_x[i + 1] = function(x[i + 1])
            self.add_scatter(
                x=[x[i], x[i+1]],
                y=[f_x[i], 0],
                mode='lines+markers',
                showlegend=False,
                line_color=color)
        # Hier wird der Rundungswert berechnet
        Rundungswert = min(6, max(7, round(1/(abs(f_x[-1]) + 0.000001)) + 2))
        self.update_layout(
            title=
            "x0=" + str(np.round(x0, 4)) +
            ", x" + str(Iterationen) + "=" + str(round(x[-1], Rundungswert)) +
            ", f(x" + str(Iterationen) + ")=" + str(round(f_x[-1], Rundungswert)) +
            ", f'(x" + str(Iterationen) + ")=" + str(round(df(x[-1]), 3)))
        

    def add_newton_iteration_extremum(self, x0, function, df, ddf, Iterationen = 1, color = "blue"):
        x = np.zeros(Iterationen + 1)
        f_x = np.zeros(Iterationen + 1)
        x[0] = np.array(x0)
        f_x[0] = function(x[0])
        for i in range(Iterationen):
            x[i + 1] = x[i] - df(x[i])/ddf(x[i])
            f_x[i + 1] = function(x[i + 1])
            self.add_scatter(
                x=[x[i], x[i+1]],
                y=[f_x[i], f_x[i+1]],
                mode='lines+markers',
                showlegend=False,
                line_color=color)
        # Hier wird der Rundungswert berechnet
        Rundungswert = min(6, max(7, round(1/(abs(f_x[-1]) + 0.000001)) + 2))
        self.update_layout(
            title=
            "x0=" + str(np.round(x0, 4)) +
            ", x" + str(Iterationen) + "=" + str(round(x[-1], Rundungswert)) +
            ", f(x" + str(Iterationen) + ")=" + str(round(f_x[-1], Rundungswert)) +
            ", <br> f'(x" + str(Iterationen) + ")=" + str(df(x[-1])) + ")")

    def function_zoom(self, f):
        x_axis = np.linspace(self.xmin, self.xmax, 500)
        y_axis = f(x_axis)
        self.update_layout(xaxis_range=[self.xmin, self.xmax], yaxis_range=[min(y_axis), max(y_axis)])

    def plot_contour(self, xmin, xmax, ymin, ymax, function):
        x_axis = np.linspace(xmin, xmax, 100)
        y_axis = np.linspace(ymin, ymax, 100)
        [x1, x2] = np.meshgrid(x_axis, y_axis)
        z = function([x1, x2])
        self.add_trace(go.Contour(x=x_axis, y=y_axis, z=z, contours_coloring='lines', showscale=False))
        self.update_layout(template='plotly_white', width=500, height=500)
        self.update_layout(scene = dict(
                    xaxis_title='x1',
                    yaxis_title='x2',
                    zaxis_title='f(x)'))
        self.for_each_trace(
            lambda t: t.update(hovertemplate="x1 %{x}<br>x2 %{y}<br>f(x) %{z}<extra></extra>"))

    def newton_iteration_contour(self, x0, function, grad, hessian, gamma=1, Iterationen=10, color=None, Nebenbedingung=None):
        x = np.zeros(shape=(Iterationen + 1, 2))
        f_x = np.zeros(Iterationen + 1)
        x[0, :] = np.array(x0)
        f_x[0] = np.round(function(x[0, :]), 3)
        for i in range(Iterationen):
            x[i + 1] = x[i, :] - np.matmul(np.linalg.inv(hessian(x[i, :])), grad(x[i, :]))
            f_x[i + 1] = np.round(function(x[i + 1, :]), 3)
        self.add_scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode='lines+markers',
            showlegend=False,
            line_color=color)
        self.result = x[-1]
        self.update_layout(title="x0=" + str(np.round(x0, 3)) + ", gamma =" +
                                    str(np.round(gamma, 3)) + ",<br> Iterationen=" + str(Iterationen) +
                                    ", f(x)=" + str(np.round(f_x[-1], 3)) + ", x=" + str(np.round(self.result, 3)))
        self.for_each_trace(
            lambda t: t.update(hovertemplate="x1 %{x}<br>x2 %{y}<extra></extra>"))
        
    def plot_surface(self, xmin, xmax, ymin, ymax, function, opacity=1, showscale=True, colorscale=None):
        x_axis = np.linspace(xmin, xmax, 100)
        y_axis = np.linspace(ymin, ymax, 100)
        [x, y] = np.meshgrid(x_axis, y_axis)
        z = function([x, y])
        if colorscale is None:
            self.add_surface(x=x, y=y, z=z, opacity=opacity, showscale=showscale)
        else:
            self.add_surface(x=x, y=y, z=z, opacity=opacity, showscale=showscale, colorscale=colorscale)
        self.update_layout(template='plotly_white', width=500, height=500)
        self.update_layout(scene = dict(
                    xaxis_title='x1',
                    yaxis_title='x2',
                    zaxis_title='y'))
        self.for_each_trace(
            lambda t: t.update(hovertemplate="x1 %{x}<br>x2 %{y}<br>f(x) %{z}<extra></extra>"))

    def newton_iteration_surface(self, x0, function, grad, hessian, gamma=1, Iterationen=10, color=None, Nebenbedingung=None):
        x = np.zeros(shape=(Iterationen + 1, 2))
        f_x = np.zeros(Iterationen + 1)
        x[0, :] = np.array(x0)
        f_x[0] = np.round(function(x[0, :]), 3)
        for i in range(Iterationen):
            x[i + 1] = x[i, :] - np.matmul(np.linalg.inv(hessian(x[i, :])), grad(x[i, :]))
            f_x[i + 1] = np.round(function(x[i + 1, :]), 3)
        self.add_scatter3d(
            x=x[:, 0],
            y=x[:, 1],
            z=f_x,
            showlegend=False,
            line_color=color)
        self.result = x[-1]
        self.for_each_trace(
            lambda t: t.update(hovertemplate="x1 %{x}<br>x2 %{y}<br>f(x) %{f_x}<extra></extra>"))
        
def show_plot(contour_plot, surface_plot):
    fig = make_subplots(rows=1,
                        cols=2,
                        specs = [[{"type": "contour"}, {"type": "surface"}]],
                        shared_yaxes = True)

    fig.layout.update(contour_plot.layout)
    fig.update_layout(template='plotly_white', width=1000, height=500)

    for i in range(len(surface_plot.data)):
        fig.add_trace(
            surface_plot.data[i],
            row=1, col=2
        )

    for i in range(len(contour_plot.data)):
        fig.add_trace(
            contour_plot.data[i],
            row=1, col=1
        )

    fig.show()