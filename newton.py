# import von numpy fuer arrays und wesentliche mathematische Funktionen
import numpy as np
# import von plotly fuer die Graphen
import plotly.graph_objects as go
# TODO: kann ich plotly.subplots rausnehmen?
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
        self.update_layout(template='plotly_white')# width=500, height=500)

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
            ", f(x" + str(Iterationen) + ")=" + str(round(f_x[-1], Rundungswert)))
        

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

#         function_plot.add_point(x_n_plus_1, f)
#         x_n = x_n_plus_1
#         plot(function_plot)
#         #TODO: Hier werte in der Überschrift
#         #TODO: Hier scheinen sich zwei Kurven zu überlappen
#                 h = np.abs(f(x)/df(x))
#                 self.add_scatter(
#                     x = [x-h, x+h],
#                     y = [f(x) - h*df(x), f(x) + h*df(x)],
#                     mode = "lines",
#                     showlegend = False)
#                 self.function_zoom(min(self.data[0]['x']), max(self.data[0]['x']), f)
                
    def function_zoom(self, f):
        x_axis = np.linspace(self.xmin, self.xmax, 500)
        y_axis = f(x_axis)
        self.update_layout(xaxis_range=[self.xmin, self.xmax], yaxis_range=[min(y_axis), max(y_axis)])
        #self.update_layout(yaxis_range=[ymin, ymax])

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

    def add_gradients(self, gradf, xmin = -4, xmax = 1, ymin = -2, ymax = 3):
        for X in range(xmin, xmax):
            for Y in range(ymin, ymax):
                self.add_annotation(
                    ax=X,  # arrows' head
                    ay=Y,  # arrows' head
                    x=X + gradf([X, Y])[0],  # arrows' tail
                    y=Y + gradf([X, Y])[1],  # arrows' tail
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    text='',  # if you want only the arrow
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=2,
                    arrowcolor='red')

    def add_gradient_descent(self, x0, function, grad, gamma=1, Iterationen=10, color=None, Nebenbedingung=None):
        x = np.zeros(shape=(Iterationen + 1, 2))
        f_x = np.zeros(Iterationen + 1)
        x[0, :] = np.array(x0)
        f_x[0] = np.round(function(x[0, :]), 3)
        for i in range(Iterationen):
            x[i + 1] = -gamma * grad(x[i, :]) + x[i, :]
            f_x[i + 1] = np.round(function(x[i + 1, :]), 3)
        self.add_scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode='lines+markers',
            showlegend=False,
            line_color=color)
        self.result = x[-1]
        if Nebenbedingung is None:
            self.update_layout(title="x0=" + str(np.round(x0, 3)) + ", gamma =" +
                                     str(np.round(gamma, 3)) + ",<br> Iterationen=" + str(Iterationen) +
                                     ", f(x)=" + str(np.round(f_x[-1], 3)) + ", x=" + str(np.round(self.result, 3)))
        else:
            self.update_layout(title="x0=" + str(np.round(x0, 3)) + ", gamma =" +
                                     str(np.round(gamma, 3)) + ",<br> Iterationen=" + str(Iterationen) +
                                     ", f(x)=" + str(np.round(f_x[-1], 3)) + ", h(x) = "
                                     + str(np.round(Nebenbedingung(self.result), 3))
                                     + ",<br> x=" + str(np.round(self.result, 3)))
        self.for_each_trace(
            lambda t: t.update(hovertemplate="x1 %{x}<br>x2 %{y}<extra></extra>"))
        
    def add_gradient_descent_momentum(self, x0, function, grad, gamma=1, Iterationen=10, color=None, Nebenbedingung=None, mu = 0.5, Nesterov = False):
        velocity = np.array([0, 0])
        x = np.zeros(shape=(Iterationen + 1, 2))
        f_x = np.zeros(Iterationen + 1)
        x[0, :] = np.array(x0)
        f_x[0] = np.round(function(x[0, :]), 3)
        for i in range(Iterationen):
            if Nesterov == False:
                velocity = mu * velocity + grad(x[i, :])
                x[i + 1, :] = -gamma * velocity + x[i, :]
            else:
                x_tilde = -gamma * mu * velocity + x[i, :]
                velocity = mu * velocity + grad(x_tilde)
                x[i + 1, :] = -gamma * velocity + x[i, :]
            f_x[i + 1] = np.round(function(x[i + 1, :]), 3)
        self.add_scatter(
            x=x[:, 0],
            y=x[:, 1],
            mode='lines+markers',
            showlegend=False,
            line_color=color)
        self.result = x[-1]
        if Nebenbedingung is None:
            self.update_layout(title="x0=" + str(np.round(x0, 3)) + ", gamma =" +
                                     str(np.round(gamma, 3)) + ",<br> Iterationen=" + str(Iterationen) +
                                     ", f(x)=" + str(np.round(f_x[-1], 3)) + ", x=" + str(np.round(self.result, 3)))
        else:
            self.update_layout(title="x0=" + str(np.round(x0, 3)) + ", gamma =" +
                                     str(np.round(gamma, 3)) + ",<br> Iterationen=" + str(Iterationen) +
                                     ", f(x)=" + str(np.round(f_x[-1], 3)) + ", h(x) = "
                                     + str(np.round(Nebenbedingung(self.result), 3))
                                     + ",<br> x=" + str(np.round(self.result, 3)))
        self.for_each_trace(
            lambda t: t.update(hovertemplate="x1 %{x}<br>x2 %{y}<extra></extra>"))

    def add_h(self):
        def h_(x):
            return x ** 3 + 9 * x ** 2 + 27 * x + 27
        xmin = self.data[0]['x'].min()
        xmax = self.data[0]['x'].max()
        ymin = self.data[0]['y'].min()
        ymax = self.data[0]['y'].max()
        x = [x for x in np.linspace(xmin, xmax, 1000) if ymin <= h_(x) <= ymax]
        x = np.array(x)
        y = h_(x)
        self.add_trace(go.Scatter(x=x, y=y, showlegend=False, marker={'color': '#FF6692'}))

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

    def contour_zoom(self, xmin, xmax, ymin, ymax, function):
        self.data[0]['x'] = x_axis = np.linspace(xmin, xmax, 100)
        self.data[0]['y'] = y_axis = np.linspace(ymin, ymax, 100)
        [x, y] = np.meshgrid(x_axis, y_axis)
        self.data[0]['z'] = function([x, y])
        self.update_layout(xaxis_range=[xmin, xmax])
        self.update_layout(yaxis_range=[ymin, ymax])

    def add_gradient_descent_surface(self, x0, function, grad, gamma=1, Iterationen=10, color=None, Nebenbedingung=None):
        x = np.zeros(shape=(Iterationen + 1, 2))
        f_x = np.zeros(Iterationen + 1)
        x[0, :] = np.array(x0)
        f_x[0] = np.round(function(x[0, :]), 3)
        for i in range(Iterationen):
            x[i + 1] = -gamma * grad(x[i, :]) + x[i, :]
            f_x[i + 1] = np.round(function(x[i + 1, :]), 3)
        self.add_scatter3d(
            x=x[:, 0],
            y=x[:, 1],
            z=f_x,
            #mode='lines+markers',
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