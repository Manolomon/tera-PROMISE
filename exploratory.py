# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploración de base de datos de Atributos de Calidad
#
# > Se visitó el 14 de Diciembre de 2019 el sitio http://ctp.di.fct.unl.pt/RE2017/pages/submission/data_papers/
#
# Se encuentran tres *datasets*, sin embargo in interés principal de este estudio es identificar *Quality Attributes*, así que se utiliza la base de datos de [_Quality Attributes_](http://ctp.di.fct.unl.pt/RE2017//downloads/datasets/nfr.arff) que pertenece a [_TeraPROMISE_](https://terapromise.csc.ncsu.edu/!/#repo/view/head/requirements/nfr) y se queda a discusión usar [_SecReq_](http://www.se.uni-hannover.de/pages/en:projekte_re_secreq), que sólo involucra atributos de seguridad.
#
# Un vistazo al dataset se puede observar en la siguiente lectura:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import altair as alt
alt.renderers.enable("default")

from pywaffle import Waffle

# Code for hiding seaborn warnings
import warnings
warnings.filterwarnings("ignore")

# Data read
data = pd.read_csv('nfr.csv')
data

# %% [markdown]
# ## Análisis Exploratorio de Datos

# %%
bars = alt.Chart(data).mark_bar(size=50).encode(
    x=alt.X("class", axis=alt.Axis(title='Categoría')),
    y=alt.Y("count():Q", axis=alt.Axis(title='Frecuencia')),
    tooltip=[alt.Tooltip('count()', title='Frecuencia'), 'class'],
    color='class'

)

text = bars.mark_text(
    align='center',
    baseline='bottom',
).encode(
    text='count()'
)

(bars + text).interactive().properties(
    height=300, 
    width=700,
    title = "Distribución de las Categorías",
)

# %%
data['id'] = 1
df2 = pd.DataFrame(data.groupby('class').count()['id']).reset_index()

bars = alt.Chart(df2).mark_bar(size=50).encode(
    x=alt.X('class'),
    y=alt.Y('PercentOfTotal:Q', axis=alt.Axis(format='.0%', title='Porcentaje de requisitos')),
    color='class'
).transform_window(
    TotalArticles='sum(id)',
    frame=[None, None]
).transform_calculate(
    PercentOfTotal="datum.id / datum.TotalArticles"
)

text = bars.mark_text(
    align='center',
    baseline='bottom',
    #dx=5  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text=alt.Text('PercentOfTotal:Q', format='.2%')
)

(bars + text).interactive().properties(
    height=300, 
    width=700,
    title = "Distribución porcentual de las categorías",
)

# %%
percentage = data['class'].value_counts(normalize=True) * 100
percentage = percentage.round(2)
fig = plt.figure(
    FigureClass=Waffle, 
    columns=60,
    values=data['class'].value_counts(),
    labels=["{0} ({1}%)".format(k, v) for k, v in percentage.items()],
    figsize=(18, 10),
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2), 'ncol': len(data), 'framealpha': 0},
)
plt.title('Distribución porcentual de las categorías')
plt.show()

# %%
data['req_length'] = data['RequirementText'].str.len()

# %%
plt.figure(figsize=(12.8,6))
sns.distplot(data['req_length']).set_title('Distribución de tamaño de requisitos');

# %% [markdown]
# ### Descripción del tamaño de los requisitos

# %%
data['req_length'].describe()

# %%
plt.figure(figsize=(12.8,6))
sns.boxplot(data=data, x='class', y='req_length', width=.5);

# %% [markdown]
# ### Distribución hasta el Cuantil 95

# %%
quantile_95 = data['req_length'].quantile(0.95)
df_95 = data[data['req_length'] < quantile_95]

# %%
plt.figure(figsize=(12.8,6))
sns.distplot(df_95['req_length']).set_title('Distribución de tamaño de requisitos');

# %%
plt.figure(figsize=(12.8,6))
sns.boxplot(data=df_95, x='class', y='req_length');

# %% [markdown]
# ## Distribución de Dataset
#
# Análisis proporcionado en _Automatically Classifying Functional and Non-functional Requirements Using Supervised Machine Learning_
#
# | Categoría | Cantidad | Porcentage | Tamaño |
# | - | -: | -: | -: |
# | Funcional (F) | 255 | 40.80% | 20 |
# | Avalilability (A) | 21 | 3.36% | 19 |
# | Faul Tolerance (FT) | 10 | 1.60% | 19 |
# | Legal (L) | 13 | 2.08% | 18 |
# | Look & Feel (LF) | 38 | 6.08% | 20 |
# | Mantainabilty (MN) | 17 | 2.72% | 28 |
# | Operational (O) | 62 | 9.92% | 20 |
# | Performance (PE) | 54 | 8.64% | 22 |
# | Portability (PO) | 1 | 0.16% | 14 |
# | Scalability (SC) | 21 | 3.36% | 18 |
# | Security (SE) | 66 | 10.56% | 20 |
# | Usability (US) | 67 | 10.72% | 22 |
# | **Total** | **625** | **100%** |  |
