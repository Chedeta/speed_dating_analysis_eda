#!/usr/bin/env python
# coding: utf-8

# # Speed dating data analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import isnan
import plotly.graph_objects as go


df_speed = pd.read_csv('Speed Dating Data.csv')
df_speed.head()
df_speed.describe()


# ## Transformation des indices en texte

hf_dict = {0 : 'Femme', 1 : 'Homme'}
df_speed['gender'] = df_speed['gender'].apply(lambda x: hf_dict.get(x, "Inconnu"))
print(df_speed['gender'])

race_dict = {1 : 'African American', 2 : 'Européen', 3:'Hispanique', 4:'Asiatique',
        5:'Native American', 6:'Autre'}
df_speed['race'] = df_speed['race'].apply(lambda x: race_dict.get(x, "Inconnu"))
print(df_speed['race'])


# ## Les données comportent des doublons, on les supprime par <code>iid</code> pour avoir un aperçu de chaque participant

df_cleaned = df_speed.drop_duplicates('iid').copy()
nb_participants = len(df_cleaned)
hf_count = df_cleaned['gender'].value_counts()


# ## Distribution des âges des participants

age_histplot = sns.histplot(x='age', hue='gender', data=df_cleaned, element='step')
plt.ylabel('Nombre')
print('\nIl y a au total {} participants, dont {} femmes et {} hommes.\n'.format(nb_participants, hf_count['Femme'], hf_count["Homme"]))


# ## Origine des participants

race_plot = sns.catplot(x='race', data=df_cleaned, kind='count')
race_plot.set_xticklabels(rotation=45)
plt.xlabel('Origine')
plt.ylabel('Nombre')
plt.title('Répartition des origines des participants')
race_count = df_cleaned['race'].value_counts().apply(lambda x:(x/nb_participants))
print('\nLes participants sont pour : {:.0%} des Européens, {:.0%} des Asiatiques, {:.0%} des Hispaniques et {:.0%} des African American\n'
      .format(race_count['Européen'],race_count['Asiatique'], race_count['Hispanique'], race_count['African American']))

race_plot_gender = sns.catplot(x='race',hue='gender', data=df_cleaned, kind='count')
race_plot_gender.set_xticklabels(rotation=45)
plt.xlabel('Origine')
plt.ylabel('Nombre')
plt.title('Répartition des origines des participants par sexe')
print("\nLe nombre d'hommes et de femmes est globalement équilibré selon leur origine\n")


# ## Activités et interêts des participants :

activites = ['sports','tvsports','exercise','dining','museums','art','hiking',
              'gaming','clubbing','reading','tv','theater','movies', 'concerts',
              'music','shopping','yoga']

interests_gender = df_cleaned[['gender'] + activites].groupby('gender').mean().transpose()
interests_gender_mean = interests_gender.mean()
interests_gender_plot = interests_gender.plot(kind='bar', figsize=(17,9))
plt.title("Intérêt dans les activités par sexe")
plt.ylabel("Note donnée /10")
plt.xlabel("Activité")
print("\nLes femmes sont globalement plus intéressées par les activités en général, les hommes sont quant à eux plus interessés par le sport, regarder le sport à la TV, et les jeux videos.\n")
plt.show()

interests_race = df_cleaned[['race'] + activites].groupby('race', as_index=False).mean()
interests_race_plot = go.Figure()
buttons=[]

for i in activites:
    interests_race_plot.add_trace(go.Pie(labels=interests_race['race'],
                         values=interests_race[i],
                         visible=True, hole=0.15,  texttemplate="%{value:.2f}<br>%{percent}", textinfo='percent+value'))
    buttons.append(dict(label=i,args = [{'label':[interests_race['race']],'values':[interests_race[i]]}]))
    
interests_race_plot.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.3}], 
        title={'text': "Répartition des notes attribuées aux activités par origine",'x':0.475,'y':0.9},
        font=dict(family="Arial, monospace",size=13))
interests_race_plot.show()


# ## Corrélation entre les intérêts et les match

sns.lmplot(data=df_speed,x='int_corr',y='match')
print("\nPlus les interêts entre les deux participants sont corrélés, plus la variable 'int_corr' se rapproche de 1.\n"
      "Ainsi, on peut voir une légère influence positive de cette variable sur les match.\n")

imprace_avg = df_speed[['imprace','gender','race']].groupby(["race",'gender'], as_index=False).mean()

imprace_avg_plot = sns.catplot(y='imprace', x='race', data=imprace_avg, kind='bar', hue='gender')
imprace_avg_plot.set_xticklabels(rotation=45)
imprace_avg_plot.set(xlabel='Origine', ylabel="Importace de l'origine", title="Importance de l'origine par origine et par sexe")
print("\nLes Européens accordent plus d'importance à l'origine du partenaire, contrairement aux Hispaniques. Une tendance montre que les femmes sont aussi plus "
      "attachées à l'origine du partenaire que les hommes\n")

imprace_gender_avg = df_speed[['imprace','gender']].groupby('gender', as_index=False).mean()

imprace_gender_plot = sns.catplot(data=imprace_gender_avg, kind='bar', y='imprace', x='gender')
imprace_gender_plot.set(ylabel="Importance de l'origine", xlabel="Sexe", title="Importance de l'origine par genre")
print("\nL'origine du partenaire semble être effectivement plus important pour une femme que pour un homme, d'après les notations.\n")


# ## Dans les faits, quelle est la vérité sur l'importance de l'origine ?

# Nous allons étudier l'impact réel d'être de la même origine dans les réponses obtenues pour un date, et non pas selon les déclarations.

race_true_decision = df_speed[['samerace','race','dec']].groupby(['samerace','race'], as_index=False).mean()

race_true_decision_plot = sns.catplot(data= race_true_decision, kind='bar', x='samerace', y='dec', hue='race')
race_true_decision_plot.set(ylabel='Décision après le date', xlabel='Même origine', title='Décision après le date selon si même origine, filtré par origine')
print("\nOn constate une importante augmentation de la probabilité qu'une personne 'African American' dise oui si la personne en face partage son origine."
      " C'est le cas dans une moindre mesure pour les autres origines à l'exception des asiatiques où l'on remarque à l'inverse une baisse\n")


# ## Existe-t-il une différence entre les hommes et les femmes pour ce critère ?

race_true_decision_bis = df_speed[['samerace','race','dec','gender']].groupby(['samerace','race','gender'], as_index=False).mean()
race_femme_true_decision = race_true_decision_bis[race_true_decision_bis['gender']  == 'Femme']
race_homme_true_decision = race_true_decision_bis[race_true_decision_bis['gender']  == 'Homme']

fig, ax = plt.subplots(1,2, figsize=(21,7))
race_femme_true_decision_plot = sns.barplot(data=race_femme_true_decision, x='samerace', hue='race', y='dec', ax=ax[0] , palette = "flare")
race_femme_true_decision_plot.set(ylabel = "Décision", xlabel='Même origine', title="Décision selon si l'origine est la même, chez les femmes, par origine")
race_homme_true_decision_plot = sns.barplot(data=race_homme_true_decision, x='samerace', y='dec', hue='race', ax=ax[1], palette = "flare")
race_homme_true_decision_plot.set(ylabel = "Décision", xlabel='Même origine', title="Décision selon si l'origine est la même, chez les hommes, par origine")
print("\nPlusieurs constats :\n"
        "- Les femmes d'origine Afro-Américaine semblent être plus enclines à dire oui à une personne de la même origine.\n"
        "- Les femmes de façon générale répondent plus favorablement lorsque leur partenaire est de la même origine, ce qui reflète les résultats précédents sur l'importance de l'origine.\n"
        "- Enfin, de façon générale, les hommes donnent une réponse positive à un date plus fréquemment que les femmes.\n")


# ## Quel est l'impact réel de l'origine sur le taux de match ?

sns.lmplot(data=df_speed,x='samerace',y='match')
print("\nAu final, il n'y a pas réellement d'influence persceptible du fait d'être de la même origine sur le taux de match.\n")


# ## Bilan

corr_race_interests = df_speed[['dec','int_corr','samerace','attr']].corr()
sns.heatmap(corr_race_interests, annot=True)
print("En modélisant la matrice de corrélation associée à la variable dec, correspondant au fait que la personne qui reçoit le partenaire dise oui ou non, le résultat montre une très faible corrélation "
      "avec les variables d'interêt et d'origine.\n"
      "La corrélation semble beaucoup plus marquée avec la variable attr (attirance).\n"
      "Ce qui veut dire que les personnes au final ne s'interessent que peu à l'origine de la personne et aux interêts partagés, mais plus à des critères physiques")




