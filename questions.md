
Page 281 : c'est quoi le fichier donnees.csv
Page 289 : mod..predict ou model.predict
Page 294 : SAh ou SAheart
Page 306 : Prev ou Prevention
Page 307 : N.malaria ou Nmalaria ?
Page 314 : modP2 pas défini (dans le tex) ?
Page 323 : SAHEART ou SAheart
2 derniers chunk du chapitre 14 : 1 erreur avec les alphas
Page 347 : **kfaxes**, à supprimer ?
Page 347 : Code ne fonctionne pas, il manque Cs_lasso ?
Page 368 : citation pour smote
Page 369 : remplacer sm1.fit par smote1.fot, ideam pour sm2
Page 373 : changer le nom du fichier "simu_ch15_2_2.csv"
Page 379 : code R à la dernière ligne + calcul terminés pour l'ensemble des métriques ?


Page 242 : problème avec
from ols_step_sk import LinearRegressionSelectionFeatureIC
reg_bic = LinearRegressionSelectionFeatureIC(crit="bic")
reg_bic.fit(X, y)
X.columns[reg_bic.feature_selected_]

Page 193 : procédure de sélection (avec un s à procédures ?)
Page 195 : ajouter lecture du jeu de données ?
Page 175 : section notations trop courte
Page 209 : revoir les sections
Page 375 - Figure 16.5 : importer les données ou dire dans quel fichier se trouve le nuage de points ?
Page 380 dans la remarque : l'argument dist existe bien ? Ou c'est un résidu de R ?
Page 382 : on dit la ou le balanced accuracy (plutot le pour moi ou je pense me tromper) ?
Page 387 : item 3 ajouter une référence pour LogiticRegressionCV



Lasso :
page 206 : on met la pénalité group avec avec une norme euclidienne où le 2 apparait, ce n'est pas le cas pour ridge. On harmonise comment ?
page 208 : $z\in\partial q$, $\partial$ pas définie ?
page 209 : la notation )_+ n'est pas définie ?
page 215 : à partir de "nous allons détailler..." : il faudrait pas supprimer jusqu'à la section suivante ? J'ai l'impression que cette partie date d'avant l'intégration de lasso et ridge ?
page 219 : on dit qu'il existe lassoCV et elasticnetCV mais pas ridgeCV (bizarre). On peut pas utiliser elasticnetCV pour faire du ridgeCV (comme dans R) ? Avec l1_ratio=0 ?
page 219 : "Le choix de la grille est proposée dans la fonction \fct{LassoCV}. Un choix possible pour la grille de ridge correspond à la grille de lasso multipliée par 100,
alors que la grille d'elasticnet est la grille de lasso élevée au carré.", j'ai pas compris ?
page 220 : problème avec le tracé du chemin de régularisation de ridge ?
page 224 : "Dans le cadre de ce chapitre, la régression pénalisée, le problème de la sur
paramétrisation n'est pas un problème en soi et nous avons même évoqué la régression
ridge comme un remède à la surparamétrisation dans la section (\ref{sec:ridgehisto})." Je suis pas tout à fait d'accord, à discuter.
page 227 : exercice 9.8. Changer nom du jeu de données + correction 


Annexe A3 :
chapitre 2 : a t-on besoin de "from mlp_toolkits.mplot3d import Axes3D" ?




p195 : il faudrait pas ajouter 
modsel['Cp'] = modsel['SSR']/modcomplet.mse_resid - modcomplet.nobs + 2 *  modsel['nb_var']

p240 : on a dans le chapitre
from ols_step_sk import LinearRegressionSelectionFeatureIC
ne faut-il pas l'ajouter dans l'annexe ?

p240 : le code en bas ne marche pas ! bestpcr n'existe pas (apparemment calculé plus loin)
