#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 08:15:15 2025

@author: gabrielsaid
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from pyomeca import Markers, Analogs
import Fonction as f
from scipy.constants import G, g #importation de la gravité pour le poids du sujet et du système
from scipy.signal import find_peaks #fonction qui permet de trouver les maximums d'une courbe

#%% Importer tous les fichers de mon sujet
# Récupérer le chemin vers le dossier contenant les données
DAT_PATH = "/Users/gabrielsaid/Documents/ENS/1A/ENS/SVS/Mémoire SVS/Data"
DAT_PATHS = [os.path.join(DAT_PATH, "S1GFA"), os.path.join(DAT_PATH, "S01JP")]
print(f"Répertoires des données : {DAT_PATHS}")
print()


# Paramètres
participant_id = ["S02CT", "S03DM", "S04CL", "S05AG", "S06AP", "S07AB"]
plateform_name = "Amti Gen 5 BP6001200-2K-CT_1"
trialsstr = ["CG", "HG", "MG"]
repsstr = ["1", "2"]
freq_cutoff = 10
freq_samp = 100 #pour un éventuelle lissage

poids_participants = {
    "S02CT": 65.66,
    "S03DM" : 80.15,
    "S04CL": 72.27,
    "S05AG": 93.88,
    "S06AP" : 96.07,
    "S07AB": 52.3
    
}

modalite_prehension = {
    "S02CT": "CG",
    "S03DM": "HG",
    "S04CL": "HG",
    "S05AG": "HG",
    "S06AP": "CG",
    "S07AB": "CG"
}

chemin_participants = {
    "S02CT": os.path.join(DAT_PATH, "S02CT"),
    "S03DM" : os.path.join(DAT_PATH, "S03DM"), 
    "S04CL" : os.path.join(DAT_PATH, "S04CL"),
    "S05AG" : os.path.join(DAT_PATH, "S05AG"),
    "S06AP" : os.path.join(DAT_PATH, "S06AP"), 
    "S07AB" : os.path.join(DAT_PATH, "S07AB")
}
#%% Importer les données et filtrer données
def import_and_lift_data(DAT_PATH, participant, plateform_name, trialstr, repstr):
        file_path = os.path.join(chemin_participants[participant], f"{participant}_{trialstr}_{repstr}.c3d")
        if os.path.exists(file_path):
            rep = Analogs.from_c3d(file_path)
            try : 
                fx = rep.sel(channel=f"{plateform_name}_Fx").values
                fy = rep.sel(channel=f"{plateform_name}_Fy").values
                fz = rep.sel(channel=f"{plateform_name}_Fz").values
                time = rep.time
                
                #Filtrage des données
                valid_mask = ~np.isnan(fx) & ~np.isnan(fy) & ~np.isnan(fz) & ~np.isnan(time)
                fx, fy, fz, time = fx[valid_mask], fy[valid_mask], fz[valid_mask], time[valid_mask]
                representation = np.array([fx, fy, fz]).T
                norms = np.linalg.norm(representation, axis=1)
                return norms, fx, fy, fz, time
            
            except KeyError:
                print(f"Canaux manquants pour {file_path}.")
                return
            else:
                print(f"Fichier introuvable : {file_path}")
                return None
            
#%% Détecter le maximum
def find_maxima(norms, fx, fy, fz, participant, trialstr, repstr):
    # Appliquer les valeurs absolues comme dans votre section graphique
    fx_abs = np.abs(fx)
    fy_abs = np.abs(fy)
    fz_abs = np.abs(fz)
    peaks, _ = find_peaks(norms)
    
    if len(peaks) == 0:  # Vérifier si des pics ont été détectés
        print(f"Aucun pic détecté pour {participant}, {trialstr}, {repstr}.")
        return None
    
    # Trouver le maximum global parmi les maxima détectés
    max_peak_index = peaks[np.argmax(norms[peaks])]  # Indice du maximum global parmi les pics
    
    # Définir une fenêtre de lissage
    fenetre_lissage = 5  # Taille de la fenêtre (5 points)
    start = max(0, max_peak_index - fenetre_lissage // 2)
    end = min(len(norms), max_peak_index + fenetre_lissage // 2 + 1)
    
    # Calculer le maximum lissé
    maximum_lissee = np.mean(norms[start:end])
    
    # Calculer la charge réelle
    poids_sujet = poids_participants[participant]
    charge_reelle = maximum_lissee - ((poids_sujet + 20) * g)  # Calcul de la charge réelle après compensation du poids
    
    # Trouver l'indice du maximum de Fz
    max_fz_index = np.argmax(fz)
    max_fz = fz[max_fz_index]
    
    
    # Récupérer Fx et Fy au moment de Fzmax
    fx_at_max_fz = fx[max_fz_index]
    fy_at_max_fz = fy[max_fz_index]
    
    return max_peak_index, maximum_lissee, charge_reelle, max_fz_index, max_fz, fx_at_max_fz, fy_at_max_fz

#%% Collecter les résultats
results = []
for participant in participant_id:
    for trialstr in trialsstr:
        for repstr in repsstr:
            data = import_and_lift_data(DAT_PATH, participant, plateform_name, trialstr, repstr)
            if data:
                norms, fx, fy, fz, time = data
                # Calculer la norme des forces
                norms = np.linalg.norm(np.array([fx, fy, fz]).T, axis=1)  
                
                # Appliquer les valeurs absolues avant de trouver les maximums
                fx_abs = np.abs(fx)
                fy_abs = np.abs(fy)
                fz_abs = np.abs(fz)
                
                # Trouver les maximums et valeurs associées
                max_results = find_maxima(norms, fx, fy, fz, participant, trialstr, repstr)

                # Vérification de la validité des résultats
                if max_results is not None:
                    max_peak_index, max_lissee, charge_reel, max_fz_index, max_fz, fx_at_max_fz, fy_at_max_fz = max_results
                    # Enregistrer les résultats sous forme de dictionnaire
                    results.append({
                        "Participant": participant,
                        "Modalité_Préférée": modalite_prehension[participant],
                        "Poids_du_Participant": poids_participants[participant],
                        "Condition": trialstr,
                        "Répétition": repstr,
                        "Maximum_lissé": max_lissee,
                        "Charge_réelle": charge_reel,
                        "Fz_max": max_fz,
                        "Fx_at_Fzmax": fx_at_max_fz,
                        "Fy_at_Fzmax": fy_at_max_fz
                    })

#%% Créer un tableau Pandas pour obtenir tous les résultats
print()
df = pd.DataFrame(results)  # Utiliser uniquement les valeurs du dictionnaire
pd.set_option("display.precision", 2)
pd.set_option("colheader_justify", "center")  # Centrer les titres de colonnes
print(df)

# Définir le chemin du dossier où enregistrer le fichier Excel
output_folder = "/Users/gabrielsaid/Documents/ENS/1A/ENS/SVS/Mémoire SVS/Data/Tableur_Excel"
os.makedirs(output_folder, exist_ok=True)  # Crée le dossier s'il n'existe pas déjà

# Construire le chemin complet pour le fichier Excel
excel_file = os.path.join(output_folder, "tous_les_essais_tous_les_sujets.xlsx")

#%% Mettre à jour le DataFrame avec les valeurs correctes
# Ce code doit être placé juste avant de sauvegarder le DataFrame en Excel

# Créer une copie du DataFrame original
df_updated = df.copy()

# Parcourir chaque ligne du DataFrame
for idx, row in df.iterrows():
    participant = row['Participant']
    trialstr = row['Condition']
    repstr = row['Répétition']
    
    # Importer les données de cet essai
    data = import_and_lift_data(DAT_PATH, participant, plateform_name, trialstr, repstr)
    if data:
        norms, fx, fy, fz, time = data
        
        # Appliquer les valeurs absolues comme dans la section graphique
        fx_abs = np.abs(fx)
        fy_abs = np.abs(fy)
        fz_abs = np.abs(fz)
        
        # Trouver l'indice du maximum pour Fz
        max_fz_index = np.argmax(fz_abs)
        max_fz = fz_abs[max_fz_index]
        
        # Récupérer Fx et Fy au moment de Fzmax
        fx_at_max_fz = fx_abs[max_fz_index]
        fy_at_max_fz = fy_abs[max_fz_index]
        
        # Mettre à jour les valeurs dans le DataFrame
        df_updated.loc[idx, 'Fz_max'] = max_fz
        df_updated.loc[idx, 'Fx_at_Fzmax'] = fx_at_max_fz
        df_updated.loc[idx, 'Fy_at_Fzmax'] = fy_at_max_fz

# Remplacer le DataFrame original par le DataFrame mis à jour
df = df_updated

# Maintenant vous pouvez sauvegarder le DataFrame mis à jour

# Enregistrer le fichier Excel dans le dossier choisi
df.to_excel(excel_file, index=False, engine="openpyxl")
print(f"Fichier Excel enregistré dans : {excel_file}")
print()

# Enregistrer un nouveau fichier Excel avec les composantes des forces
composantes_excel_file = os.path.join(output_folder, "composantes_forces_tous_les_essais.xlsx")
df.to_excel(composantes_excel_file, index=False, engine="openpyxl")
print(f"Fichier Excel des composantes de forces enregistré dans : {composantes_excel_file}")
print()

#%%Illustration graphique
## Paramètres de temps

temps_avant_max = 30  # 30 secondes avant le maximum
temps_apres_max = 15  # 15 secondes après le maximum
frequence_echantillonnage = 100  # 100 Hz
# Ajouter une colonne unique pour chaque essai
df["Essai"] = df["Condition"] + "_Rep" + df["Répétition"]

# Définir le dossier pour enregistrer les graphiques
graph_folder = "/Users/gabrielsaid/Documents/ENS/1A/ENS/SVS/Mémoire SVS/Data/Schéma_force_totale"
os.makedirs(graph_folder, exist_ok=True)  # Crée le dossier s'il n'existe pas déjà

# Créer un graphique principal par participant
for participant in participant_id:
    # Créer une figure avec des sous-graphiques
    fig, axs = plt.subplots(1, len(trialsstr), figsize=(15, 6), sharex=False, sharey=False)
    axs = np.atleast_1d(axs)  # Assure que axs est toujours un tableau, même pour un seul sous-graphe
    fig.suptitle(f"Participant {participant}", fontsize=16)

    # Parcourir chaque condition pour afficher le meilleur essai
    for idx, trialstr in enumerate(trialsstr):
        # Sélectionner le meilleur essai pour cette condition
        best_trials = df.loc[df.groupby(["Participant", "Condition", "Essai"])["Maximum_lissé"].idxmax().dropna().values]
        best_trial = best_trials[(best_trials["Participant"] == participant) & (best_trials["Condition"] == trialstr)]
        if not best_trial.empty:
            repstr = best_trial.iloc[0]["Répétition"]

            # Importer les données du meilleur essai pour cette condition
            data = import_and_lift_data(DAT_PATH, participant, plateform_name, trialstr, repstr)
            if data:
                norms, fx, fy, fz, time = data

                # Calculer la norme des forces
                norms = np.linalg.norm(np.array([fx, fy, fz]).T, axis=1)

                # Trouver le pic maximum
                max_results = find_maxima(norms, fx, fy, fz, participant, trialstr, repstr)
                if max_results is not None:
                    max_peak_index, max_lissee, charge_reel, max_fz_index, max_fz, fx_at_max_fz, fy_at_max_fz = max_results

                if max_peak_index is not None:
                    # Calculer les indices correspondant à la fenêtre de 30 secondes avant et 15 secondes après
                    start_time = max(0, max_peak_index - temps_avant_max * frequence_echantillonnage)
                    end_time = min(len(time), max_peak_index + temps_apres_max * frequence_echantillonnage)

                    # Extraire la fenêtre de temps et de données
                    time_window = time[start_time:end_time]
                    norms_window = norms[start_time:end_time]

                    # Tracer les données de force dans le sous-graphe
                    axs[idx].plot(time_window, norms_window, label=f"{trialstr} - Rep {repstr}")

                    # Ajouter un point indiquant le maximum lissé
                    axs[idx].plot(
                        time[max_peak_index], max_lissee, 'ro',
                        label=f"Max Lissé {trialstr}: {max_lissee:.2f} N"
                    )

                    # Ajouter des éléments graphiques spécifiques à chaque sous-graphe
                    axs[idx].set_title(f"{trialstr} (Meilleur Essai)")
                    axs[idx].set_ylabel("Force Normale (N)")
                    axs[idx].legend()
                    axs[idx].grid(True)

                    # Ajuster dynamiquement les limites de l'axe x pour afficher la fenêtre de 45 secondes
                    axs[idx].set_xlim([time[max(0, max_peak_index - temps_avant_max * frequence_echantillonnage)],
                                       time[min(len(time) - 1, max_peak_index + temps_apres_max * frequence_echantillonnage)]])

    # Ajouter des éléments communs au graphique principal
    axs[-1].set_xlabel("Temps (s)")  # L'axe x est commun à tous les sous-graphes
    
    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajuster la mise en page pour inclure le titre

    # Sauvegarder le graphique dans le dossier spécifié
    graph_file = os.path.join(graph_folder, f"Participant_{participant}_force_totale_par_condition.png")
    plt.savefig(graph_file, dpi=300)
    print(f"Graphique enregistré dans : {graph_file}")

    # Afficher le graphique (facultatif)
    plt.show()

#%% Illustration graphique pour les forces x, y et z
# Ajouter une colonne unique pour chaque essai
df["Essai"] = df["Condition"] + "_Rep" + df["Répétition"]

# Définir le dossier pour enregistrer les graphiques
graph_folder = "/Users/gabrielsaid/Documents/ENS/1A/ENS/SVS/Mémoire SVS/Data/Schéma_vecteur_force_x_y_z"
os.makedirs(graph_folder, exist_ok=True)  # Crée le dossier s'il n'existe pas déjà

# Créer un graphique principal par participant
for participant in participant_id:
    # Créer une figure avec des sous-graphiques
    fig, axs = plt.subplots(1, len(trialsstr), figsize=(15, 6), sharex=False, sharey=False)
    axs = np.atleast_1d(axs)  # Assure que axs est toujours un tableau, même pour un seul sous-graphe
    fig.suptitle(f"Participant {participant}", fontsize=16)

    # Parcourir chaque condition pour afficher le meilleur essai
    for idx, trialstr in enumerate(trialsstr):
        # Sélectionner le meilleur essai pour cette condition
        best_trials = df.loc[df.groupby(["Participant", "Condition", "Essai"])["Maximum_lissé"].idxmax().dropna().values]
        best_trial = best_trials[(best_trials["Participant"] == participant) & (best_trials["Condition"] == trialstr)]
        if not best_trial.empty:
            repstr = best_trial.iloc[0]["Répétition"]

            # Importer les données du meilleur essai pour cette condition
            data = import_and_lift_data(DAT_PATH, participant, plateform_name, trialstr, repstr)
            if data:
                norms, fx, fy, fz, time = data

                # Appliquer np.abs() pour obtenir les valeurs absolues des forces
                fx = np.abs(fx)
                fy = np.abs(fy)
                fz = np.abs(fz)
            
                # Trouver le pic maximum pour Fz
                max_fz_index = np.argmax(fz)
                max_fz = fz[max_fz_index]
            
                # Pas de fenêtre de temps autour du maximum, prendre toute la durée
                time_window = time  # Toute la durée
                fx_window = fx
                fy_window = fy
                fz_window = fz
            
                # # Récupérer Fx et Fy au moment de Fzmax
                fx_at_max_fz = fx[max_fz_index]
                fy_at_max_fz = fy[max_fz_index]

            
                # Tracer les courbes de force dans le sous-graphe
                axs[idx].plot(time_window, fx_window, label=f"Fx à Fzmax :{fx_at_max_fz:.2f} N")
                axs[idx].plot(time_window, fy_window, label=f"Fy à Fzmax :{fy_at_max_fz:.2f} N")
                axs[idx].plot(time_window, fz_window, label=f"Fzmax : {max_fz:.2f} N")
            
                # Ajouter des points indiquant les maxima pour chaque axe
                axs[idx].plot(time[max_fz_index], fx_at_max_fz, 'ro')
                axs[idx].plot(time[max_fz_index], fy_at_max_fz, 'go')
                axs[idx].plot(time[max_fz_index], max_fz, 'bo')
            
                # Ajouter des éléments graphiques spécifiques à chaque sous-graphe
                axs[idx].set_title(f"{trialstr} (Meilleur Essai)")
                axs[idx].set_ylabel("Force (N)")
                axs[idx].legend()
                axs[idx].grid(True)

                # Modifier l'axe Y pour qu'il soit positif (croît vers le haut)
                axs[idx].set_ylim(bottom=0)  # Fixe le minimum de l'axe Y à 0

    # Ajouter des éléments communs au graphique principal
    axs[-1].set_xlabel("Temps (s)")  # L'axe x est commun à tous les sous-graphes

    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajuster la mise en page pour inclure le titre

    # Sauvegarder le graphique dans le dossier spécifié
    graph_file = os.path.join(graph_folder, f"Participant_{participant}_force_axes_individuels.png")
    plt.savefig(graph_file, dpi=300)
    print(f"Graphique enregistré dans : {graph_file}")

    # Afficher le graphique (facultatif)
    plt.show()
