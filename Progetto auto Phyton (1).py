#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importo le librerie utili per l'analisi statistica"
print("librerie caricate")


# # Fase 1: Analisi dei dati
# Osserviamo distribuzioni e relazioni tra le variabili per capire la struttura dei dati.

# In[4]:


tab_auto = pd.read_csv("CarPrice_Assignment.csv")
tab_auto.head()
#Carico il file.csv e ne visualizzo le prime 5 righe

#Controllo lo shape del dataset
print("Dimensioni del dataset:", tab_auto.shape)


# In[5]:


# 1) Creiamo istogrammi per tutte le colonne 

tab_auto.hist(figsize=(12, 8)) #Crea un istogramma per ciascuna colonna specificando (larghezza,altezza)
plt.tight_layout()             #Regola automaticamente gli spazi tra i singoli plot, evitando sovrapposizioni
plt.show()                     #Mi permette di mostrare i grafici


# In[6]:


print("Commento sugli istogrammi:")
print("""Il prezzo delle auto si concentra nei livelli bassi con una presenza tra le auto di medio costo, troviamo poche auto costose.
Le dimensioni (lunghezza, larghezza, peso) sono prevalentemente intorno a un valore medio. Troviamo dei modelli molto grandi o molto pesanti.
Per quanto riguarda il motore ne troviamo di media/bassa cilindrata, con qualche picco per cilindrate alte.
I consumi in città e in autostrada sono abbastanza concentrati su una fascia intermedia.""")


# In[7]:


from pandas.plotting import scatter_matrix
#importo da pandas la funzione scatter_matrix

# 1) Seleziono un sottoinsieme di colonne d’interesse
col_int = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'price']

# 2) Creo la scatter matrix
scatter_matrix(tab_auto[col_int], figsize=(8, 8)) #creo i scatter_plot e ne aggiunsto lunghezza e altezza
plt.show()

#La funzione scatter matrix mi permette di visualizzare la relazione che c'è tra le variabili prima selezionate
#nella lista cols


# In[8]:


print("Commento sugli scatter-plot:")
print("""Da un punto di vista del prezzo notiamo come all'aumentare della cilindrata e potenza del motore aumenti il prezzo delle auto    
Il peso mostra una correlazione moderata con il prezzo: ci sono auto pesanti ma non costose.  """)


# # Fase 2: Pulizia del dataset
# Pulizia e preparazione dei dati per l’addestramento del modello.

# In[9]:


# Copia di sicurezza del dataset originale
auto_prep = tab_auto.copy()

# 1) Rimuovo colonne che non servono per il modello
auto_prep.drop(['car_ID', 'CarName'], axis=1, inplace=True) 
#inplace mi permete di fare le modifiche ditettamente sul dataset senza crearne uno nuovo

# 2) Controllo il nuovo shape del dataset
print("Dimensioni del dataset dopo il drop:", auto_prep.shape)

# 3) Mostro le prime righe per verificare
auto_prep.head()


# In[10]:


# Gestione delle variabili categoriche

# 1) Identifico le colonne categoriche, ossia, variabili testuali che indicano categorie (es. tipo carburante)
col_cat = auto_prep.select_dtypes(include=['object']).columns
print("Colonne categoriche trovate:", list(col_cat))

# 2) Applico il One-Hot Encoding
auto_prep = pd.get_dummies(auto_prep, columns=col_cat, drop_first=True)
# Trasformo le colonne di testo in colonne numeriche 0/1
# Questo permette al modello di interpretare le categorie senza assumere ordini inesistenti


# 3) Controllo il nuovo dataset
print("Nuove dimensioni del dataset:", auto_prep.shape)
auto_prep.head()


# In[11]:


from sklearn.preprocessing import StandardScaler
#importo da sklearn la funzione StandarScaler

# 1) Inizializzo lo scaler (StandardScaler porta media=0 e deviazione standard=1)
scaler = StandardScaler()

# 2) Applico lo scaling a tutte le colonne
auto_scaled = pd.DataFrame(scaler.fit_transform(auto_prep), columns=auto_prep.columns)

#la funzione .fit: calcola la media e la deviazione standard di ogni colonna.
#la funzione transform: usa quei valori per trasformare i dati, portando ogni variabile a media 0 e deviazione standard 1.

# 3) Controllo il risultato
print("Media dopo scaling:\n", auto_scaled.mean().round(2))  #.round arrotonda la media a 2 decimali
print("\nDeviazione standard dopo scaling:\n", auto_scaled.std().round(2)) #.round arrotonda la deviazione standard a 2 decimali
auto_scaled.head()


# In[12]:


print("""Ho standardizzato tutte le variabili del dataset, portandole a media 0 e deviazione standard 1.
In questo modo le feature sono sulla stessa scala e nessuna variabile domina le altre per grandezza dei valori.""")


# # Fase 3: Divisione del dataset
# Separazione delle feature dal target e suddividiamo i dati in training set e test set per addestrare e valutare il modello.

# In[13]:


from sklearn.model_selection import train_test_split

# Separo le variabili indipendenti (X) dalla variabile target (y).
# X conterrà tutte le colonne eccetto 'price', mentre y conterrà solo 'price'.

X = auto_scaled.drop('price', axis=1)   # tutte le colonne tranne 'price'
y = auto_scaled['price']               # solo la colonna 'price'


# In[14]:


# - Training set (80% dei dati) usato per far "imparare" il modello.
# - Test set (20% dei dati) usato per verificare quanto il modello funziona su dati mai visti.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Il parametro random_state=42 serve a ottenere sempre la stessa divisione ad ogni esecuzione.


# In[15]:


# Verifico che le dimensioni di X e y nei set di training e test siano corrette.
# Ci assicuriamo che ogni insieme abbia lo stesso numero di righe per X e y e che le proporzioni 80/20 siano rispettate.

print("Dimensioni training set:", X_train.shape, y_train.shape)
print("Dimensioni test set:", X_test.shape, y_test.shape)


# # Fase 4: Addestramento della Regressione Lineare
# Creo e addestro un modello di regressione lineare utilizzando i dati di training, per poi testarlo sui dati di test.

# In[16]:


# Importo le funzioni dal pacchetto sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Creazione del modello
linreg = LinearRegression()

# 2) Addestramento su training set
linreg.fit(X_train, y_train)
print("\n— Modello addestrato.")

# 3) Predizione sul test set
y_pred = linreg.predict(X_test)
print("— Predizione completata sul test set.")

# 4) Verifiche rapide sui risultati 

mae = mean_absolute_error(y_test, y_pred)
#media dell’errore assoluto. Quanto le previsioni si discostano mediamente dal valore reale, senza considerare il segno.

mse = mean_squared_error(y_test, y_pred)
#media degli errori al quadrato.

rmse = np.sqrt(mse)
#radice quadrata dell’MSE, da più peso agli errori più grandi

r2 = r2_score(y_test, y_pred)
#Coefficiente di determinazione: misura quanto bene il modello spiega la variabilità dei dati.
#Valore compreso tra 0 e 1, dove 1 = perfette

print("\nMetriche sul test set:")
print(f"  MAE  = {mae:.4f}")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R^2  = {r2:.4f}")


# In[17]:


# Interpretazione automatica dei dati
print("\nInterpretazione automatica:")
if rmse > mae * 2:
    print(" RMSE molto più alto del MAE → possibili outlier con errori grandi.")
else:
    print(" RMSE vicino al MAE → nessun segnale forte di outlier.")

if r2 >= 0.9:
    print(" R² ≥ 0.90 → il modello spiega quasi tutta la variabilità dei dati.")
elif r2 >= 0.75:
    print(" R² tra 0.75 e 0.90 → buona capacità predittiva.")
else:
    print(" R² < 0.75 → il modello non spiega bene la variabilità.")

if mae < 0.3:
    print(" MAE basso → previsioni mediamente molto vicine ai valori reali.")
else:
    print(" MAE alto → errori medi da valutare in base al contesto.")


# # Fase 5: Valutazione del modello
# Analizzo le prestazioni con grafici diagnostici e validazione incrociata.

# In[18]:


from scipy import stats
from sklearn.model_selection import KFold, cross_val_score  
#Kfold: definisce come suddividere il training in k parti.
#cross_val_score: esegue il ciclo di CV e restituisce gli score.
from sklearn.dummy import DummyRegressor

#Calcolo dei residui: differenza tra il valore reale del target e il valore stimato dal modello.
resid = np.ravel(y_test) - np.ravel(y_pred)

# Creazione di una figura con 1 riga e 2 colonne
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#Grafico Reale vs Predetto
axes[0].scatter(y_test, y_pred, alpha=0.8)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, 'r--')  # linea ideale
axes[0].set_xlabel("Valori reali")
axes[0].set_ylabel("Valori predetti")
axes[0].set_title("Reale vs Predetto")

# Grafico Residui vs Predetto
axes[1].scatter(y_pred, resid, alpha=0.8)
axes[1].axhline(0, linestyle="--", color='red')  # linea a residuo=0
axes[1].set_xlabel("Valori predetti")
axes[1].set_ylabel("Residui")
axes[1].set_title("Residui vs Predetto")

plt.tight_layout()
plt.show()


# In[19]:


# Creiamo due grafici affiancati: QQ-plot e istogramma
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Il QQ-Plot serve a confrontare la distribuzione dei residui con una distribuzione normale (gaussiana)
# Se i residui seguono una distribuzione normale, i punti si dispongono circa lungo la retta rossa
stats.probplot(resid, dist="norm", plot=axes[0])
axes[0].set_title("QQ-plot dei residui")

# Istogramma dei residui 
# Mostra la frequenza dei residui: ci aspettiamo una forma "a campana" centrata su 0
axes[1].hist(resid, bins=30, edgecolor='black')
axes[1].set_xlabel("Residuo")
axes[1].set_ylabel("Frequenza")
axes[1].set_title("Distribuzione residui")

plt.tight_layout()
plt.show()


# In[20]:


cv = KFold(n_splits=5, shuffle=True, random_state=42)  #cv=cross-validation
# 5-Fold (validazione incrociata) → divido il dataset in 5 parti uguali. 
# In ogni iterazione uso 4 parti per addestrare il modello e la restante per testarlo.
# shuffle=True → prima di dividere, mescolo i dati in modo casuale.
# random_state=42 → fisso un numero di partenza, così ogni esecuzione produce la stessa divisione.

# Calcolo MAE medio con validazione incrociata
mae_scores = -cross_val_score(linreg, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
# Gli errori sono restituiti come valori negativi, quindi aggiungo il segno meno per ottenere valori positivi.

# Calcolo RMSE medio con validazione incrociata
rmse_scores = -cross_val_score(linreg, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
# Stesso discorso del MAE: inverto il segno per ottenere valori positivi.

# Stampa dei risultati
print("\nValidazione incrociata (5-Fold) sul training set:")
print(f"  MAE medio = {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
print(f"  RMSE medio = {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
#Riassume media e deviazione standard delle metriche sulle 5 porzioni di dataset.


# # Fase 6: Interpretazione e conclusioni
# Analizzo i risultati ottenuti e traggo considerazioni finali sulle prestazioni del modello.

# In[21]:


# Sintesi metriche principali dal test set (Fase 4)
print("Metriche sul test set:")
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R²   = {r2:.4f}")

# Interpretazione automatica di base
print("\nInterpretazione:")
if r2 >= 0.9:
    print(" Il modello spiega quasi tutta la variabilità dei dati (R² molto alto).")
elif r2 >= 0.75:
    print(" Il modello ha una buona capacità predittiva (R² sopra 0.75).")
else:
    print(" R² basso: il modello non spiega bene la variabilità, potrebbe essere migliorato.")

if rmse > mae * 2:
    print(" RMSE molto più alto del MAE: possibili outlier con errori grandi.")
else:
    print(" RMSE vicino al MAE: nessun segnale forte di outlier.")

if mae < 0.3:
    print(" MAE basso: previsioni mediamente molto vicine ai valori reali.")
else:
    print(" MAE alto: gli errori medi sono da valutare in base al contesto.")


# In[ ]:





# In[ ]:




