import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# pip install pandas numpy matplotlib seaborn yfinance scipy scikit-learn
import pandas as pd              # Per la gestione di DataFrame, calcoli finanziari (es. rolling, std) e tabelle.
import numpy as np               # Per operazioni matematiche veloci (es. logaritmo, radice quadrata) e istogrammi.
import matplotlib.pyplot as plt # Per la creazione di tutti i grafici.
import seaborn as sns            # Per grafici statistici avanzati e visivamente gradevoli (es. Boxplot, Heatmap).
import yfinance as yf            # Per scaricare i dati finanziari (prezzi) da Yahoo Finance.
from scipy import stats          # Per funzioni statistiche avanzate (es. test di normalità Jarque-Bera)
from sklearn.decomposition import PCA # Per la riduzione della dimensionalità  usato per lo studio del PCA(Principal Component Analysis)
from sklearn.preprocessing import StandardScaler # Per standardizzare le variabili

""" Questo script Python esegue un'analisi finanziaria completa su un set di titoli
 azionari italiani (Top 10 FTSE MIB), scaricando i dati storici, pulendoli e applicando
 diverse tecniche statistiche per valutare rischio e rendimento.

La funzione 'main()' agisce come "gestore" centrale:
1. DOWNLOAD: Definisce i ticker e le date, e scarica i prezzi di chiusura (Close) da Yahoo Finance (yf.download).
2. PULIZIA DATI: Rimuove i dati mancanti (NaN) e riempie i buchi (ffill).
3. RENDIMENTI: Calcola i rendimenti logaritmici sia dei titoli che dell'indice, necessari per le analisi successive.
4. STATISTICHE: Calcola e stampa una tabella riassuntiva (media, varianza, min/max, moda).
5. ESECUZIONE ANALISI: Chiama in sequenza tutti i def per produrre i risultati e i grafici finali."""

# ---------------------------------------------------------
# LISTA E DOWNLOAD DATI
# ---------------------------------------------------------
# Sezione dedicata alla definizione dei titoli, al download e alla pulizia preliminare dei dati.
# Analizziamo i primi 10 titoli per capitalizzazione del FTSE MIB

# Parametro opzionale 'esegui_plot' per evitare il blocco quando usato dal web server
def main(esegui_plot=True):
    # Ticker dell'indice di mercato di riferimento (FTSE MIB)
    market_ticker = 'FTSEMIB.MI'

    # Lista dei ticker azionari da analizzare (Top 10 per capitalizzazione del FTSE MIB)
    tickers = [
        'ENEL.MI',   # Enel
        'ISP.MI',    # Intesa Sanpaolo
        'RACE.MI',   # Ferrari
        'PST.MI',    # Poste Italiane
        'ENI.MI',    # Eni
        'UCG.MI',    # Unicredit
        'BMPS.MI',   # Monte Paschi
        'G.MI',      # Generali
        'LDO.MI',    # Leonardo
        'PRY.MI'     # Prysmian
    ]

    start_date = "2019-01-01"
    print(f"Scarico dati da {start_date}...") #Feedback per l'utente per evitare che pensi che il programma sia bloccato

    prezzi = yf.download(tickers + [market_ticker], # Aggiungo l'indice di mercato alla lista dei ticker
                         start=start_date,
                         auto_adjust=False)['Close'] # Prezzi di chiusura non aggiustati
    # ---------------------------------------------------------
    # PULIZIA DATI
    # Sezione dedicata alla gestione dei dati mancanti (NaN) per garantire analisi accurate.
    # ---------------------------------------------------------
    # Rimuovo colonne completamente vuote
    df_prezzi_clean = prezzi.dropna(axis=1, how='all')

    # Riempio i buchi (festività/sospensioni) tecnica di forward fill
    df_prezzi_clean = df_prezzi_clean.ffill()
    df_prezzi_clean = df_prezzi_clean.dropna() # Rimuovo righe iniziali vuote
    
    # Controlla se l'indice esiste prima di provare a rimuoverlo
    if market_ticker in df_prezzi_clean.columns:
        df_solo_azioni = df_prezzi_clean.drop(columns=[market_ticker])
    else:
        df_solo_azioni = df_prezzi_clean # Se non c'è, prendiamo tutto

    # Aggiorno la lista dei titoli validi in base a cosa è stato scaricato davvero
    valid_stocks = []
    for ticker in tickers:
        if ticker in df_prezzi_clean.columns:
            valid_stocks.append(ticker)
    tickers = valid_stocks  # Aggiorno la lista originale per coerenza

    print(f"\nTitoli scaricati correttamente: {len(tickers)}")
    print(f"I titoli validi sono: {tickers}")

    # Se l'indice di mercato manca, notifico l'errore
    if market_ticker not in df_prezzi_clean.columns:
        print(f"ATTENZIONE: {market_ticker} non trovato. Impossibile scaricare un indice di mercato. L'analisi Beta/PCA potrebbe fallire.")

    # ---------------------------------------------------------
    # ANDAMENTO PREZZO
    # ---------------------------------------------------------
    # Esegue il plot dei prezzi normalizzati dei titoli.
    # Chiama la funzione per disegnare l'andamento dei prezzi normalizzati a Base 100
    
    # Eseguo i grafici solo se esegui_plot è True
    if esegui_plot:
        andamento_titoli (df_solo_azioni) # Andamento prezzi azione

    # ---------------------------------------------------------
    # CALCOLO RENDIMENTI
    # ---------------------------------------------------------
    # Preparazione del set di dati per le analisi finanziarie, calcolando i rendimenti logaritmici.

    # Calcolo dei rendimenti logaritmici: log(Pt / Pt-1).
    log_returns = np.log(df_prezzi_clean / df_prezzi_clean.shift(1)).dropna() # Rendimenti dei titoli e dell'inidce di mercato

    log_rendimenti_titoli = log_returns[tickers] # Solo i rendimenti dei titoli, escludendo l'indice di mercato

    # ---------------------------------------------------------
    # ANALISI  DEI RENDIMENTI
    # ---------------------------------------------------------
    # Calcolo delle principali statistiche dei rendimenti (media, varianza, min/max, moda).
    # rendimenti cumulativi per ogni titolo
    rendimenti_cumulativi = (log_rendimenti_titoli + 1).cumprod()

    # media varianza deviazione standard min max
    media_rendimenti = log_rendimenti_titoli.mean() * 252  # Annualizzo la media
    deviazione_standard = log_rendimenti_titoli.std() * np.sqrt(252)  # Annualizzo la deviazione standard
    varianza_rendimenti = log_rendimenti_titoli.var() * 252  # Annualizzo la varianza
    min_rendimenti = log_rendimenti_titoli.min() # Minimo rendimento giornaliero
    max_rendimenti = log_rendimenti_titoli.max() # Massimo rendimento giornaliero
    # moda dei rendimenti
    # moda_rendimenti_originale = log_rendimenti_titoli.mode().iloc[0] # .iloc[0] per ottenere la prima riga (moda)
    # Nuova Moda Statistica (raggruppata titolo per titolo)
    moda_statistica = log_rendimenti_titoli.apply(calculate_modal_return, axis=0)

    #stampo tabella riassuntiva
    riepilogo = pd.DataFrame({
        'Rendimenti Cumulativi Finali': rendimenti_cumulativi.iloc[-1], # iloc[-1] per l'ultimo valore
        'Media Annualizzata': media_rendimenti,
        'Deviazione Standard Annualizzata': deviazione_standard,
        'Varianza Annualizzata': varianza_rendimenti,
        'Min Rendimento Giornaliero': min_rendimenti,
        'Max Rendimento Giornaliero': max_rendimenti,
        'Moda Statistica (Raggruppata)': moda_statistica,
    })
    
    print("\nRiepilogo Statistiche dei Rendimenti:")
    print(riepilogo)

    # ---------------------------------------------------------
    # ANALISI AVANZATE
    # ---------------------------------------------------------
    # Esegue le analisi più complesse attraverso le funzioni dedicate
    # Eseguo le funzioni di plot solo se richiesto
    if esegui_plot:
        # Statistiche Avanzate (Skew, Kurtosis, Test Normalità)
        statistiche_avanzate(log_rendimenti_titoli)
        # Analisi Distribuzione (Boxplot, Istogrammi, Q-Q Plot)
        analisi_distribuzione(log_rendimenti_titoli)
        # Matrice di correlazione e Heatmap
        matrice_correlazione_heatmap(log_rendimenti_titoli)
        # PCA
        principal_component_analysis(log_rendimenti_titoli, tickers)
        # Calcolo Beta (solo se il mercato esiste)
        if market_ticker in log_returns.columns:
            calcolo_beta(tickers, market_ticker, log_returns)
        # Volatilità Rolling
        studio_volatilita(log_rendimenti_titoli)
    
    # Restituisce anche 'riepilogo' perché serve a Flask per riempire la tabella HTML
    return df_prezzi_clean, df_solo_azioni, tickers, market_ticker, riepilogo


# ---------------------------------------------------------
# CREAZIONE GRAFICO ANDAMENTO DEI PREZZI NORMALIZZATI
# ---------------------------------------------------------
def andamento_titoli (df_solo_azioni):
    """
    Genera un grafico a linee dell'andamento dei prezzi dei titoli normalizzati a Base 100.
    
    La normalizzazione permette di confrontare la performance relativa di tutti i titoli
    a partire da un punto iniziale comune, ignorando le differenze assolute di prezzo.
    Aggiunge anche annotazioni per eventi macroeconomici rilevanti.

    Args:
        df_solo_azioni (pd.DataFrame): DataFrame contenente solo i prezzi di chiusura
                                       puliti e completi dei titoli azionari (senza l'indice di mercato).

    Returns:
        None: La funzione visualizza direttamente il grafico con Matplotlib.
    """
    plt.figure(figsize=(14, 8))

    # Normalizzazione a Base 100 (così partono tutti dallo stesso punto)
    prezzi_normalized = (df_solo_azioni / df_solo_azioni.iloc[0]) * 100

    for column in prezzi_normalized.columns:
        plt.plot(prezzi_normalized.index, prezzi_normalized[column], label=column, linewidth=1.5)

    plt.title('Confronto Andamento Prezzi Top 10 FTSE MIB (Base 100)')
    plt.xlabel('Data')
    plt.ylabel('Performance (100 = Parità)')
    plt.axvline(pd.to_datetime('2020-03-01'), color='black', linestyle='--', linewidth=0.8) #  Disegna una linea verticale tratteggiata in corrispondenza della data esatta
    plt.text(pd.to_datetime('2020-03-01') -  pd.Timedelta(days=200), plt.ylim()[1]*0.85, 'COVID-19', color='black')  # Posiziona etichetta: spostata indietro di 200gg (X) e all'85% dell'altezza (Y)
    plt.axvline(pd.to_datetime('2025-03-27'), color='black', linestyle='--', linewidth=0.8)
    plt.text(pd.to_datetime('2025-04-02') -  pd.Timedelta(days=200), plt.ylim()[1]*0.85, 'Dazi Trump', color='black')
    plt.axvline(pd.to_datetime('2022-02-24'), color='black', linestyle='--', linewidth=0.8)
    plt.text(pd.to_datetime('2022-02-24') -  pd.Timedelta(days=280), plt.ylim()[1]*0.85, 'Guerra in Ucraina', color='black')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()

# ---------------------------------------------------------
# FUNZIONE PER CALCOLARE LA MODA STATISTICA TITOLO PER TITOLO
# ---------------------------------------------------------
def calculate_modal_return(series, bins=50):
    """
    Calcola il punto centrale della classe di rendimento più frequente
    (Moda Statistica) per una singola serie di rendimenti, tramite l'analisi di un istogramma.

    Args:
        series (pd.Series): La serie di dati dei rendimenti (tipicamente logaritmici)
                             di un singolo titolo.
        bins (int, optional): Il numero di classi (bin) da utilizzare per l'istogramma.
                              Il valore predefinito è 50.
    
    Returns:
        float: Il punto medio della classe con la frequenza più alta (la moda statistica).
               Restituisce 0.0 se la serie è vuota o ha varianza zero.
    """
    # Se la serie è vuota o ha varianza zero, restituisce 0
    if series.empty or series.var() == 0:
        return 0.0

    # Calcola l'istogramma solo per questa serie
    counts, bin_edges = np.histogram(series.values, bins=bins)

    # Trova l'indice della classe (bin) con la frequenza massima
    max_count_index = np.argmax(counts)

    # Calcola il punto medio di questa classe modale
    modal_return = (bin_edges[max_count_index] + bin_edges[max_count_index + 1]) / 2

    return modal_return

# ---------------------------------------------------------
# FUNZIONE PER CALCOLARE LE STATISTICHE AVANZATE
# ---------------------------------------------------------
def statistiche_avanzate(rendimenti):
    """
    Calcola skewness (asimmetria), kurtosis (curtosi) e il p-value del test
    di normalità di Jarque-Bera per ogni titolo.
    
    Args:
    rendimenti (pd.DataFrame): DataFrame contenente i rendimenti logaritmici
                               dei titoli azionari (escluso l'indice di mercato).
    
    Returns:
        pd.DataFrame: Un DataFrame riepilogativo contenente le colonne 'Skewness',
                      'Kurtosis', 'JB p-value' e 'Normale?' (la decisione SÌ/NO).
    """

    print("\n--- Statistiche Avanzate (Skewness, Kurtosis, Normalità) ---")

   
    stats_df = pd.DataFrame(index=rendimenti.columns)  # Inizializza un DataFrame vuoto usando i nomi dei titoli (colonne) come indice
    stats_df['Skewness'] = rendimenti.skew() # Calcola asimmetria 
    stats_df['Kurtosis'] = rendimenti.kurtosis()  # Calcola la curtosi. Una normale ha kurtosis = 0 (in pandas è l'eccesso)

    # Test di Jarque-Bera per la normalità (p-value < 0.05 = NON è normale)
    jb_pvalues = [] # Inizializza una lista vuota per conservare il p-value del test di Jarque-Bera per ogni titolo
    is_normal = [] # Inizializza una lista vuota per registrare il risultato finale del test ("Sì" o "No" è Normale).
    for col in rendimenti.columns:
        jb_stat, p_value = stats.jarque_bera(rendimenti[col]) # Calcola la statistica del test (jb_stat) e il p-value.
        jb_pvalues.append(p_value)
        is_normal.append("Sì" if p_value > 0.05 else "No") # Aggiunge il risultato del test alla lista 'is_normal' in base al p-value

    stats_df['JB p-value'] = jb_pvalues # Aggiunge i p-value calcolati come nuova colonna ('JB p-value') al DataFrame statistico 'stats_df'.
    stats_df['Normale?'] = is_normal # Aggiunge i risultati della decisione ('Sì'/'No') come nuova colonna ('Normale?') in 'stats_df'.

    print(stats_df)
    return stats_df

# ---------------------------------------------------------
# FUNZIONE ANALISI DISTRIBUZIONE
# ---------------------------------------------------------
def analisi_distribuzione(rendimenti):
    """
    Genera grafici della distribuzione: Istogrammi, Curva Normale, Boxplot e Q-Q Plot.
    Args:
        rendimenti (pd.DataFrame): DataFrame contenente i rendimenti logaritmici
                                   dei titoli azionari.
    Returns:
        None: La funzione visualizza direttamente i tre tipi di grafici con Matplotlib/Seaborn.   
    """

    print("\n--- 4 & 6. Analisi Grafica della Distribuzione ---")

    # 1. Boxplot Comparativo (per vedere gli outlier)
    plt.figure(figsize=(14, 6)) # Crea una "figura" con le rispettive dimensioni
    sns.boxplot(data=rendimenti, orient="h", palette="Set2") # Crea il BoxPlot che poi verrà raffigurato basato sui rendiementi dei titoli
    plt.title("Boxplot dei Rendimenti (Identificazione Outlier)")
    plt.xlabel("Rendimento Giornaliero Log")
    plt.tight_layout()
    plt.show()

    # 2. Istogrammi + Fitting Normale (Solo per 3 titoli rappresentativi per non intasare)
    # Selezioniamo: Il più volatile, il meno volatile e uno medio
    std_dev = rendimenti.std().sort_values() # Calcola la deviazione standard per ogni titolo
    titoli_sample = [std_dev.index[0], std_dev.index[len(std_dev)//2], std_dev.index[-1]] 

    plt.figure(figsize=(15, 5))
    for i, ticker in enumerate(titoli_sample): # Ciclo for per creare più grafici, uno per ogni ticker in ordine
        plt.subplot(1, 3, i+1) # Questa riga posiziona ogni grafico in un blocco della figura
        # Istogramma dati reali
        sns.histplot(rendimenti[ticker], kde=False, stat="density", color="skyblue", label="Reale")  # Tramite il pacchetto seaborn creo l'istogramma
        #                                                                                               con i rendiementi dei titoli

        # Fit Curva Normale
        mu, std = stats.norm.fit(rendimenti[ticker]) # mu  → media stimata dal campione
        #                                               std → deviazione standard stimata
        #                                               stats.norm.fit trova i parametri migliori della normale per adattarsi ai dati  
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100) # Genera 100 valori equidistanti sull'intervallo dell'istogramma
        p = stats.norm.pdf(x, mu, std)  # Calcola la densità teorica della normale utilizzando media e deviazione standard ottenute sopra
        plt.plot(x, p, 'r', linewidth=2, label="Normale Teorica")

        plt.title(f"Distribuzione: {ticker}")
        plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Q-Q Plot (Confronto code pesanti)
    # Mostriamo solo quello del titolo più volatile (spesso il più interessante)
    risky_stock = std_dev.index[-1] # Prende il considerazione il titolo più volatile
    plt.figure(figsize=(8, 6))
    stats.probplot(rendimenti[risky_stock], dist="norm", plot=plt) # Confronta i quantili dei rendimenti osservati
    #                                                                con quelli della distribuzione normale teorica.
    #                                                                Se i punti cadono sulla diagonale avremmo una distribuzione ≈ normale
    plt.title(f"Q-Q Plot: {risky_stock} (Code Pesanti)")
    plt.grid(True)
    plt.show()

# ---------------------------------------------------------
# FUNZIONE CALCOLO MATRICE CORRELAZIONE
# ---------------------------------------------------------

def matrice_correlazione_heatmap(returns_stocks):
    """
    Calcola la matrice di correlazione dei rendimenti azionari e genera una Heatmap.
 
    Questa funzione prende in input un DataFrame di rendimenti, calcola la correlazione
    di Pearson tra le varie colonne (titoli) e visualizza il risultato tramite
    una mappa di calore (heatmap) per identificare cluster e relazioni tra i titoli.

    Arg:
        returns_stocks (pd.DataFrame): DataFrame contenente i rendimenti logaritmici
                                       o percentuali dei titoli (senza l'indice di mercato).

    Returns:
        None: La funzione visualizza direttamente il grafico.   
    """
    corr_matrix = returns_stocks.corr() # Calcola la matrice di correlazione
    #                                     Ogni valore varia tra: +1 = correlazione perfetta positiva, 0  = assenza di correlazione, -1 = correlazione perfetta negativa

    # Creo la Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix,
                annot=True,      # Mostra i numeri
                cmap='coolwarm', # Colore: Rosso (correlati) - Blu (inversi)
                fmt=".2f",       # 2 cifre decimali
                linewidths=0.5)
    plt.title('Heatmap di Correlazione dei Rendimenti (Top 10 FTSE MIB)', fontsize=16)
    plt.show()

# ---------------------------------------------------------
# FUNZIONE PCA
# ---------------------------------------------------------

def principal_component_analysis(returns_stocks, valid_stocks=None):
    """
    Esegue l'Analisi delle Componenti Principali (PCA) sui rendimenti azionari.
    
    La funzione standardizza i dati, applica la PCA per ridurre la dimensionalità
    e visualizza la varianza spiegata cumulativa. Serve a comprendere quanto i
    rendimenti siano guidati da fattori comuni (es. "Fattore Mercato").

    Args:
        returns_stocks (pd.DataFrame): DataFrame dei rendimenti dei titoli.
        valid_stocks (list, optional): Lista dei ticker validi utilizzata per determinare
                                       il numero di componenti da calcolare. Se None, usa
                                       tutte le colonne.

    Returns:
        None: La funzione visualizza il grafico della varianza spiegata e stampa
              la percentuale di rischio spiegata dalla prima componente.
    """
    # Per standardizzare i dati uso StandardScaler e PCA per l'analisi delle componenti principali
    scaler = StandardScaler() # Standardizzo i dati (Mean=0, Var=1) per la classe StandardScaler
    returns_scaled = scaler.fit_transform(returns_stocks) # Matrice standardizzata
    pca = PCA(n_components=len(valid_stocks)) # Numero di componenti per la PCA (classe)
    pca.fit(returns_scaled) # Adatto il modello PCA ai dati standardizzati
    #'explained_variance_ratio_' mostra la % di varianza catturata da ogni singola componente (dalla più importante alla meno).
   # 'cumsum' somma queste percentuali progressivamente per capire quante componenti servono a spiegare il totale.
    expl_var = np.cumsum(pca.explained_variance_ratio_) * 100

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(valid_stocks) + 1), expl_var, marker='o', linestyle='--') # inizio da 1 per una migliore leggibilità del grafico
    plt.axhline(y=80, color='r', linestyle='--', label='80% Var')
    plt.title('PCA: Varianza Spiegata')
    plt.xlabel('Componenti')
    plt.ylabel('Varianza Cumulativa %')
    plt.legend()
    plt.grid()
    plt.show()

# ---------------------------------------------------------
# FUNZIONE BETA
# ---------------------------------------------------------

def calcolo_beta(tickers=None, market_ticker=None, log_returns=None):
    """
    Calcola e visualizza il Beta (rischio sistematico) per una lista di titoli.
    
    Il Beta misura la reattività del rendimento di un titolo rispetto ai movimenti
    dell'indice di mercato.
    - Beta > 1: Titolo aggressivo (amplifica i movimenti).
    - Beta < 1: Titolo difensivo (smorza i movimenti).

    Args:
        tickers (list): Lista dei simboli (ticker) dei titoli azionari da analizzare.
        market_ticker (str): Il simbolo dell'indice di mercato di riferimento (es. 'FTSEMIB.MI').
        log_returns (pd.DataFrame): DataFrame contenente i rendimenti giornalieri sia dei
                                    titoli che dell'indice di mercato.

    Returns:
        None: La funzione visualizza un grafico a barre dei Beta e stampa i valori.
    """
    betas = {}
    for stock in tickers:
        # Covarianza tra titolo e mercato
        cov = log_returns[[stock, market_ticker]].cov().iloc[0, 1] #.iloc[0, 1] per ottenere la covarianza tra le due serie
        # Varianza del mercato
        market_var = log_returns[market_ticker].var() # Varianza storica del mercato
        beta = cov / market_var # Calcolo del Beta
        betas[stock] = beta

    # Visualizzo i Beta in un grafico a barre
    beta_series = pd.Series(betas).sort_values(ascending=False) # Trasforma questo dizionario in una series di Pandas

    plt.figure(figsize=(10, 6))
    plt.bar(beta_series.index, beta_series.values, color='orange', edgecolor='black')
    plt.xticks(rotation=90)

    plt.axhline(y=1, color='r', linestyle='--', label='Mercato (Beta = 1)')
    plt.title('Beta dei Titoli rispetto al FTSE MIB', fontsize=14)
    plt.ylabel('Beta')
    plt.legend()
    plt.show()

    print("Valori di Beta:")
    print(beta_series)


# ---------------------------------------------------------
# FUNZIONE VOLATILITA' e GRAFICO
# ---------------------------------------------------------
def studio_volatilita(rendimenti):
    """
    Analizza la volatilità storica e rolling (finestre mobili) dei rendimenti azionari.
    Svolgiamo lo studio della volatilità in quanto la deviaizone standard può essere considerata come volatilità annualizzata.

    La funzione calcola la deviazione standard annualizzata su diverse finestre temporali
    (breve, medio, lungo termine) per identificare la stabilità e il rischio dei titoli.
    Confronta inoltre visivamente il titolo più volatile con quello meno volatile.

    Args:
        rendimenti (pd.DataFrame): DataFrame contenente i rendimenti logaritmici giornalieri 
                                   dei titoli e/o dell'indice.

    Returns:
        None: La funzione genera un grafico comparativo e stampa a video le statistiche 
              e le classifiche di volatilità.
    """
    
    print("\n--- Studio della Volatilità Rolling ---") 
    
    # Giorni di trading in un anno
    trading_days_per_year = 252

    # Volatilità a breve (30 giorni) vs Lungo (120 giorni) vs Medio (60 giorni)
    vol_30gg = rendimenti.rolling(window=30).std() * np.sqrt(trading_days_per_year) # Volatilità a breve termine
    vol_60gg = rendimenti.rolling(window=60).std() * np.sqrt(trading_days_per_year)  # Volatilità a medio termine
    vol_120gg = rendimenti.rolling(window=120).std() * np.sqrt(trading_days_per_year)  # Volatilità a lungo termine

    # Intestazione sezione output
    print("## Volatilità Rolling (Ultimi 5 giorni)")
    print("\n### Volatilità a 30 Giorni (Ultimi 5)")
    print(vol_30gg.tail())

    print("\n### Volatilità a 60 Giorni (Ultimi 5)")
    print(vol_60gg.tail())

    print("\n### Volatilità a 120 Giorni (Ultimi 5)")
    print(vol_120gg.tail())
    # La deviazione standard annualizzata su tutto il periodo
    volatilita_media_storica = rendimenti.std() * np.sqrt(trading_days_per_year) # Volatilità storica annualizzata

    # Creazione del DataFrame di classifica
    df_classifica = pd.DataFrame(volatilita_media_storica, columns=['Volatilità Media Storica']).sort_values(
        by='Volatilità Media Storica', ascending=False
    )

    print("\n## Classifica dei Titoli per Volatilità Media Storica")
    print("---")

    # Visualizzazione con precisione limitata per una migliore leggibilità
    with pd.option_context('display.float_format', '{:.4f}'.format): # Limita decimali a 4 per leggibilità
        print(df_classifica)

    plt.figure(figsize=(14, 7))
    # Plot solo per 2 titoli principali per leggibilità
    # 1. Seleziona il titolo con la Volatilità Media Storica MASSIMA (primo in classifica)
    stock_max_vol = df_classifica.index[0]
    # 2. Seleziona il titolo con la Volatilità Media Storica MINIMA (ultimo in classifica)
    stock_min_vol = df_classifica.index[-1]
    # 3. Definisco la lista dei titoli da plottare (massimo e minimo)
    focus_stocks = [stock_max_vol, stock_min_vol]

    # La funzione 'enumerate()' fornisce sia l'indice numerico 'i' che il valore 'stock'  per ogni elemento di 'focus_stocks'.
    # Sebbene l'indice 'i' non sia usato nel plot, 'stock' è essenziale per selezionare la colonna corretta (il nome del titolo
    # azionario) dai DataFrames di volatilità 'vol_30gg' e 'vol_120gg' e per creare le etichette della legenda (label).
    for i, stock in enumerate(focus_stocks):
        plt.plot(vol_30gg.index, vol_30gg[stock], label=f'{stock} Vol 30gg', linestyle='-', alpha=0.8)
        plt.plot(vol_120gg.index, vol_120gg[stock], label=f'{stock} Vol 120gg', linestyle=':', linewidth=2)

    plt.title('Confronto Volatilità Rolling: 30 vs 120 giorni')
    plt.ylabel('Volatilità Annualizzata')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Confronto dei Pattern di Volatilità tra Titoli
    # Combinazione delle serie di volatilità rolling a 60 giorni
    df_vol_patterns = vol_60gg.dropna()

    print("\n## Confronto dei Pattern di Volatilità (Rolling 60gg)")
    print("---")
    print("Mostra le prime 5 e le ultime 5 osservazioni del pattern di volatilità a 60 giorni:")
    print(df_vol_patterns.head())
    print("...")
    print(df_vol_patterns.tail())

    print("\nAnalisi degli estremi di Volatilità (60gg):")
    print(f"Titoli con la Volatilità MINIMA (al 60gg roll-out): \n{df_vol_patterns.min().sort_values().head(3)}") 
    print(f"Titoli con la Volatilità MASSIMA (al 60gg roll-out): \n{df_vol_patterns.max().sort_values(ascending=False).head(3)}")

# Questo blocco fa sì che se avvii 'analisi.py' direttamente, 
# i grafici appaiono. Se invece lo importi da Flask, 'main' viene chiamato
# con esegui_plot=False (da dentro app.py) e i grafici NON bloccano il server.
if __name__ == "__main__":
    df_prezzi_totali, df_solo_azioni, lista_tickers, market_ticker, riepilogo = main(esegui_plot=True)
