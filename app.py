import analisi  # Importa il tuo file analisi.py e usa le funzioni che ci sono dentro
from flask import Flask, render_template, request  
#flask è l'applicazione vera e propria, render_template serve per prendere i file html e request legge cosa l'utente ha inviato (serve per le date)
import matplotlib
matplotlib.use('Agg') # fondamentale per non fare aprire le finistre pop-up sullo schermo e non fare andare in crash il server
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io # necessario per salvare le immagini dei grafici in memoria (senza creare file)
import base64 #traforma l'immagine in una stringa di testo leggibile dall'HTML
from scipy import stats
import seaborn as sns
from datetime import timedelta # necessario per gestire le date

app = Flask(__name__) # Inizializza l'applicazione Flask (crea l'oggetto 'app' che gestirà il sito)

# ---------------------------------------------------------
# CARICAMENTO DATI ALL'AVVIO
# ---------------------------------------------------------
# Chiamiamo la funzione che abbiamo creato nel file analisi.py
print("Avvio server: inizio scaricamente dati...")

# 1. Recupero Dati
# Chiamiamo la funzione 'main' dal file analisi.py.
# Impostiamo esegui_plot=False perché al server servono solo i dati numerici, non i grafici a schermo.
# Le variabili restituite (df_prezzi_totali, ecc.) diventano variabili GLOBALI accessibili da tutte le pagine del sito
df_prezzi_totali, df_solo_azioni, lista_tickers, market_ticker, df_riepilogo = analisi.main(esegui_plot=False)

# 2. Pulizia Dati
# Pandas a volte legge le date come semplici stringhe di testo.
# Qui forziamo la conversione dell'indice in oggetti 'datetime' veri e propri.
df_solo_azioni.index = pd.to_datetime(df_solo_azioni.index)

print("Dati scaricati e pronti.")

# ---------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------
@app.route("/") #Decoratore dice a flask dove andare
def index():
    """
    Route per la Homepage.
    Visualizza la tabella riepilogativa dei titoli.
    """
    # Il file HTML non sa leggere i DataFrame di Pandas. 
    # Convertiamo la tabella 'df_riepilogo' in un dizionario Python standard.
    # orient='index' crea una struttura dove la chiave principale è il Ticker (es. ENEL.MI),
    # rendendo facile per l'HTML creare una riga della tabella per ogni azione.
    dati_finanziari = df_riepilogo.to_dict(orient='index')
    
    # Reindirizziamo/passiamo il dizionario appena creato alla pagina 'index1.html'
    return render_template("index1.html", dati_finanziari=dati_finanziari)

# Indirizzo dinamico in relazione al ticker
# Con methods=['GET','POST'] abilito l'interattività
# GET quando apro la pagina per la prima volta  
# POST quando l'utente compiila il form delle date e clicca aggiorna
@app.route("/grafico1/<ticker>", methods=['GET', 'POST']) 
def grafico_ticker(ticker):
    """
    Route per il dettaglio del singolo Ticker.
    Gestisce sia la visualizzazione iniziale (GET) che l'aggiornamento date (POST).
    Genera grafici temporali e di distribuzione 'al volo'.
    """
    # 1. Validazione Ticker
    # Prima di fare qualsiasi calcolo, controlliamo se il ticker richiesto esiste nel nostro DataFrame
    if ticker not in df_solo_azioni.columns:
        return f"<h1>Errore</h1><p>Titolo {ticker} non trovato o dati insufficienti.</p>", 404

    # Recuperiamo la serie completa per il Ticker richesto
    prezzi_azione_full = df_solo_azioni[ticker]

    # 2. Gestione Date
    # Troviamo l'ultima data disponibile nei dati per calcolare i default coerenti
    max_date = prezzi_azione_full.index.max()
    
    # Calcolo date di default: impostiamo la vista sugli ultimi 365 giorni
    default_start = (max_date - timedelta(days=365)).strftime('%Y-%m-%d')
    default_end = max_date.strftime('%Y-%m-%d')

    # Qui avviene lo "switch" logico tra prima visita e aggiornamento:
    if request.method == 'POST':
        # Se l'utente clicca aggiorna, prendiamo le date dal form
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
    else:
        # Altrimenti usiamo i default
        start_date = default_start
        end_date = default_end

    # 2. Prepara i dati specifici (FILTRATI PER DATA)
    # Usiamo .loc per tagliare il dataframe in base alle date scelte
    try:
        #Filtriamo i dati in base al range selezionato
        prezzi_azione = prezzi_azione_full.loc[start_date:end_date]
    except:
        # Fallback in caso di formato date non valido o errori di slicing
        prezzi_azione = prezzi_azione_full # Fallback in caso di errore date

    # Controllo dati insufficienti dopo il filtro
    # Servono almeno 2 punti per calcolare un rendimento (oggi vs ieri).
    if len(prezzi_azione) < 2:
        return f"<h1>Errore</h1><p>Dati insufficienti per il periodo selezionato.</p>"

    #Calcolo Rendimenti logaritmici
    rendimento_log = np.log(prezzi_azione / prezzi_azione.shift(1)).dropna()

    # ==========================================
    # GRAFICO 1: TEMPORALE (Prezzo + Rendimento + Volatilità)
    # ==========================================
    fig1, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True) 
    
    # Prezzo
    axes[0].plot(prezzi_azione, color="steelblue", label='Prezzo')
    axes[0].set_title(f'{ticker} - Analisi Temporale', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Rendimenti Log
    axes[1].plot(rendimento_log, color='seagreen', lw=0.8, label='Rendimenti Log')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # Volatilità Rolling 30gg
    volatilita = rendimento_log.rolling(30).std() * np.sqrt(252)
    axes[2].plot(volatilita, color='lightsalmon', label='Volatilità (30gg)')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvataggio in buffer di memoria e conversione base64
    # Creazione del "File Virtuale"
    # io.BytesIO() crea un buffer in memoria RAM.
    img1 = io.BytesIO()
    # Scriviamo l'immagine nel buffer appena creato.
    # bbox_inches='tight' rimuove i bordi bianchi inutili intorno al grafico.
    fig1.savefig(img1, format='png', bbox_inches='tight')
    # Dopo aver salvato, il "cursore" di lettura è alla fine del file.
    # Dobbiamo riportarlo all'inizio (seek 0) per poter leggere i dati dall'inizio.
    img1.seek(0)
    # Matplotlib mantiene i grafici aperti in memoria finché non gli dici di chiuderli.
    # Senza plt.close(), dopo molte visite al sito, la RAM del server si riempirebbe (Memory Leak).
    plt.close(fig1) 
    # Trasformiamo i dati binari (i bit dell'immagine) in una stringa di testo.
    # .decode() serve a trasformare i byte risultanti in una stringa Python normale (UTF-8)
    # da passare al template HTML.
    plot_temporale_b64 = base64.b64encode(img1.getvalue()).decode()

    # ==========================================
    # GRAFICO 2: DISTRIBUZIONE
    # ==========================================
    fig2 = plt.figure(figsize=(10, 6))
    
    # Istogramma
    # Controllo per evitare errori se i dati sono troppo pochi per la KDE
    if len(rendimento_log) > 10:
        # 1. Disegna solo l'Istogramma
        sns.histplot(rendimento_log, kde=False, stat="density", color="skyblue", 
                     element="step", alpha=0.5, label="Distribuzione Rendimenti")
        
        # 2. Disegna il KDE separatamente per avere la sua label
        sns.kdeplot(rendimento_log, color="blue", linewidth=1, label="Distribuzione Kemel Density Estimation)")
        
        # 3. Normale Teorica (codice invariato)
        mu, std = stats.norm.fit(rendimento_log)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'r', linewidth=1, label="Distribuzione Normale")
        
    plt.title(f"{ticker} - Distribuzione dei Rendimenti")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Stessa procedura del grafico 1
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    plt.close(fig2)
    plot_distribuzione_b64 = base64.b64encode(img2.getvalue()).decode()

    # ==========================================
    # RETURN TEMPLATE
    # ==========================================
    return render_template(
        "grafico1.html", # Il nome del file HTML nella cartella 'templates'
        ticker=ticker, # Passiamo il simbolo (es. ENEL.MI) per scriverlo nel titolo della pagina
        # Passiamo le lunghe stringhe di testo (Base64) che il browser disegnerà come immagini.
        # Nell'HTML useremo <img src="data:image/png;base64,{{ plot_temporale }}">
        plot_temporale=plot_temporale_b64, 
        plot_distribuzione=plot_distribuzione_b64,
        # Ho ripassato le date scelte all'HTML.
        # In questo modo, quando la pagina si ricarica, l'utente vedrà ancora le date 
        # che ha selezionato nelle caselle di testo, invece di vederle resettate.
        start_date=start_date, 
        end_date=end_date
    )

if __name__ == "__main__":
    print("Avvio server Flask...")
    # Avvia il server di sviluppo integrato di Flask.
    # Parametri:
    # debug=True:  Attiva il debugger interattivo nel browser. Se c'è un errore,
    #                mostra lo stack trace completo invece di un errore generico.
    #
    # use_reloader=False: DISABILITA il riavvio automatico alla modifica del codice.
    # MOTIVO CRITICO: Il reloader di Flask crea un processo figlio che rieseguirebbe
    # tutto lo script dall'inizio. Poiché noi scarichiamo dati pesanti,
    # il reloader causerebbe un doppio scaricamento dei dati,
    # rallentando l'avvio e raddoppiando le chiamate.
    app.run(debug=True, use_reloader=False)
