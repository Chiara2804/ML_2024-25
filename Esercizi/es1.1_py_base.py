'''Scrivi una funzione che prenda in input una stringa e restituisca:
-Il numero di parole,
-Il numero di caratteri totali (esclusi gli spazi),
-Le 3 parole più frequenti.'''

def fun(input):
    n_words = 0
    tot_chars = 0
    frequent_words = ['', '', '']

    n_words = input.count(' ') + 1
    tot_chars = len(input)
    
    # Individuo le parole più frequenti
    input = input.lower() # rendo tutte le parole minuscole
    single_words = input.split() # divido l'input in singole parole
    frequences = {} # dizionario vuoto per contare le parole

    for word in single_words: 
        if word in frequences:
            frequences[word] += 1
        else:
            frequences[word] = 1

    ordered_words = sorted(frequences.items(), key=lambda x: x[1], reverse=True) # Ordina il dizionario in base alla frequenza, in ordine decrescente
    frequent_words = ordered_words[:3]

    return n_words, tot_chars, frequent_words


inp = input("Inserisci una stringa: ")
w, tot, freq = fun(inp)
print(f"Il numero di parole è {w}")
print(f"Il numero di caratteri totali (spazi esclusi) è {tot - w}")
print(f"Le tre parole più frequenti sono: {freq}")