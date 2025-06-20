'''Crea una lista di dizionari contenente dati anagrafici (nome, età, città).
Scrivi una funzione che:
- Restituisca la media dell'età,
- Restituisca il nome della persona più anziana,
- Restituisca una lista con tutti i nomi delle persone che vivono in una certa città.'''

an_list = [{"nome": "Anna", "età": 22, "città": "Milano"},
    {"nome": "Luca", "età": 30, "città": "Roma"},
    {"nome": "Giulia", "età": 27, "città": "Milano"},
    {"nome": "Marco", "età": 24, "città": "Napoli"}]

def calc(an_list):
    mean_age = 0
    sum = 0
    oldest_name = ""
    names_inthecity = []

    # Calcolo della media
    for person in an_list:
        sum += person["età"]
    mean_age = sum / len(an_list)

    # Individuazione della persona più anziana
    oldest_name = an_list[0]
    for person in an_list:
        if person["età"] > oldest_name["età"]:
            oldest_name = person    
    oldest_name = oldest_name["nome"]

    # Creazione della lista di chi vive in una determinata città
    city = input("Inserisci la città: ")
    city = city[0].upper() + city[1:] # tprima lettera maiuscola, il resto minuscolo
    for person in an_list:
        if person["città"] == city:
            names_inthecity.append(person["nome"])

    return mean_age, oldest_name, city, names_inthecity

mean, old, city, names = calc(an_list)
print(f"La media dell'età è {mean}")
print(f"La persona più anziana è {old}")
print(f"Nella città di {city} vivono {names}")