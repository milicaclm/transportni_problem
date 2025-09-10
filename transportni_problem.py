# %% [markdown]
# <a href="https://colab.research.google.com/github/milicaclm/transportni_problem/blob/main/transportni_problem_projekat.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import numpy as np
import pandas as pd

# %%
def izjednaci_kapacitet_i_potraznju(kapacitet, potražnja, cene_df):
    total_kapacitet = sum(kapacitet)
    total_potraznja = sum(potražnja)

    if total_kapacitet > total_potraznja:
        potražnja.append(total_kapacitet - total_potraznja)
        cene_df['*'] = 0

    elif total_potraznja > total_kapacitet:
        kapacitet.append(total_potraznja - total_kapacitet)
        cene_df.loc['*'] = 0

    return kapacitet, potražnja, cene_df

# %%
def severozapadni_ugao(kapacitet, potražnja):

    resenje = np.zeros((len(kapacitet), len(potražnja)))

    kap = kapacitet.copy()
    pot = potražnja.copy()

    i = 0
    j = 0

    while i < len(kapacitet) and j < len(potražnja):
        if kap[i] <= pot[j]:
            resenje[i][j] = kap[i]
            pot[j] -= kap[i]
            i += 1
        else:
            resenje[i][j] = pot[j]
            kap[i] -= pot[j]
            j += 1

    return resenje

# %%
def metod_najmanjih_cena(kapacitet, potražnja, cene):

    resenje = np.zeros((len(kapacitet), len(potražnja)))

    kap = kapacitet.copy()
    pot = potražnja.copy()

    cost_indices = [(cene[i][j], i, j) for i in range(len(kapacitet)) for j in range(len(potražnja))]
    cost_indices.sort()

    for cost, i, j in cost_indices:
        if kap[i] == 0 or pot[j] == 0:
            continue
        allocation = min(kap[i], pot[j])
        resenje[i][j] = allocation
        kap[i] -= allocation
        pot[j] -= allocation

    return resenje

# %%
def fogelov_metod(kapacitet, potražnja, costs):

    resenje = np.zeros((len(kapacitet), len(potražnja)))

    kap = kapacitet.copy()
    pot = potražnja.copy()
    cos = costs.copy()

    INF = int(1e10)

    while np.sum(kap) > 0 and np.sum(pot) > 0:

        redovni_penali = []
        for i in range(len(kap)):
            if kap[i] == 0:
                redovni_penali.append(-INF)
            else:
                red = sorted([costs[i][j] for j in range(len(pot)) if pot[j] > 0])
                if len(red) > 1:
                    redovni_penali.append(red[1] - red[0])
                else:
                    redovni_penali.append(red[0] if len(red) > 0 else -INF)

        kolonski_penali = []
        for j in range(len(pot)):
            if pot[j] == 0:
                kolonski_penali.append(-INF)
            else:
                kolona = sorted([cos[i][j] for i in range(len(kap)) if kap[i] > 0])
                if len(kolona) > 1:
                    kolonski_penali.append(kolona[1] - kolona[0])
                else:
                    kolonski_penali.append(kolona[0] if len(kolona) > 0 else -INF)

        max_red_penal = max(redovni_penali)
        max_kol_penal = max(kolonski_penali)

        if max_red_penal > max_kol_penal:
            i = redovni_penali.index(max_red_penal)
            j = np.argmin([cos[i][j] if pot[j] > 0 else INF for j in range(len(pot))])
        else:
            j = kolonski_penali.index(max_kol_penal)
            i = np.argmin([cos[i][j] if kap[i] > 0 else INF for i in range(len(kap))])

        allocation = min(kap[i], pot[j])
        resenje[i][j] = allocation
        kap[i] -= allocation
        pot[j] -= allocation

        if kap[i] == 0:
            for k in range(len(pot)):
                cos[i][k] = INF
        if pot[j] == 0:
            for k in range(len(kap)):
                cos[k][j] = INF

    return resenje

# %%
def iterativni_postupak(cene: np.ndarray, inicijalno_resenje: pd.DataFrame, ispisi_postupak: bool = False):
    u = [None for idx in inicijalno_resenje.index]
    v = [None for col in inicijalno_resenje.columns]
    degenerativne_nule = []
    i_min = None
    j_min = None
    c_prim = None
    resenje = inicijalno_resenje.where(inicijalno_resenje != 0, other=np.nan)
    indexi = inicijalno_resenje.index
    kolone = inicijalno_resenje.columns
    resenje = pd.DataFrame(resenje.values)
    pozitivne_indeksi = []
    najpopunjeniji_red = None  # Inicijalizacija sa prvim redom
    # Stavlja nulu u 'u' za najpopunjeniji red inicijalnog resenja (sto manje praznih polja)

    # c[i][j] == u[i] + v[j]
    def korak1():
        nonlocal u, v, najpopunjeniji_red, resenje, degenerativne_nule
        u = [np.nan] * inicijalno_resenje.shape[0]
        v = [np.nan] * inicijalno_resenje.shape[1]
        nove_degenerativne = []

        m, n = resenje.shape

        def broj_osnovnih_polja():
            return int(resenje.notna().to_numpy().sum())

        # Osiguraj tačno m+n-1 osnovnih polja i izbegni izolovane redove/kolone
        stepen_reda = resenje.notna().sum(axis=1).to_list()
        stepen_kolone = resenje.notna().sum(axis=0).to_list()

        # Priključi izolovane redove na neku neizolovanu kolonu
        for i in range(m):
            if broj_osnovnih_polja() >= m + n - 1:
                break
            if stepen_reda[i] == 0:
                for j in range(n):
                    if stepen_kolone[j] > 0 and pd.isna(resenje.iat[i, j]):
                        resenje.iat[i, j] = 0.0
                        stepen_reda[i] += 1
                        stepen_kolone[j] += 1
                        nove_degenerativne.append((i, j))
                        break

        # Priključi izolovane kolone na neki neizolovani red
        for j in range(n):
            if broj_osnovnih_polja() >= m + n - 1:
                break
            if stepen_kolone[j] == 0:
                for i in range(m):
                    if stepen_reda[i] > 0 and pd.isna(resenje.iat[i, j]):
                        resenje.iat[i, j] = 0.0
                        stepen_reda[i] += 1
                        stepen_kolone[j] += 1
                        nove_degenerativne.append((i, j))
                        break

        # Dodaj degenerativne nule dok ne dođemo do m+n-1; preferiraj red/kolonu sa malim stepenom
        while broj_osnovnih_polja() < m + n - 1:
            postavljeno = False
            for i in range(m):
                if postavljeno:
                    break
                for j in range(n):
                    if pd.isna(resenje.iat[i, j]) and (stepen_reda[i] <= 1 or stepen_kolone[j] <= 1):
                        resenje.iat[i, j] = 0.0
                        stepen_reda[i] += 1
                        stepen_kolone[j] += 1
                        nove_degenerativne.append((i, j))
                        postavljeno = True
                        break
            if not postavljeno:
                # Rezervni slučaj: bilo gde prazno
                zavrseno = False
                for i in range(m):
                    for j in range(n):
                        if pd.isna(resenje.iat[i, j]):
                            resenje.iat[i, j] = 0.0
                            stepen_reda[i] += 1
                            stepen_kolone[j] += 1
                            nove_degenerativne.append((i, j))
                            zavrseno = True
                            break
                    if zavrseno:
                        break

        # Izaberi početni red za potencijale
        najpopunjeniji_red = (resenje.notna().sum(axis=1)).idxmax()
        najpopunjeniji_red = resenje.index.get_loc(najpopunjeniji_red)
        u[najpopunjeniji_red] = 0.0

        if ispisi_postupak:
            print("resenje sa degenerativnim nulama:")
            print(resenje)

        # Izračunaj potencijale u i v BFS-om preko svih komponenti
        from collections import deque
        poseceni_redovi = [False] * m
        posecene_kolone = [False] * n

        def bfs_pocni_od_reda(r):
            red_cekanja = deque([('red', r)])
            while red_cekanja:
                tip, indeks = red_cekanja.popleft()
                if tip == 'red':
                    if poseceni_redovi[indeks]:
                        continue
                    poseceni_redovi[indeks] = True
                    for j in range(n):
                        if pd.notna(resenje.iat[indeks, j]):
                            if pd.isna(v[j]) and pd.notna(u[indeks]):
                                v[j] = cene[indeks, j] - u[indeks]
                            if not posecene_kolone[j]:
                                red_cekanja.append(('kolona', j))
                else:  # 'kolona'
                    if posecene_kolone[indeks]:
                        continue
                    posecene_kolone[indeks] = True
                    for i in range(m):
                        if pd.notna(resenje.iat[i, indeks]):
                            if pd.isna(u[i]) and pd.notna(v[indeks]):
                                u[i] = cene[i, indeks] - v[indeks]
                            if not poseceni_redovi[i]:
                                red_cekanja.append(('red', i))

        # Počni BFS od izabranog reda
        bfs_pocni_od_reda(najpopunjeniji_red)

        # Ako postoje nepovezane komponente, inicijalizuj ih sa 0 i propagiraj
        while (any(pd.isna(x) for x in u) or any(pd.isna(x) for x in v)):
            napredovalo = False
            for i in range(m):
                if stepen_reda[i] > 0 and pd.isna(u[i]):
                    u[i] = 0.0
                    bfs_pocni_od_reda(i)
                    napredovalo = True
                    break
            if not napredovalo:
                for j in range(n):
                    if stepen_kolone[j] > 0 and pd.isna(v[j]):
                        v[j] = 0.0
                        # Poveži preko bilo kog reda koji ima osnovno polje u toj koloni
                        for i in range(m):
                            if pd.notna(resenje.iat[i, j]):
                                bfs_pocni_od_reda(i)
                                break
                        napredovalo = True
                        break
            if not napredovalo:
                break

        if any(pd.isna(val) for val in u) or any(pd.isna(val) for val in v):
            print("v = ", v)
            print("u = ", u)
            print("najpopunjeniji red:", najpopunjeniji_red)
            print("popunjenost resenja:", resenje.notna().sum().sum())
            print("cene:")
            print(cene)
            raise ValueError("Nedostaju vrednosti u[i] ili v[j]")

        degenerativne_nule.extend(nove_degenerativne)
        if ispisi_postupak:
            print("KORAK 1:")
            print("najpopunjeniji_red:", najpopunjeniji_red)
            print("u=", u)
            print("v=", v)
            print("resenje:")
            print(resenje)
                            
        for _ in range(len(resenje.index) + len(resenje.columns)):
            for i in resenje.index:
                for j in resenje.columns:
                    i = resenje.index.get_loc(i)
                    j = resenje.columns.get_loc(j)
                    if pd.notna(resenje.iloc[i, j]):
                        if pd.isna(u[i]) and pd.notna(v[j]):
                            u[i] = cene[i, j] - v[j]
                        elif pd.isna(v[j]) and pd.notna(u[i]):
                            v[j] = cene[i, j] - u[i]
                if all(pd.notna(val) for val in u) and all(pd.notna(val) for val in v):
                    degenerativne_nule.extend(nove_degenerativne)
                    if ispisi_postupak:
                        print("KORAK 1:")
                        print("najpopunjeniji_red:", najpopunjeniji_red)
                        print("u=", u)
                        print("v=", v)
                        print("resenje:")
                        print(resenje)
                    return
        if any(pd.isna(val) for val in u) or any(pd.isna(val) for val in v):
            print("v = ", v)
            print("u = ", u)
            print("najpopunjeniji red:", najpopunjeniji_red)
            print("popunjenost resenja:", resenje.notna().sum().sum())
            print("cene:")
            print(cene)
            raise ValueError("Nedostaju vrednosti u[i] ili v[j]")

    def korak2():
        nonlocal j_min, i_min, pozitivne_indeksi, c_prim
        pozitivne_indeksi.clear()
        j_min = None
        i_min = None
        c_prim = np.zeros_like(cene)
        for i in resenje.index: 
            for j in resenje.columns:
                i = resenje.index.get_loc(i)
                j = resenje.columns.get_loc(j)
                if pd.isna(resenje.iloc[i, j]):
                    if pd.notna(u[i]) and pd.notna(v[j]):

                        c_prim[i, j] = cene[i, j] - u[i] - v[j]
                        if c_prim[i, j] < 0 and (i_min is None or j_min is None or c_prim[i, j] < c_prim[i_min, j_min]):
                            #print(f"Ažuriram i_min={i}, j_min={j}, c_prim[i, j]={c_prim[i, j]}")
                            i_min = i
                            j_min = j
                    else:
                        print("u=", u)
                        print("v=", v)

                        raise ValueError("Nedostaju vrednosti u[i] ili v[j] za indeks {} i kolonu {}".format(i, j))
                else:
                    pozitivne_indeksi.append((i, j))
        if ispisi_postupak:
            print("KORAK 2:")
            print("c_prim:")
            print(c_prim)
            print("i_min:", i_min, "j_min:", j_min)
            print("pozitivne_indeksi:", pozitivne_indeksi)
            print("cene:")
            print(cene)
        return (i_min is not None) and (j_min is not None)
    
    def korak3():
        nonlocal resenje, pozitivne_indeksi
        pozitivne_indeksi += [(i_min, j_min)]        

        def rekurzivna_pretraga(put, is_row):
            if len(put) > len(pozitivne_indeksi) + 1:
                return None 
            if (put[-1][0], put[-1][1]) == (i_min, j_min) and len(put) > 4 and len(put) % 2 != 0:
                return put[:-1]
                
            for i_sl, j_sl in pozitivne_indeksi:
                # Ne moze biti naredni cvor ako se do njega dolazi dijagonalno
                if i_sl != put[-1][0] and j_sl != put[-1][1]:
                    continue
                
                # Ne moze da bude već u putu osim ako nije početni čvor
                if (i_sl, j_sl) in put and (i_sl, j_sl) != (i_min, j_min):
                    continue
                # Nije vreme da se vrati u pocetni cvor kada nije potpuna kontura
                if (i_sl, j_sl) == (i_min, j_min) and len(put) < 4:
                    continue
                
                # Mora biti teme konture da bi se dodao u put
                if not is_row and i_sl == put[-1][0] and j_sl != put[-1][1]:
                    continue
                if is_row and j_sl == put[-1][1] and i_sl != put[-1][0]: 
                    continue
                    
                novi_put = put + [(i_sl, j_sl)]
                    
                rez = rekurzivna_pretraga(novi_put, not is_row)
                if rez is not None:

                    return rez
                
            return None

                    
        def novo_resenje(put):
            nonlocal resenje, degenerativne_nule
            min_oznaka = min(resenje.iloc[i, j] for i, j in put if put.index((i,j)) % 2 != 0)
            print("min_oznaka:", min_oznaka)
            resenje.iloc[i_min, j_min] = 0
            for i, j in put:
                if put.index((i,j)) % 2 != 0:
                    resenje.iloc[i, j] += -min_oznaka
                    if resenje.iloc[i, j] == 0:
                        resenje.iloc[i, j] = np.nan
                    if resenje.iloc[i, j] < 0:
                        raise ValueError(f"Negativna vrednost u resenju na indeksu ({i}, {j}): {resenje.iloc[i, j]}")
                elif put.index((i,j)) % 2 == 0:
                    resenje.iloc[i, j] += min_oznaka
                    if (i,j) in degenerativne_nule and (min_oznaka > 0):
                        degenerativne_nule.remove((i, j))
            

        
        put = rekurzivna_pretraga([(i_min, j_min)], True) or rekurzivna_pretraga([(i_min, j_min)], False)
        if ispisi_postupak:
            print("KORAK 3:")        
            print("Pronađen put:", put)
            print("Novo rešenje:")
            print(resenje)
        
        if put is None:
            print("Nije pronađen validan put za optimizaciju.")
            return False
        else:
            novo_resenje(put)
            return True
        
    count = 0
    while True:
        if ispisi_postupak: 
            print(f"ITERACIJA {count + 1}")
        korak1()
        if not korak2():
            resenje = resenje.fillna(0)
            resenje.index = indexi
            resenje.columns = kolone
            return resenje
        if not korak3():
            return None
        count += 1
        if count > 10:
            raise ValueError("Previše iteracija, proverite inicijalno rešenje.")

# %%    
if __name__ == "__main__":
    kapacitet = [20, 30, 25]
    potražnja = [10, 15, 25, 25]

    cene = np.array([
        [8, 6, 10, 9],
        [9, 7,  4, 2],
        [3, 4,  2, 5]
    ])

    cost_df = pd.DataFrame(cene, columns=['D1', 'D2', 'D3', 'D4'], index=['F1', 'F2', 'F3'])

    
    """kapacitet = [20,30, 20, 10]
    potražnja = [10, 40, 30]

    cene = np.array([
        [10, 12, 0],
        [8, 4, 3],
        [6, 9, 4],
        [7, 8, 5]
    ])

    cost_df = pd.DataFrame(cene, columns=['D1', 'D2', 'D3'], index=['F1', 'F2', 'F3', 'F4'])"""
    

    kapacitet, potražnja, cene_df = izjednaci_kapacitet_i_potraznju(kapacitet, potražnja, cost_df)

    print("Kapaciteti (Supply):", kapacitet)
    print("Zahtevi (Demand):", potražnja)
    print("Troškovi transporta (Costs):")
    print(cost_df)

    resenje_severozapadni = severozapadni_ugao(kapacitet, potražnja)
    resenje_severozapadni_df = pd.DataFrame(resenje_severozapadni, columns=cost_df.columns, index=cost_df.index)

    print("Inicijalno rešenje (Severozapadni ugao):")
    print(resenje_severozapadni_df)

    resenje_najmanje_cene = metod_najmanjih_cena(kapacitet, potražnja, cene)
    resenje_najmanje_cene_df = pd.DataFrame(resenje_najmanje_cene, columns=cost_df.columns, index=cost_df.index)

    print("Inicijalno rešenje (Metod najmanjih cena):")
    print(resenje_najmanje_cene_df)

    resenje_fogelov = fogelov_metod(kapacitet, potražnja, cene)
    resenje_fogelov_df = pd.DataFrame(resenje_fogelov, columns=cost_df.columns, index=cost_df.index)

    print("Inicijalno rešenje (Fogelov metod):")
    print(resenje_fogelov_df)

    tabela_resenje_severozapadni = iterativni_postupak(cene, resenje_severozapadni_df, True)
    print("Optimalno rešenje (Severozapadni ugao):")
    print(tabela_resenje_severozapadni)
    tabela_resenje_najmanje_cene = iterativni_postupak(cene, resenje_najmanje_cene_df)
    print("Optimalno rešenje (Metod najmanjih cena):")
    print(tabela_resenje_najmanje_cene)
    tabela_resenje_fogelov = iterativni_postupak(cene, resenje_fogelov_df)
    print("Optimalno rešenje (Fogelov metod):")
    print(tabela_resenje_fogelov)
