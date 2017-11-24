# Szakdolgozat

## 1. morfologiai_reinflexio.py
Ez a python script a megadott forrásfájlon véletlen paraméterösszeállítású, megadott számú kísérletet végez. Feltanítja a modellt és lementi a betanított modelleket.

Használata parancssorból:
    
    python3 morfologiai_reinflexio.py task1.tsv 1
    
## 2. inferencia.py
  A script az inferencia megvalósítása. Megadott forrás szóalakon elvégzi a morfológiai tag-eknek megfelelő reinflexiós lépéseket. Végül szöveges formába visszakonvertálja a megjósolt cél szóalakot. Ehhez csak a betanított modell forrásfájlját kell megadni neki parancssori paraméterként, amit visszatölt.
  
  Használata parancssorból:
    
    python3 inferencia.py "államigazgatás<tab>N;IN+ESS;SG" trained_model0
    
## 3. test_accuracy.py
  A script tetszőlegesen lementett modellt képes tesztelni tetszőleges input fájlon és megadja, hogy az adott fájlon milyen accuracy-t sikerült elérni teljes szóegyezést vizsgálva. 
  
  Használata parancssorból:
    
    python3 test_accuracy.py task1_test.tsv trained_model0
    
## Forrásfájlok, amiket használhatsz:
- task1.tsv - tanító adat
- task1_test.tsv - teszt adat
