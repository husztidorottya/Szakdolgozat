# Szakdolgozat

# 1. simple_task.py
Ez a python script a megadott forrásfájlon feltanítja a modellt és a megadott néven lementi a betanított modellt.

Használata parancssorból:
    
    python3 simple_task.py task1.tsv trained_model_100_adam_idenitity
    
# 2. restore_model.py
  A script az inferencia megvalósítása. Megadott forrás szóalakon elvégzi a morfológiai tag-eknek megfelelő reinflexiós lépéseket. Végül szöveges formába visszakonvertálja a megjósolt cél szóalakot. Ehhez csak a betanított modell forrásfájlját kell megadni neki parancssori paraméterként, amit visszatölt.
  
  Használata parancssorból:
    
    python3 restore_model.py "államigazgatás N;IN+ESS;SG" trained_model_100_adam_identity.meta
    
# 3. test_accuracy.py
  A script tetszőlegesen lementett modellt képes tesztelni tetszőleg input fájlon és megadja, hogy az adott fájlon milyen accuracy-t sikerült elérni teljes szóegyezést vizsgálva. 
  
  Használata parancssorból:
    
    python3 test_accuracy.py task1_test.tsv trained_model_100_adam_identity.meta
    
# Forrásfájlok, amiket használhatsz:
- task1.tsv - tanító adat
- task1_test.tsv - teszt adat
