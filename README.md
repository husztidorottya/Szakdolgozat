# Szakdolgozat
A nyelvfüggetlen morfológiai reinflexió megvalósítását végző rendszer forráskódját tartalmazó repository. A futtatásához mindenképpen szükséges a Tensorflow könyvtár feltelepítése, valamint ajánlott a Python3.6.0 használata.

## 1. morphological_reinflection.py
  Ez a python script a megadott forrásfájlon véletlen paraméterösszeállítású, megadott számú kísérletet végez. Feltanítja a modellt és lementi a betanított modelleket az általuk használt paraméterkombinációkkal. A paramétereket a parameters, míg a betanított modelleket a trained_models mappákba szervezi.
Amennyiben a modell számára magyar nyelven legnagyobb pontosságot biztosító paraméterkombinációt szeretné kipróbálni, ebben az esetben a kódban található erre vonatkozó kikommentezett sort kell feloldania. 

Használata parancssorból:
    
    python3 morphological_reinflection.py task1.tsv 1
    
A feltanított modellek veszteségének alakulását a Tensorboard segítségével vizualizáltam, melyet skalár diagram és hisztogram formájában is megtekinthet. A diagramokat a feltanítás sorrendjében sorszámmal látja el, így modellenként külön-külön megtekinthető. Ehhez a fájlokat az output mappában tárolja.

Vizualizáció megtekintése:
    
    tensorboard --logdir=output

## 2. inference.py
  A script az inferencia megvalósítása. Megadott forrás szóalakon elvégzi a morfológiai tag-eknek megfelelő reinflexiós lépéseket. Végül szöveges formába visszakonvertálja a megjósolt célszóalakot. Ehhez csak a betanított modell forrásfájlját kiterjesztés nélkül kell megadni  parancssori paraméterként, amit visszatölt.
  
  Használata parancssorból:
    
    python3 inference.py "államigazgatás<tab>N;IN+ESS;SG" trained_model0
    
## 3. test_accuracy.py
  A script tetszőlegesen lementett modellt képes tesztelni tetszőleges input fájlon és megadja, hogy az adott fájlon milyen pontosságot sikerült elérni teljes szóegyezést vizsgálva. 
  
  Használata parancssorból:
    
    python3 test_accuracy.py task1_test.tsv trained_model0
    
## Tanító és teszt adathalmaz:
  A kód kipróbálásához az adathalmazokat kérésre továbbítom.
