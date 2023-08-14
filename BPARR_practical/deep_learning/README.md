# Řešení pomocí hlubokého učení

- jako metoda hlubokého učení byla vybrána segmentace s architekturou sítě U-Net

Popis skriptů:
- [deep_learning.py](./deep_learning.py) je kompletní řešení pomocí hlubokého učení včetně preprocessingu i postprocessingu
- [undistort_dataset.py](./undistort_dataset.py) je nástroj pro korekci obrázků pro následnou tvorbu datasetu
- [COCO_2_masks.py](./COCO_2_masks.py) je skript pro konvertaci masek v COCO formátu na masku s třídami s odpovídajícími hodnotami (třídy jsou značeny od 1, protože 0 je neoznačená třída pozadí)
- [model.py](./model.py) skript pro konstrukci samotné sítě U-Net
- [DatasetGenerator.py](./DatasetGenerator.py) třída pro načítání datasetu do U-Netu při trénování
- [train.py](./train.py) skript obsluhující trénování

> Datasety jsou uloženy ve složce [datasets](./datasets) a modely ve složce [models](./models).